# Smoke test — FIPA, plaintext and encrypted

**Why it exists.** The FedAvg smoke test proved the transport *around* FIPA and not
one line of FIPA itself. This one makes FIPA execute: the gradient collection pass,
the low-rank factors, the delta round trip, and the encrypted projection route —
none of which a unit test reaches, because all of them need torch and a dataset.
Three rounds on a subsample: **not an experiment.**

**What it runs.** `smoke_config_fipa.json`, twice: `no_encryption`, then
`direct_encrypted_update`. **Comparing the two is the point** — the encrypted route
is designed to produce the same numbers as the plaintext one, so a gap between them
is the failure this test looks for.

**The round schedule is the design of this config.** `global_epoch: 3` with
`fipa_warmup_rounds: 1` is the smallest run that reaches both encrypted paths:

| round | rule | payload | what the server does |
|---|---|---|---|
| 0 | FedAvg (warmup) | initial parameters | ordinary weighted sum |
| 1 | **FIPA** | still the FedAvg sum, `N x theta` | **rescales by `1/N` onto the fixed-point grid**, then adds the increment |
| 2 | **FIPA** | the finished model, denominator 1.0 | adds the increment, no rescale |

Round 1 is the only round where the model has to be moved onto the grid; round 2
shows the representation then stays put. `global_epoch: 2` would exercise the first
and never the second. Keep `fipa_warmup_rounds < global_epoch`, or every round is
FedAvg while the results row still says FIPA.

Sizing: `max_units_per_class: 2` (**2580 images**), `num_custom_layers: 1`
(**p = 22 059**), 4 clients, `fipa_rank: 5`, `num_parallel_executions: 1`, separate
output paths under `csv_smoke_fipa*` / `logs_smoke_fipa*` / `run_dataset_smoke_fipa*`.

## Cells

```python
# 1. Clone the working branch EXPLICITLY. On main, main() takes no argument and
#    the config on the command line is ignored - the inherited grid runs instead.
!git clone -b features/feddisco <repo-url> fdsml
%cd fdsml

# 2. Install only what Colab is missing. Do NOT install requirements_gpu.txt
#    as-is: re-pinning numpy==1.26.4 breaks the preinstalled CUDA torch.
!pip install -q phe gmpy2 flask-socketio python-socketio eventlet

# 3. Build the dataset (downloads ~200 MB, writes dataset/gtsrb/train/00000..00042)
!python datasets_prep/prepare_gtsrb.py --splits train

# 4. Run both configurations
!python federated_grid_search.py smoke_config_fipa.json
```

Use `%cd`, not `!cd`. Install `gmpy2`: `phe` uses it for modular exponentiation if
present and **falls back to pure Python in silence** if not — one encrypted round
measured 135 s without it against 16 s with.

## What "passed" looks like

1. **The process exits on its own.** A hang means a client never sent its update —
   look in the client log for an exception inside the worker thread, which is
   caught and logged rather than raised.
2. **FIPA rounds appear only after the warmup.** In the client logs, for rounds 1
   and 2 and not round 0:
   `Round 1 is a FIPA refinement round: collecting curvature factors.`
   Round 0 having it too means the warmup boundary is off by one, which raises
   nothing and produces a model preconditioned by curvature measured at
   initialisation.
3. **In the encrypted run the client encrypts 5 numbers, not 22 059:**
   `Encrypting the 5-value curvature projection instead of the 22059-parameter delta.`
   `Starting encryption of 2 layers...` on a FIPA round means the client encrypted
   the whole parameter vector and then threw it away.
4. **The model is put on the grid exactly once.** In the server log, at round 1 and
   nowhere else: `Rescaling the encrypted model by 1/<N> onto the fixed-point grid.`
   Twice shrinks the finished model by `N`; never leaves every later round training
   from a model scaled by ~20 000. Both look like divergence, not like a bug.
5. **`fipa_explained_variance` is in the per-round CSV and is not nan** for rounds 1
   and 2, and blank for round 0 (the warmup collects no curvature).
6. **The two encryption modes agree to several decimals** on rounds 1 and 2 — the
   fixed point pins the exponent on both sides of every multiplication, so the
   encrypted arithmetic reproduces the plaintext to ~1e-13 relative. Compare the
   per-round CSVs, not just the best-model row: if `best_round` is 0 the best model
   is the warmup and is identical in both modes by construction.
7. **The partition report** is unchanged from the FedAvg smoke: 4 rows summing to
   2580 images.

## If something fails

| error | meaning |
|---|---|
| `FIPA expects updates of kind 'fipa_z', but client ... sent 'delta'` | client and server disagree about whether the round is encrypted. The guard did its job — this would otherwise have aggregated a plaintext delta as a curvature projection |
| `... rounds to zero on the fixed-point grid` | a whole preconditioner fell below `16^-13`: the increment for that round would have been exactly zero |
| `... past the Paillier ceiling` | the opposite end. Crossing it *silently* is the failure the fixed point exists to prevent, so this raising is the design working |
| too slow | drop `"direct_encrypted_update"`; plaintext FIPA costs nothing beyond the gradient pass |
