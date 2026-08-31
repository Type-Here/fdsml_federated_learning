# Smoke test

A two-run sanity check on GTSRB, meant for Colab with a GPU. It is **not** an
experiment: it subsamples the dataset and trains for two rounds. Its only job is
to prove that the plumbing added on `features/common-ground` actually executes,
because none of it can run on a machine without torch.

## What it exercises

| Piece | Why it needs a real run |
|---|---|
| B1 - client identity | `_on_client_ready` / `_on_client_update` are Socket.IO handlers; nothing about them is reachable from a unit test |
| B2 - denominator | The `aggregation_denominator` round trip, including round 0 sending 0 |
| B3 - Dirichlet split | The partition on the real GTSRB directory tree, not a synthetic fixture |
| A1 - encrypted dispatch | `ExtendedAggregator.aggregate_encrypted_updates` has **never been executed** |
| A4 / A6 | `aggregate_train_loss` is still only reachable via `weighted_aggregation: false`; seeding runs per config |

Two configurations run sequentially: `no_encryption`, then
`direct_encrypted_update`.

## Sizing

`smoke_config.json` keeps it small on purpose:

- `max_units_per_class: 2` - 2 tracks x 30 frames x 43 classes = **2580 images**
  instead of 26640, so the per-client file copy is quick.
- `num_custom_layers: 1` - the trainable head is `Linear(512 -> 43)`, about
  **22k parameters**. This matters for the encrypted run: Paillier encrypts one
  ciphertext per parameter, per client, per round. The production head
  (`num_custom_layers: 2`, ~142k parameters) is roughly 6x that cost, which
  turns a sanity check into a coffee break.
- `global_epoch: 2`, `local_epoch: 1`, `num_clients: 4`.
- `num_parallel_executions: 1` - one FL simulation at a time. Colab has a single
  GPU; the default of 12 concurrent runs would exhaust its memory.
- Separate output paths (`csv_smoke/`, `logs_smoke/`, `run_dataset_smoke/`) so
  smoke results never land in the real results CSV. This is not cosmetic: the
  grid search deduplicates against that CSV by config fingerprint, so a smoke
  run written there could cause a real run to be skipped later.

## Running it on Colab

```python
# 1. Clone and enter the repo
!git clone <repo-url> fdsml && cd fdsml

# 2. Install only what Colab is missing. Do NOT install requirements_gpu.txt
#    as-is: re-pinning numpy==1.26.4 breaks the preinstalled CUDA torch.
!pip install -q phe gmpy2 flask-socketio python-socketio eventlet

# 3. Build the dataset (downloads ~200 MB, writes dataset/gtsrb/train/00000..00042)
!python datasets_prep/prepare_gtsrb.py --splits train

# 4. Run the smoke
!python federated_grid_search.py smoke_config.json
```

Step 3 takes a few minutes and only has to be done once per session. If the
Colab runtime disconnects, everything under `dataset/` is lost and step 3 has to
be repeated.

## What "passed" looks like

1. **The process exits on its own**, printing
   `=== All parallel executions have finished. ===`. A hang is the most likely
   failure mode: it means the server is waiting for an update from a client
   that never registered, i.e. the B1 identity handling is wrong.
2. **`run_dataset_smoke/.../partition_report.csv` exists**, with 4 rows, and its
   `n_images` column sums to 2580. The `d_k` column should be well above 0 - at
   `alpha=0.5` over 43 classes the clients are genuinely skewed.
3. **`csv_smoke/` contains two result rows**, one per encryption mode, and they
   carry the new columns: `partition_strategy`, `dirichlet_alpha`,
   `partition_unit`, `seed`, `max_units_per_class`.
4. **In the client logs** (`logs_smoke/<ddmm>/FL-Client-LOG/`), each round after
   the first logs `Rescaling server payload by denominator <N>` with N equal to
   the round's total training size. Round 0 logs
   `Aggregation denominator is 0. Using weights as is.` - that one is expected
   and correct, because round 0 carries the initial weights, not a sum.
5. **The two encryption modes reach a comparable validation F1.** They are not
   required to be bit-identical (the encrypted path quantises), but a large gap
   means the encrypted dispatch is aggregating differently from the plaintext
   one, which is exactly the A1 failure we are checking for.

## If it is too slow

Drop `"direct_encrypted_update"` from `encryption_mode` and run the plaintext
check first. The encrypted run is the slow one and the only one whose cost grows
with the head size.

## What it does NOT cover

- FedDisco and FIPA: neither exists yet, so `client_denominator` returning `1.0`
  for them is unit-tested but never exercised end to end.
- `weighted_aggregation: false`, which is the only way to reach
  `aggregate_train_loss` (A4).
- The global checkpoint (B6), which is not implemented.