# Smoke test — FedAvg, plaintext and encrypted

**Why it exists.** Everything the federated pipeline does that touches torch is
unreachable from a unit test: the Socket.IO handlers, the Dirichlet split on the
real directory tree, the encrypted dispatch. This run proves that plumbing
executes. It subsamples the dataset and trains for two rounds — **it is not an
experiment and its accuracy numbers mean nothing.**

**What it runs.** `smoke_config.json`, twice in sequence: `no_encryption`, then
`direct_encrypted_update`. GTSRB, ResNet18, 4 clients, `global_epoch: 2`,
`dirichlet_alpha: 0.5` on tracks.

Kept small on purpose: `max_units_per_class: 2` (2 tracks x 30 frames x 43
classes = **2580 images**), `num_custom_layers: 1` (head is `Linear(512->43)`,
~22k parameters — the encrypted run costs one ciphertext per parameter per client
per round), `num_parallel_executions: 1` for Colab's single GPU. Output goes to
`csv_smoke/`, `logs_smoke/`, `run_dataset_smoke/` so a smoke row never lands in
the real results CSV — the grid deduplicates against that file by fingerprint, and
a stray row there would cause a real run to be skipped later.

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
!python federated_grid_search.py smoke_config.json
```

Use `%cd`, not `!cd` — `!cd` opens a subshell that dies with the cell. Step 3 takes
a few minutes, once per session; a runtime disconnect loses `dataset/`.

## What "passed" looks like

1. **The process exits on its own**, printing
   `=== All parallel executions have finished. ===`. A hang means the server is
   waiting on a client that never registered.
2. **`run_dataset_smoke/.../partition_report.csv`** has 4 rows, `n_images` summing
   to **2580**, and `d_k` well above 0.
3. **`csv_smoke/` has two result rows**, one per encryption mode, carrying
   `partition_strategy`, `dirichlet_alpha`, `partition_unit`, `seed`,
   `max_units_per_class`.
4. **In the client logs**, every round after the first logs
   `Rescaling server payload by denominator <N>` with `N` the round's total
   training size. Round 0 logs `Aggregation denominator is 0. Using weights as is.`
   — expected: round 0 carries the initial weights, not a sum.
5. **The two encryption modes agree to several decimals.** Paillier is not lossy
   here: `phe` reproduces the plaintext arithmetic to ~1e-15 relative. A visible
   gap means the encrypted path is aggregating by a different rule than the
   plaintext one.

Too slow? Drop `"direct_encrypted_update"` and run the plaintext check first — the
encrypted run is the slow one, and the only one whose cost grows with head size.
