# Smoke test - FIPA, plaintext and encrypted

A two-run sanity check on GTSRB, for Colab with a GPU. Like its sibling
`SMOKE_TEST_1.md` it is **not** an experiment: it subsamples the dataset and
trains for three rounds. Its job is to make FIPA actually execute - the previous
smoke ran `FedAvg`, so it proved the transport *around* FIPA and not one line of
FIPA itself.

The two runs are the same configuration under `no_encryption` and then
`direct_encrypted_update`, and **comparing them is the point**: the encrypted
route is designed to produce the same numbers as the plaintext one, so a gap
between the two is the failure this test is looking for.

## What it exercises, that nothing else can

Everything below needs torch and a dataset, so no unit test reaches it.

| Piece | Why it needs a real run |
|---|---|
| The gradient collection pass | `ExtendedModelManager.collect_gradient_factors` builds the gradient matrix `G` with a forward/backward sweep over the client's data. Never executed |
| `explained_variance_ratio` on real data | The number that turns `fipa_rank = 5` from a value someone picked into a defensible choice. **Never measured** |
| The delta round trip, plaintext | The client sends `theta_local - theta_global`, the server adds the preconditioned sum back onto its snapshot |
| The projection round trip, encrypted | The client sends `Enc(U^T Delta)` - 5 ciphertexts - and the whole delta stays at home |
| The one-off rescale at the warmup boundary | The server holds `Enc(N x theta)` and must multiply homomorphically by `1/N` before adding an increment |
| The payload-kind guard | `weights` / `delta` / `fipa_z` are three different things of the same type; the server refuses a round that mixes them |

## The round schedule, which is the design of this config

`global_epoch: 3` with `fipa_warmup_rounds: 1`. That is the smallest run that
reaches **both** encrypted code paths:

| round | rule | what the payload is | what the server does with it |
|---|---|---|---|
| 0 | FedAvg (warmup) | the initial parameters | ordinary weighted sum. Clients send absolute weights |
| 1 | **FIPA** | still the FedAvg sum, `N x theta` | **rescales by `1/N` onto the fixed-point grid**, then adds the increment |
| 2 | **FIPA** | the finished model, denominator 1.0 | adds the increment, no rescale |

Round 1 and round 2 are not the same test. Round 1 is the only round in the whole
run where the model has to be moved onto the grid; round 2 is the one that shows
the representation then stays put. A run with `global_epoch: 2` would exercise
the first and never the second.

Keep `fipa_warmup_rounds < global_epoch`. A warmup at least as long as the run
makes every round FedAvg, and the CSV row still says FIPA.

## Sizing, and what the encrypted run actually costs

`smoke_config_fipa.json` is the previous smoke config with FIPA switched on:

- `max_units_per_class: 2` - 2 tracks x 30 frames x 43 classes = **2580 images**.
- `num_custom_layers: 1` - the trainable head is `Linear(512 -> 43)`, so
  **p = 22059 parameters**.
- `num_clients: 4`, `models_percentage: 1` (every client every round),
  `batch_size: 16`, `local_epoch: 1`.
- `num_parallel_executions: 1` - one simulation at a time, for Colab's single GPU.
- Separate output paths. Not cosmetic: the grid search deduplicates against the
  results CSV by config fingerprint, so a smoke row written into the real one
  could cause a real run to be skipped later. The machine's hostname is appended
  to most of them, so with `PCNAME` = `<pc>` the run writes:

  | what | where |
  |---|---|
  | per-round metrics | `csv_smoke_fipa_<pc>/runs/<timestamp>_gtsrb_smoke_fipa_worker_0.csv` |
  | one row per run | `csv_smoke_fipa/<pc>/federated_grid_search_results_<pc>.csv` |
  | client logs | `logs_smoke_fipa_<pc>/<run>/<ddmm>/FL-Client-LOG/` |
  | server log | `logs_smoke_fipa_<pc>/<run>/<ddmm>/FL-Server-LOG/` |
  | per-client data | `run_dataset_smoke_fipa_<pc>/` |

  The two CSV paths are spelled differently - suffix for one, subdirectory for
  the other - in the received `federated_grid_search.py`. That is not a typo
  here.

**Where the homomorphic cost goes, and why it is not frightening here.** Per
round the server does `p x r x M` ciphertext multiplications - each ciphertext
multiplied exactly once, which is the whole point of the fused operator:

```
22059 params x 5 directions x 4 clients = 441k multiplications  ~ 20 s / round
```

against the warmup round, which is an ordinary encrypted FedAvg:

```
clients: 4 x 22059 encryptions   ~ 6 s      server: 88k operations  ~ 4 s
```

So the two encrypted FIPA rounds are the expensive part, and they are under a
minute. **The client side is free**: 5 encryptions instead of 22059. That
inversion - the server pays more, the clients stop paying at all - is the
headline number of the encrypted design, and this run is where to check it
against a clock rather than against a spreadsheet.

`fipa_grad_batches: 64` caps the extra gradient pass. It does not bind here: the
smallest client holds a few hundred images at batch 16, so a few dozen batches.
It also has to be at least `fipa_rank` - asking for 5 directions from 3 gradients
is not a crash (the rank is clamped) but it is a meaningless estimate.

## Running it on Colab

```python
# 1. Clone the working branch EXPLICITLY. On main, main() takes no argument and
#    the config on the command line is ignored - the inherited grid runs instead.
!git clone -b features/fipa <repo-url> fdsml
%cd fdsml

# 2. Install only what Colab is missing. Do NOT install requirements_gpu.txt
#    as-is: re-pinning numpy==1.26.4 breaks the preinstalled CUDA torch.
!pip install -q phe flask-socketio python-socketio eventlet

# 3. Build the dataset (downloads ~200 MB, writes dataset/gtsrb/train/00000..00042)
!python datasets_prep/prepare_gtsrb.py --splits train

# 4. Run the smoke
!python federated_grid_search.py smoke_config_fipa.json
```

Use `%cd` and not `!cd`: `!cd` opens a subshell that dies with the cell. Step 3
takes a few minutes, once per session - a runtime disconnect loses `dataset/`.

## What "passed" looks like

**1. The process exits on its own**, printing
`=== All parallel executions have finished. ===`. A hang means the server is
waiting for an update from a client that never sent one - look in the client log
for an exception inside the worker thread, which is caught and logged rather than
raised.

**2. The client log shows FIPA rounds, and only after the warmup.**
In the client logs, one line per client for rounds 1 and 2 and none for round 0:

```
Round 1 is a FIPA refinement round: collecting curvature factors.
```

If round 0 has it too, the warmup boundary is off by one - which produces no
error anywhere, only a model preconditioned by curvature measured at
initialization.

**3. In the encrypted run, the client encrypts 5 numbers and not 22059.**
This is the single most informative line in the whole log:

```
Encrypting the 5-value curvature projection instead of the 22059-parameter delta.
```

If instead the encrypted run shows `Starting encryption of 2 layers...` on a FIPA
round, the client encrypted the whole parameter vector and then threw it away -
the branch ordering in `_update_worker` is wrong.

**4. In the encrypted run, the model is put on the grid exactly once.**
In the server log, at round 1 and nowhere else:

```
Rescaling the encrypted model by 1/<N> onto the fixed-point grid.
```

`<N>` is the round's total training size, the sum of the four clients' train
sizes. Twice means round 2 rescaled a model that was already the finished one and
shrank it by `N`; never means round 1 added an increment to `N x theta`, and
every subsequent round trains from a model scaled by ~20000. Both look like
divergence, not like a bug.

**5. `fipa_explained_variance` is in the per-round metrics, and is not nan.**
The per-round CSV (`csv_smoke_fipa_<pc>/runs/...`) carries one column per metric
per round. Rounds 1 and 2 must have a number in `fipa_explained_variance`; round
0 will not, because the warmup collects no curvature.

**This is the first measurement of it, and it is a result, not a check.** It says
what share of the gradients' variance the 5 kept directions account for. A high
value makes `fipa_rank = 5` defensible; a low one says FIPA is discarding signal
rather than tail, and that the rank has to go up. Write down whatever it says -
including if it is disappointing.

**6. The two encryption modes agree.** Compare the per-round CSVs of the two runs
on rounds 1 and 2, not just the best-model row.

The expectation is **agreement to several decimals, not a gap**. Paillier is not
lossy here: `phe` writes a float as an integer times a power of 16, and the fixed
point pins that power on both sides of every multiplication, so the encrypted
arithmetic reproduces the plaintext one to about 1e-13 relative - four orders
below the float32 the weights travel in. *(The older `SMOKE_TEST_1.md` says the
encrypted path "quantises" and to expect a gap. It does not, at this scale.)*

A visible gap means one of: the rescale of criterion 4, the pairing of each
client's preconditioner with its own projection, or the fixed-point grid. A gap
of exactly a factor of `N` on the weights points at the first.

**Check `best_round` before reading anything into the best-model row.** If it is
0, the best model is the warmup round and is identical in both modes by
construction - the comparison then says nothing and the per-round CSV is the only
evidence.

**7. The partition report is unchanged** from the previous smoke: 4 rows summing
to 2580 images, `d_k` well above 0 at `alpha = 0.5` over 43 classes.

## If something fails

- **`ValueError: FIPA expects updates of kind 'fipa_z', but client ... sent 'delta'`**
  - the client and the server disagree about whether the round is encrypted. The
  guard did its job: this would otherwise have aggregated a plaintext delta as if
  it were a curvature projection.
- **`ValueError: ... rounds to zero on the fixed-point grid`** - a whole
  preconditioner fell below `16^-13`. Real information, not a nuisance: it means
  the increment for that round would have been exactly zero.
- **`ValueError: ... past the Paillier ceiling`** - the opposite end. Crossing it
  silently is the failure the fixed point exists to prevent, so this raising is
  the design working.
- **Too slow** - drop `"direct_encrypted_update"` and run the plaintext FIPA
  check alone first. The plaintext run costs nothing extra beyond the gradient
  collection pass.

## What it does NOT cover

- FedDisco, which does not exist yet.
- `weighted_aggregation: false`, still the only route to the unweighted training
  loss aggregation.
- The global model checkpoint and its BatchNorm recalibration pass.
- `ConvNet`, which trains its whole network rather than a frozen backbone plus a
  head, so its parameter count - and therefore the encrypted server cost - is an
  order of magnitude larger. Worth a separate run before trusting the cost
  figures for the real grid.