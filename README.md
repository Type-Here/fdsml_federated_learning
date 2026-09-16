# Federated Learning under Distribution Shift

Robustness of a federated image classifier to distribution shift, across the whole
model lifecycle — **shift at training time** (clients hold non-IID data) and
**shift at inference time** (the deployed model sees corrupted input). One dataset,
one model, one checkpoint joining the two halves.

> **Provenance.** The core federated framework — Flask/SocketIO server, threaded
> clients, Paillier trusted authority, grid-search orchestrator — was received from
> Lab 8 at Unisa (see the initial commit) and is **not** our work. Ours are the
> aggregation rules, the non-IID partitioning, the test-time adaptation package,
> and the fixes listed under [Fixes to the received code](#fixes-to-the-received-code).
> The received files keep their original Italian comments; everything we added is
> in English.

---

## Status

| | what it is | state |
|---|---|---|
| **FedAvg / FedProx** | baselines, received | run |
| **FedDisco** | aggregation weight discounted by label skew | code complete and unit-tested, **not yet run on a GPU** |
| **FIPA** | per-parameter aggregation weights from Fisher curvature | run, plaintext and encrypted — **measured, and it loses to FedAvg** |
| **Test-time adaptation** | BatchNorm adaptation + a bank of corruption-specific states | **complete and measured** |
| Full experiment grid | 60 runs, ~111 h | in progress |

249 tests, all runnable without a GPU: `pytest tests/ -q`.

---

## Results

Numbers are on GTSRB, ResNet18 with a frozen ImageNet backbone and a trained
2-layer head (142 379 trainable parameters), images at 128 px.

### Test-time adaptation — the finished half

Four arms over a stream of 49 corrupted conditions plus clean, 100 000 images each,
batch 128:

| arm | what it does | accuracy | vs. `source` |
|---|---|---|---|
| `source` | nothing | 0.3018 | — |
| **`blind`** | normalises on the batch in hand | **0.4224** | **+12.06 pts** |
| `routed` | picks a stored state, or refuses | 0.4153 | +11.35 pts |
| `oracle` | is handed the true corruption's state | 0.3997 | +9.78 pts |

```
blind  - source   +0.1206  +-0.0088   significant, and large
routed - blind    -0.0071  +-0.0095   not significant
oracle - blind    -0.0228  +-0.0093   significant, and the wrong sign
```

**Blind BatchNorm adaptation is the method.** It costs no stored state and
+0.02 ms/image. The bank of corruption-specific states is a **measured negative
result**: even the oracle — the ceiling of any routing scheme — sits below it. At
the exact severity the states were built for, the bank *ties* an estimate made on
the spot from 128 images. The routing itself works (0.786 hit rate where the bank
is valid), which is what makes the negative a statement about the bank rather than
about the router.

Two numbers that must travel together: recalibrating the BatchNorm statistics on
clean GTSRB data is worth **+25.8 points on clean images** and **−10.9 on noise**.
Statistics tuned tightly to one distribution are more fragile under shift.

Repeating the whole run at batch 16 flips the middle result and not the conclusion:
the bank then beats the on-the-spot estimate by 2.1 points where it is valid
(11 of 12 conditions, sign test p = 0.003), but the descriptor that has to *find*
the right state is read off the same 16 images, so the calibrated gate widens 5x
and `routed − blind` gets worse, not better.

### FIPA — measured, and negative

At `dirichlet_alpha = 0.1`, over 30 rounds:

| | best f1 | train loss, round 3 → 29 |
|---|---|---|
| FedAvg | **0.5936** | 0.614 → 0.353, monotone |
| FIPA | 0.4743 | 0.622 → 0.616, **flat for 27 rounds** |

The model stops moving exactly at the warmup boundary, where FIPA takes over. The
cause is geometric, not a bug: the update lives in the span of the kept curvature
directions, at most `M·r = 20` dimensions out of `p = 142 379`, and the measured
share of the local displacement that survives the projection is **2%**. Four runs
changing only the optimizer confirm the mechanism and refute the easy explanation —
SGD raises that share 3.7x and the plateau is still there, at a different height.
The real ceiling is `sqrt(r / local_steps)`, not `sqrt(r / p)`, and it holds for any
optimizer.

**`explained_variance_ratio` does not defend `fipa_rank`.** It measures how
concentrated the *gradient spectrum* is, and it is anti-correlated with quality —
the run with the highest value (0.848) is the one that diverges. The number that
answers the question is `fipa_delta_in_local`, the share of the displacement the
kept directions retain, logged per round alongside it.

### FedDisco — weights measured, quality not

No model has been trained with it yet. What *is* measured, on the real Dirichlet
partitions, is what the rule does to the aggregation weights:

| `dirichlet_alpha` | clients | `d_k` range | clients dropped | max departure from FedAvg |
|---|---|---|---|---|
| 1.0 | 4 / 8 | 0.157 – 0.227 | 0 / 0 | 0.006 / 0.017 |
| 0.5 | 4 / 8 | 0.135 – 0.328 | 0 / 0 | 0.005 / 0.059 |
| 0.1 | 4 / 8 | 0.238 – 0.459 | 0 / **2** | 0.034 / **0.177** |

Near-IID it reproduces FedAvg to within 1.7 points of weight; the discount bites
only where the skew is. At `alpha = 0.1` with 8 clients the two most skewed clients
are given weight zero and their local training is discarded — the ReLU in the rule
exists precisely to allow that.

### Reading any of these numbers

GTSRB is **30 consecutive frames per physical sign**. 12 630 test images are 421
signs; if the model is wrong on one frame it is wrong on all thirty. Every interval
above is computed on an effective sample size (`iot/metrics.py`), not on the raw
count. At 2 000 images per condition, `n_eff ≈ 419` and the 95% half-width at
accuracy 0.5 is **±4.8 points, not ±2.2**. A two-point difference between two
methods is not significant.

---

## Quickstart

Python **3.11** — `numpy==1.26.4`, `scikit-learn==1.5.0` and the pinned Flask stack
have no wheels for 3.13+.

```bash
# 1. environment (no torch: everything except training runs here)
uv venv --python 3.11 --seed .venv
.venv/bin/python -m pip install -r requirements_dev.txt

# 2. dataset — downloads the same archives torchvision uses, stdlib only
.venv/bin/python datasets_prep/prepare_gtsrb.py
#   -> dataset/gtsrb/train/00000..00042  (26 640 images, federated training)
#   -> dataset/gtsrb/test/00000..00042   (12 630 images, held out for Part B)

# 3. tests
.venv/bin/python -m pytest tests/ -q
```

On a machine with a GPU, add torch and run the grid:

```bash
pip install -r requirements_gpu.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python federated_grid_search.py grid_search_config.json
```

`./run_grid.sh [config.json]` does all of the above in one command (clone,
environment, dataset, grid) and is safe to re-run: the grid deduplicates against
the results CSV by a fingerprint of the configuration, so an interrupted session
resumes.

> **Do not install `requirements.txt` on Linux** — it is a Windows `pip freeze`
> from the received codebase and pulls `pywin32`/`pywinpty`. It is kept for
> reference only. `requirements_dev.txt` is the torch-free environment,
> `requirements_gpu.txt` the training one, `environment_gpu.yml` its conda twin.

---

## The three aggregation rules

All three are **server-side only** in this codebase, and the server never divides:
it emits a weighted sum and tells each client what to divide by. That split exists
so the encrypted path works — the server has no private key, and Paillier gives it
addition and multiplication by a plaintext scalar, nothing else.
`aggregation_policy.client_denominator` is the single source of truth for that
number.

### FedAvg / FedProx (received)

```
W = (1/N) * sum_k n_k * W_k        n_k = client k's sample count, N = their sum
```

Weight by data volume, identically for every parameter. Client divides by `N`.

### FedDisco

```
p_k = n_k / N                      what FedAvg would give client k
d_k = || D_k - u ||_2              distance of its label distribution from uniform
a_k = ReLU(p_k - a*d_k + b)        the score, floored at zero
w_k = a_k / sum_j a_j              normalised: the weights sum to 1
W   = sum_k w_k * W_k              already the average
```

`D_k` is the client's per-class counts as fractions; `u = (1/C, ..., 1/C)`. `d_k` is
0 for a balanced client and at most `sqrt(1 - 1/C) ≈ 0.988` for one holding a single
class. `a` (`feddisco_a`) sets how hard skew is punished, `b` (`feddisco_b`) is a
floor that keeps a moderately skewed client in the round.

Because the weights sum to 1 the server's output **is** the averaged model, so the
client divides by `1.0`. Note that `a = 0` alone is *not* FedAvg: the additive `b`
flattens the weights towards uniform before normalisation. FedAvg is recovered at
`a = b = 0`.

Implemented in `aggregation_policy.feddisco_weights` (the arithmetic, torch-free)
and `aggregator_ext.py` (both dispatch branches). It needs no client-side change at
all: a FedDisco round sends the same absolute weights a FedAvg round does.

Two diagnostics are written to the per-round CSV so a finished run can be checked:
`feddisco_weight_shift` = `max_k |w_k − n_k/N|` (0 means the round behaved exactly
like FedAvg) and `feddisco_clients_dropped`.

### FIPA

```
H     = sum_m a_m H_m              the round's consensus curvature
B_m   = a_m H^+ H_m                client m's preconditioner
theta = theta + sum_m B_m Delta_m  the update
```

`H_m` is client m's empirical Fisher information matrix `(1/n) G^T G`, with `G` its
stacked per-mini-batch gradients; `Delta_m = theta_m − theta_global`;
`a_m = N_m / N`; `H^+` is a pseudo-inverse, needed because `H` is singular by
construction. The weight becomes **per direction of parameter space** instead of one
number per client.

`H_m` written out is `p x p`, i.e. 81 GB per client per round, so **no function in
`fipa.py` ever materializes it**: everything stays in the low-rank form
`U_m diag(L_m) U_m^T` with `r = 5` directions from a randomized SVD, and
`H_m Delta_m` is computed right-to-left. `H^+` factors out of the sum, which is what
makes the whole round `O(p·r)`.

The first `fipa_warmup_rounds` rounds run as plain FedAvg — a low-rank curvature
estimate only says something useful near a minimum. Server and client resolve that
boundary through the *same* function (`aggregation_policy.effective_algorithm`);
computing it twice and disagreeing by one round would raise nothing.

**Under Paillier**, every appearance of `Delta_m` sits behind `U_m^T`, so the client
sends `z_m = U_m^T Delta_m` — **`r` ciphertexts instead of `p`** — and the encrypted
result equals the plaintext one exactly. Measured cost: ~1.3x encrypted FedAvg, with
the homomorphic work moving from the clients to the server. Two things make it
correct rather than merely linear:

- the server's three steps are **fused into one plaintext matrix** per client
  (`fipa.preconditioners`), so each ciphertext is multiplied exactly once — written
  naturally, the third product's integer crosses Paillier's ceiling;
- `fipa_encrypted.py` **pins the fixed-point exponent** on both sides of that
  multiplication. `phe` encodes a float as an integer times a power of 16, that
  integer must stay under `n/3`, and crossing the ceiling **does not raise** — it
  wraps and decrypts to a plausible wrong number.

`U_m` travels in the clear, because a QR decomposition of a ciphertext does not
exist. Encrypted FIPA therefore hides the parameters and the updates but **not the
curvature** — a weaker guarantee than encrypted FedAvg, and one worth stating rather
than claiming equivalence.

---

## Non-IID partitioning

`data_splitter_ext.PartitionedDatasetSplitter` subclasses the received splitter and
overrides only the choice of who gets what. The received `split_dataset` uses
`StratifiedKFold`, i.e. an IID split.

**The partition unit is the track, not the image.** GTSRB filenames are
`<track>_<frame>.png`: 30 consecutive frames of the same physical sign, so 26 640
training images are only **888 distinct signs**. A random per-image split leaks
near-duplicates into validation and, under Dirichlet, hands the same sign to several
clients. Cost to state: granularity drops to 888 units, and the rarest class has
only 5 tracks, so with 10 clients at most 5 can hold it.

**Class numbering is canonical, derived once from the source root**
(`class_mapping.py`). `ImageFolder` numbers the class subdirectories *it finds*,
alphabetically, from 0 — and a Dirichlet client does not hold all 43 classes in both
of its shares, so its labels shift from the first gap on, differently for each
client. Under the received IID split every numbering coincides and the problem is
invisible; it appears exactly when the partition becomes non-IID.

---

## Test-time adaptation (`iot/`)

Runs on one machine, after training. No server, no clients, no rounds: by the time
the bank of BatchNorm states exists there are no clients left to federate it.

```
image -> conv1 -> [bn1] -> relu -> maxpool -> ... -> layer4 -> head
                    |                                           |
        descriptor read HERE                        never touched by Part B
        (input of bn1: downstream of a frozen
         conv, so independent of which state
         the router loads)
```

| module | torch? | what |
|---|---|---|
| `corruption_shim.py` | no | patches `imagecorruptions` for numpy 2 / scikit-image 0.26 — import `corrupt` through it, never from the package |
| `gtsrb_c.py` | no | builds the corrupted test set under `dataset/gtsrb_c/` |
| `routing.py` | no | descriptors, symmetric-KL and L2 distances, the choice and its refusal threshold |
| `metrics.py` | no | effective sample size and its intervals, footprint, latency |
| `bn_bank.py` | yes | extracts states and descriptors; enforces descriptor independence |
| `bn_adapt.py` | yes | blind adaptation, and loading a chosen state |
| `source_model.py` | yes | rebuilds the checkpoint through `ModelManager`, plus offline recalibration |
| `stream_eval.py` | yes | the four arms over the stream, and the CLI |

```bash
# build the corrupted test set: 49 conditions x 2000 images, ~2 min on 16 cores
python -m iot.gtsrb_c

# recalibrate the checkpoint's BatchNorm statistics on clean training data
python -m iot.source_model --checkpoint checkpoints/<run>.pkl

# the experiment
python -m iot.stream_eval --checkpoint checkpoints/<run>_bn.pkl --out results/tta
```

Three design points that are easy to get wrong and raise nothing when you do:

- **The descriptor must be upstream of everything the router swaps.** Reading a
  BatchNorm layer's *output* is circular; so is reading the *input* of any layer
  downstream of one the bank overwrites. `bn_bank.check_descriptor_independence`
  turns the wrong combination into an error and is called before every stream.
- **Statistics are accumulated as raw moments** (count, sum, sum of squares, in
  float64), never by averaging per-batch variances — which underestimates. PyTorch's
  own accumulation is not a substitute: it is an exponential moving average that
  depends on batch order.
- **`clean` is a bank entry.** A bank of corruptions only would send a device
  looking at clean images to the nearest corruption.

---

## Repository layout

```
Received from the lab (extended, not rewritten)
  federated_server.py      Flask+SocketIO server, one per grid run
  federated_client.py      one thread per simulated client
  aggregator.py            plaintext and encrypted aggregation, metrics, early stop
  model_manager.py         model construction, local training, validation
  data_splitter.py         dataset -> per-client train/valid trees (IID)
  trusted_authority.py     Paillier keypair service
  utils.py                 serialisation and homomorphic helpers
  federated_grid_search.py the orchestrator; entry point for every run
  run_multiple_clients.py  splits the data, starts the client threads

Ours — torch-free, testable anywhere
  aggregation_policy.py    algorithm families, denominators, FedDisco weights
  fipa.py                  all of FIPA's linear algebra
  fipa_encrypted.py        the fixed-point discipline for the encrypted path
  data_splitter_ext.py     Dirichlet partitioning, selectable unit
  class_mapping.py         one canonical class numbering for every loader
  config_fingerprint.py    grid deduplication
  augmentation.py          rigid geometry only; refuses anything photometric
  seeding.py               global seeding

Ours — needs torch
  aggregator_ext.py        FedDisco and FIPA server side, checkpointing
  model_manager_ext.py     gradient collection, seeded shuffle, class remapping
  iot/                     test-time adaptation (above)

Configs, notebooks, tests
  grid_search_config.json              the 60-run grid
  grid_search_config_checkpoints.json  4 runs producing the adaptation checkpoint
  smoke_config*.json, test_config.json sanity runs
  diagnostics/                         narrow configs answering one question each
  notebooks/                           Colab notebooks and their run instructions
  tests/                               249 tests, no GPU needed
```

`nets_benchmark.py`, `conv_models/resnet.py`, `readme.txt` and `start.txt` come from
the received codebase and are unused — the first imports a module that does not
exist. They are left in place so the received tree stays recognisable.

---

## Configuration

Keys go in `common_search_space` **as lists**, even when single-valued: they are
search axes, and the grid deduplicates by a fingerprint built from them. Declared as
fixed parameters instead, two runs differing only in that key would fingerprint
identically and the second would be silently skipped.

| key | default | what it does |
|---|---|---|
| `aggregation_algorithm` | `FedAvg` | `FedAvg`, `FedProx`, `FedLC`, `FedDisco`, `FIPA` |
| `partition_strategy` | `stratified` | `dirichlet` for the non-IID split |
| `dirichlet_alpha` | 0.5 | smaller = more skew. Ignored unless the strategy is `dirichlet` |
| `partition_unit` | `track` | `image` reproduces the received per-image behaviour |
| `min_samples_per_client` | 10 | resample a partition that starves a client |
| `feddisco_a` | 0.5 | how hard label skew is discounted |
| `feddisco_b` | 0.1 | floor added to every score before the ReLU |
| `fipa_warmup_rounds` | 0 | rounds of plain FedAvg before FIPA takes over |
| `fipa_rank` | 5 | `r`, curvature directions kept per client |
| `fipa_grad_batches` | all | mini-batches used to estimate the curvature |
| `fipa_pinv_rtol` | 1e-8 | relative cut below which a curvature eigenvalue is dropped |
| `train_augmentation` | false | rigid geometry only; see below |
| `seed` | 42 | not fingerprinted, on purpose |
| `num_parallel_executions` | 12 | concurrent runs; lower it off the lab machine |

Three constraints the code will not catch for you:

- **`fipa_warmup_rounds` must be smaller than `global_epoch`**, or every round is
  FedAvg while the results row still says FIPA.
- **`fipa_grad_batches` must be at least `fipa_rank`.** Asking for more directions
  than there are gradients is not a crash — the rank is clamped — but it is a
  meaningless estimate.
- **Widening `dirichlet_alpha` does nothing while `partition_strategy` is
  `stratified`**, because alpha is pinned to a sentinel under an IID split.

Note that `global_epoch` is the number of **rounds**, not epochs — an inherited
misnomer. `local_epoch` is the real epoch count.

### The grids

| file | what it answers | runs |
|---|---|---|
| `grid_search_config.json` | do the aggregation rules differ under label skew? 4 algorithms x 3 alphas x 2 client counts x 2 local epochs, plaintext | 60 |
| `grid_search_config_checkpoints.json` | produces the checkpoint the adaptation half consumes. A strict subset of the above, so the grid skips these | 4 |
| `diagnostics/grid_search_config_encryption_cost.json` | what does Paillier cost? Both modes, 5 rounds | 8 |

Accuracy and encryption cost are separate grids on purpose: the encrypted path is
designed to reproduce the plaintext one exactly, so running the whole comparison
twice would measure only the clock. That was verified — the two modes differ by no
more than two runs of the *same* mode do.

A round costs ~85 s at `local_epoch: 1`, 128 px, batch 16 on a Colab GPU. The full
60-run grid is therefore ~111 h. A round is about **two** passes, not one: training
forward and backward, then validation forward-only over the same images.

### Augmentation is off, and that is a result

`augmentation.py` allows rigid geometry (rotation ±10°, translation 0.1, scale
0.9–1.1) and **refuses** horizontal and vertical flips — GTSRB has mirror-image
class pairs, so a flip relabels one class as another — along with everything
photometric or degrading. Brightness, contrast, blur and noise are four of the
conditions the adaptation half is evaluated on: augmenting with them would be
training on the test distribution.

The one legitimate lever was measured in a paired comparison and buys nothing:
f1 0.617 (off) against 0.557 (on), with the augmented arm below at 15 rounds out of
15, and `train_loss` 1.21 against 0.688 — it cannot fit its own training data. With
a frozen backbone the only trainable thing is a head on top of features it cannot
reshape, and augmentation buys invariance *by* reshaping features. It traded
overfitting for underfitting.

---

## Fixes to the received code

Found by reading the received code, not assumed. Each is a branch or an additive
config key, not a rewrite.

| what was wrong | consequence | where |
|---|---|---|
| The encrypted path ignored `aggregation_algorithm` and always weighted by sample count | Latent. Correct for FedAvg/FedProx/FedLC, wrong the moment an algorithm weights differently | `aggregator_ext.py` dispatches and raises on unknown rules |
| `image_size` was assigned, not defaulted, so the config axis collapsed to 224 | Duplicates silently dropped by the fingerprint | `federated_grid_search.py` |
| `aggregate_train_loss` was called but never defined | Latent crash | defined in `aggregator_ext.py` |
| Client statistics were appended to a flat list, discarding the sender | No way to pair a client's update with its own label distribution — and a reconnect double-counted | keyed by a stable `client_id`, sent in both directions |
| The "already started" guard only ever fired for FedLC | A repeated `client_ready` restarted round 0 under FedAvg | explicit `training_started` flag |
| No global seeding anywhere | — | `seeding.py`, called once per configuration |
| `NUM_PARALLEL_EXECUTIONS = 12` hardcoded | Out of memory off the lab machine | config key |
| A real `time.sleep` on the eventlet hub before shutdown | The delay that existed to let the shutdown reach the clients was what stopped it being sent: runs finished and never exited | cooperative sleep, outside the lock |
| Clients reconnected forever with no attempt limit | A finished run never returned, so its results row was never written | `reconnection_attempts` |
| The stratified train/valid split checked one of its two requirements | The client with the *best* class coverage fell back to validating on a single track of a single class | unstratified retry instead of a degenerate one |

**BatchNorm statistics are never federated, and this is left as it is.**
`get_weights` returns trainable *parameters*; `running_mean` and `running_var` are
buffers, so they are excluded regardless of `requires_grad`. Each client's
normalisation statistics therefore drift locally and are never aggregated, while the
model rebuilt from the server's weights carries ImageNet's. With a frozen backbone,
exchanging only the head is the intended scheme — changing it would change payload
size, encryption cost and comparability. The statistics are captured in the
checkpoint instead, and recalibrated once before deployment, which is where the
+25.8 points above come from.

---

## What is not claimed

- **A run is reproducible only up to the head initialisation.** The head's weights
  are drawn from torch's *global* RNG while clients build their models in concurrent
  threads; a seed fixes the sequence, not how threads divide it. The training
  shuffle is seeded per client; the head is not. A single run per grid cell carries
  more noise than the grid was designed assuming.
- **Every absolute accuracy is bounded by a linear probe on a frozen ImageNet
  backbone**, which centralised training puts at ~0.61 on GTSRB. What the
  adaptation experiment measures are the *differences between arms*.
- **The adaptation numbers rest on one checkpoint** (FedAvg, `dirichlet_alpha` 0.5,
  4 clients, validation accuracy 0.5844).
- **The effective sample size assumes maximal within-track correlation.** GTSRB
  shuffled the test filenames, so track identity is not recoverable and the true
  value cannot be measured — this is the conservative end.
- **Batch 16 is one extra point, not a curve.** Batches of 8 or 4 were not run.
- **The augmentation comparison used nearest-neighbour resampling**, itself a
  distribution shift on a frozen backbone. Bilinear was set afterwards and the
  comparison was not repeated.
- **FIPA's rank was not swept.** The ceiling argument is measured; whether a much
  larger `r` with proportionally more gradient batches closes the gap is not.

---

## Notebooks

`notebooks/` holds the Colab notebooks and, beside each, a short markdown file
saying what it runs and what to check:

| notebook | what it does |
|---|---|
| `smoke_tests_fl.ipynb` | end-to-end sanity runs — `SMOKE_TEST_1.md` (FedAvg, plaintext and encrypted), `SMOKE_TEST_FIPA.md` (both FIPA paths) |
| `Diagnostics_and_TTA_Checkpoint_Training.ipynb` | the checkpoint runs (`CHECKPOINT_RUNS.md`) and the narrow diagnostic ladder (`DIAGNOSTIC_RUNS.md`) |
| `TTA_RUN.ipynb` | builds the corrupted set and runs the four arms (`TTA_RUN.md`) |

On Colab, install only what is missing (`phe`, `flask-socketio`, `python-socketio`,
`eventlet`, `imagecorruptions`, `thop`, `gmpy2`) and leave the preinstalled
numpy/torch/scikit-learn alone — re-pinning `numpy==1.26.4` breaks the CUDA torch
that is already there. The header of `requirements_gpu.txt` has the exact command.

`gmpy2` is worth installing and easy to miss: `phe` uses it for modular
exponentiation if present and **falls back to pure Python in silence** if not. One
encrypted aggregation round measured 135 s without it and 16 s with. That matters
beyond the clock, because aggregation runs inside the socket handler — for its whole
duration the single-threaded server answers nothing.
