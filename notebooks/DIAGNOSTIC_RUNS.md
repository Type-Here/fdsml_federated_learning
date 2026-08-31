# Diagnostic runs - why the four checkpoint models learned nothing

**Answered, 2026-08-29. The cause was a label bug, not the aggregation.**
`ImageFolder` numbers the class subdirectories *it finds*, from 0, and under a
Dirichlet partition a client does not hold all 43 classes in both of its shares -
so labels were shifted, differently for every client. The fix is
`class_mapping.py`. This document is kept because the *method* is reusable and
because two of its five configs have never been launched; the sections below say
plainly which parts are answers and which are still open questions.

Written before the answer was known, corrected after. Where the first version
claimed something that later turned out to be too strong, the retraction is left
visible rather than edited away - see "A claim made and withdrawn".

---

## The answer, first

| | |
|---|---|
| **question** | four checkpoint runs (ResNet18 x {FedAvg, FIPA} x {alpha 0.1, 0.5}) produced models at or below trivial baselines. Why? |
| **answer** | the gate run, `num_clients: 1`, reproduced the failure **with no aggregation in the picture at all**. That eliminated both hypotheses below in one result and moved the search to `ModelManager`, between the training path and the validation path |
| **mechanism** | `_copy_images` (`data_splitter.py:111`) creates a class directory only where that class has images; `_get_dataloader` (`model_manager.py:239`) builds a *separate* `ImageFolder` per split. A client's `train/` and `valid/` therefore hold different class sets and number them differently, and two clients disagree with each other |
| **fix** | `class_mapping.py` - one numbering derived from the source root, applied to every loader. See `CLAUDE.md` **7T** |
| **cost of the bug** | the four checkpoint runs are void and must be re-run |

**The cleanest single piece of evidence.** After the fix, the gate's `train_loss`
is **identical digit for digit** to the pre-fix run, in both head sizes:

| round | `train_loss` before / after | `test_loss` before / after | `test_acc` before / after |
|---|---|---|---|
| 0 | 1.2582 / **1.2582** | 6.724 / **1.465** | 0.040 / **0.588** |
| 1 | 0.7290 / **0.7290** | 8.341 / **1.473** | 0.030 / **0.624** |
| 2 | 0.6177 / **0.6177** | 8.876 / **1.554** | 0.043 / **0.607** |
| 3 | 0.5611 / **0.5611** | 9.770 / **1.605** | 0.031 / **0.622** |
| 4 | 0.5343 / **0.5343** | 10.686 / **1.704** | 0.041 / **0.617** |

Exactly what the diagnosis predicts: with one client the `train/` tree holds all
43 class directories, so its numbering was already canonical and training did not
change by a bit - it was the 40-directory `valid/` that was shifted. **It is not
a better model. It is the same run, read with the right label.** (6.724 is 79%
*above* `ln(43) = 3.761`, the loss of a model answering uniformly: not ignorance,
a model being marked wrong on purpose.)

---

## A claim made and withdrawn

The first version of this document said *"none of the four models learnt anything
usable"*, resting on the strongest of three possible baselines. **That was too
wide**, and the table it rested on is replaced here.

Recomputed on the real partitions, size-weighted the way `aggregator.py:99`
aggregates:

| baseline | alpha=0.1 | alpha=0.5 |
|---|---|---|
| uniform over 43 classes | 0.0233 | 0.0233 |
| always answer that client's most frequent class | 0.0659 | 0.0889 |
| draw from the global class prior | 0.0409 | 0.0454 |

| run | accuracy | 95% CI (`n_eff = 90`) | reading |
|---|---|---|---|
| FedAvg alpha=0.1 | 0.164 | [0.088, 0.240] | **above every baseline** |
| FIPA alpha=0.1 | 0.237 | [0.149, 0.324] | **above every baseline** |
| FedAvg alpha=0.5 | 0.070 | [0.018, 0.123] | indistinguishable from ignorance |
| FIPA alpha=0.5 | 0.065 | [0.014, 0.116] | indistinguishable from ignorance |

So the two `alpha=0.1` runs *were* above chance; only the two `alpha=0.5` runs
were not. The blanket statement was wrong and is withdrawn.

**Two arguments survive any choice of baseline**, and they are the ones to keep:

- **the loss.** The two `alpha=0.5` runs sit at **5.92** and **5.59** against
  `ln(43) = 3.761`. They give the true class about `e^-5.9 = 0.27%` against the
  2.3% of a uniform guess. Not ignorant - *systematically wrong*;
- **the ceiling.** A pretrained ResNet18 with a GTSRB head is an easy problem.
  Whatever the baseline, 7% is not it.

**And no difference between the four is significant.** The validation sets are
whole tracks - 2730 and 2700 images are **91** and **90** distinct physical
signs, 30 frames each - so `n_eff` is ~90, not ~2700 (`iot/metrics.py:119`):

```
alpha=0.1:  FIPA - FedAvg = +7.3 points   threshold +-11.6   -> NOT significant
alpha=0.5:  FIPA - FedAvg = -0.6 points   threshold  +-7.3   -> NOT significant
```

"FIPA beats FedAvg at alpha=0.1" is not a claim these runs support - and now that
they are void, not a claim they could support anyway.

---

## What was ruled out first

Checked before hypothesising, so the search space stayed small. All still true:

- **the backbone really is pretrained** - `model_manager.py:162`,
  `ResNet18_Weights.IMAGENET1K_V1`;
- **the head has the right shape** - `Linear(512->256) -> ReLU -> Dropout(0.4)
  -> Linear(256->43)`, 142379 parameters, matching the shapes in the `.pkl`;
- **the checkpoints are sound** - no NaN, weight standard deviation 0.033
  against the 0.044 of a fresh `Linear(512,256)`, so the descaling was applied
  correctly;
- **all four clients train every round** (`MIN_NUM_WORKERS = 4`) and start from
  the same initial weights (`aggregator.py:18`);
- **the partition is reproducible** - the training totals come out at **23910**
  and **23940**, identical to the `aggregation_denominator` recorded in the two
  FedAvg checkpoints, so every later local measurement is on the same partition
  the runs saw.

The failure was not in the infrastructure. It was in the labels - which is a
third category the first version of this list did not have.

---

## The two hypotheses, and what became of them

Both were formulated before the gate ran. **Neither was ever reached**, and that
is worth keeping straight: they were not tested and refuted, they were made
unnecessary.

**H1 - the two-layer head.** With `num_custom_layers: 1` the head is a single
`Linear(512->43)`. With 2 there is a hidden layer of 256 units. FedAvg averages
weights coordinate by coordinate, and a hidden layer has *permutation symmetry*:
client A's hidden unit 5 and client B's hidden unit 5 encode different features,
because nothing aligned them. Averaging them does not interpolate, it destroys
both. With a single linear layer the problem does not exist - averaging linear
classifiers is well defined.

> **Status: dead as an explanation, untested as an effect.** The post-fix gate
> gives f1 **0.6090** (one layer) against **0.5883** (two layers) - a gap of
> 0.021 against a +-0.144 threshold at `n_eff = 89`. So a two-layer head is
> perfectly trainable centrally, which removes it as the cause of a *total*
> failure. Whether averaging a hidden layer costs something **under skew** is
> still open, and the re-run checkpoint runs are the next data on it.

**H2 - BatchNorm statistics are never federated.** The backbone is frozen, but
`_run_training_epoch` (`model_manager.py:261`) calls `model.train()`, and in
train mode BatchNorm updates `running_mean` / `running_var` on every forward
pass - they are *buffers*, not parameters, so `requires_grad=False` does not
stop them.

```
round i:  client_0 normalises with ITS statistics (data skewed to its classes)
          client_1 normalises with ITS ...          never synchronised, never aggregated
             |
             +-> each trains the head on a different feature space
             +-> the server averages the heads   -> destructive interference
             +-> and each client VALIDATES with its own statistics
```

> **Status: untested, and untestable by the gate.** With one client there is
> nothing to drift *apart from*. H2 remains a real property of the inherited
> scheme (`CLAUDE.md` **A2**) and it is exactly what Part B exploits; it is just
> not what broke these runs.

The two make different predictions, which is what the ladder was built to
separate: **H1 does not care about the skew** - averaging misaligned hidden units
goes wrong on IID data too. **H2 lives on the skew** - clients whose data look
alike develop statistics that look alike, so it should nearly vanish at a large
alpha. That design is still correct and still the right experiment, if the
question ever returns.

---

## The five configs, and what happened to each

All five write to `*_diagnostic` paths - `base_csv_path`, `base_log_path`,
`base_plot_path`, `base_split_data_path`, `base_checkpoint_path` - so their rows
land in **their own** results CSV. Two consequences, both wanted: they never
pollute the shared results file the real grid deduplicates against, and they are
never mistaken for grid work already done. Verified: they expand to 2 / 4 / 2 /
3 / 1 configurations with that many **distinct** fingerprints - no silent
collapse, which is the failure mode the normalisations can produce.

| config | what it asks | runs | status |
|---|---|---|---|
| `..._central.json` | **the gate.** `num_clients: 1`, so the server sums one weighted update and the client divides by the same total - the identity, i.e. no aggregation at all. What remains under test is only what every run shares: backbone, transform, head, learning rate. `num_custom_layers` crossed 1 x 2 | 2 | **ran twice** - `models/diagnostics/central/` pre-fix, `central2/` post-fix. This is the one that found it |
| `..._federated.json` | **the ladder.** 4 clients, `num_custom_layers` x `dirichlet_alpha` (100 / 0.5) | 4 | **ran once, pre-fix - the results are void.** All four cells sit at f1 0.039-0.067, which is the label bug and not an answer to H1 or H2. Re-run it, do not read it |
| `..._augmentation.json` | does geometric augmentation buy anything? Paired, off against on | 2 | **ran, post-fix.** Answer: no. See `CLAUDE.md` **7V** |
| `..._backbone.json` | is the starting model worth changing? 128 vs 224 px, and ResNet34 | 3 | **never launched** |
| `..._unfrozen.json` | `num_custom_layers: 0`, all 11.2M parameters. **Reference only** - never a checkpoint candidate | 1 | **never launched** |

### The gate - `grid_search_config_diagnostic_central.json`

**2 runs, ~10 rounds, roughly 15 minutes.** `global_epoch: 5` with
`local_epoch: 1` is plain centralised training for 5 epochs.

| outcome | reading | what actually happened |
|---|---|---|
| both reach a high f1 | model, transform and learning rate are fine; the failure is in the federated part - go to the ladder | - |
| **neither learns** | the fault is upstream of federation entirely, and the four checkpoint runs say nothing about FedAvg or FIPA. Look at the transform, the learning rate, **the label mapping** | **this one.** f1 0.040 / 0.047, loss 8.876 / 8.917 |
| `num_custom_layers: 1` learns and `2` does not | the two-layer head is not trainable at all | - |

**A prediction in the original version was wrong, and the correction is useful.**
It said "expect > 0.9; ResNet18 pretrained on GTSRB is an easy problem". Post-fix
the gate reaches **0.61**, and that is not a residual defect: this is a **linear
probe on frozen ImageNet features**, and GTSRB is far from ImageNet - small,
low-resolution signs, 18% of them under 32 px. The trajectory agrees, `test_loss`
bottoming at round 0-1 and rising after while accuracy flattens. **0.6 is roughly
the ceiling of this architectural choice.**

Two things follow. `global_epoch: 30` in the checkpoint runs is **generous** -
expect early stopping around round 6-9, with `best_model_weights` keeping the
best anyway. And "the frozen backbone costs us N points" is a sentence worth
having in the write-up that **nobody has measured**: `..._unfrozen.json` is the
run that would measure N.

### The ladder - `grid_search_config_diagnostic_federated.json`

**4 runs, 40 rounds, roughly 50 minutes.** `num_clients: 4`, `global_epoch: 10`,
two axes crossed:

| | `num_custom_layers: 1` | `num_custom_layers: 2` |
|---|---|---|
| **`dirichlet_alpha: 100`** (near-IID) | control: averaging on easy data | **H1 alone** - no skew, so no BatchNorm drift to blame |
| **`dirichlet_alpha: 0.5`** (the failing cell) | **H2 alone** - no hidden layer to misalign | reproduces the failure |

Reading the four cells:

- only the bottom-right fails -> **both** effects are needed, they compound;
- the whole right column fails, both alphas -> **H1**, the hidden layer;
- the whole bottom row fails, both head sizes -> **H2**, the skew and therefore
  the BatchNorm statistics;
- nothing fails -> the failure needs more than 10 rounds to appear, which would
  itself be informative.

**Do not launch this next.** The reason it existed was to separate H1 from H2,
and the gate made that unnecessary; the checkpoint runs are federated, so they
are themselves the test of the path that is still unproven - and they are also
the deliverable. Come back to the ladder only if the checkpoint runs are
disappointing in a way that needs explaining.

---

## What to read, in any of them

`early_stop_patience` is set to **999**, i.e. disabled on purpose, so the whole
trajectory is visible rather than being cut where the loss first rises five times
in a row. The per-round CSV in `csv_diagnostic_<hostname>/runs/` carries
`train_loss` and `test_loss` side by side, and the pair is the diagnosis:

| `train_loss` | `test_loss` | reading |
|---|---|---|
| falls | rises | the clients are learning and the aggregate is losing it - drift, so H1 or H2 |
| falls | **rises monotonically, well above `ln(43)`** | **the labels disagree between the two loaders.** This is the signature that was missed the first time |
| flat | flat | nothing is learning locally either: the head, the learning rate or the transform |
| falls | falls | that cell works, and the comparison with the cell that does not is the answer |

**Keep the per-round CSVs and the logs.** They are small, they are the evidence,
and one session expired before they were downloaded - which is why the first
reading had to be reconstructed from four checkpoint files. Everything kept so
far is under `models/diagnostics/`.

---

## A bug found while preparing this, and fixed

Reproducing the `alpha=100, num_clients=4` cell locally showed `client_3`
receiving a validation set of **one track - 30 frames of one sign, one class**,
while its training set held 7440 images.

`_split_units_train_valid` (`data_splitter_ext.py:400`) checked one of the two
requirements for a stratified split - at least 2 units per class - and not the
other: a stratified validation share must have at least as many slots as there
are classes. A client holding all 43 classes over 249 tracks asks for
`0.1 * 249 = 25` validation tracks against 43 classes, `train_test_split`
raises, and the sole fallback was `units[:-1], units[-1:]`.

The perverse part is who it hits. A *skewed* client holding 14 classes never
attempts to stratify and is fine; the client with the **best** class coverage is
the one that ends up with a one-class validation set. It then reports an accuracy
on a one-class problem, the server folds it into the round's weighted mean
(`aggregator.py:99`), and nothing raises.

The retry is now unstratified rather than degenerate, and the one-unit split
survives only as the last resort it was meant to be. Behaviour changes **only
where the old code raised**, so the four checkpoint runs' partitions are
untouched - verified: at `alpha` 0.1 and 0.5 with 4 clients the fallback never
fired. What it does change:

```
alpha=100, num_clients=4, client_3    before:  30 valid images,  1 class
                                       after: 630 valid images, 14 classes
        total validation set          before: 1980 images   after: 2700
```

Three tests added, `tests/test_dirichlet_partition.py`.

One cell of the full 45-run grid still has a small worst client -
`num_clients: 8, alpha: 0.1` leaves someone with 6 tracks over 4 classes - but
that is the partition doing what it was asked, not a fallback, and its weight in
the aggregate is proportionally small.

---

## What to run now

**Not a diagnostic. The checkpoint runs.** The gate has answered the question
this document was opened to ask, and the remaining unproven path - 4 clients
under Dirichlet, where each client numbers its own subset of classes in `train/`
too - is exercised by the checkpoint runs themselves.

```python
# 1. Clone the working branch EXPLICITLY. On main, main() takes no argument and
#    the config on the command line is ignored - the inherited grid runs instead.
!git clone -b features/tta <repo-url> fdsml
%cd fdsml

# 2. Install only what Colab is missing. Do NOT install requirements_gpu.txt
#    as-is: re-pinning numpy==1.26.4 breaks the preinstalled CUDA torch.
!pip install -q phe gmpy2 flask-socketio python-socketio eventlet

# 3. Build the dataset (downloads ~200 MB, writes dataset/gtsrb/train/00000..00042)
!python datasets_prep/prepare_gtsrb.py --splits train

# 4. The four checkpoint runs. ~2.8 h at the measured 85 s/round.
!python federated_grid_search.py grid_search_config_checkpoints.json
```

Use `%cd`, not `!cd`: `!cd` opens a subshell that dies with the cell.

**Watch round 2 of the first run, and be ready to stop it.** If f1 is in the
0.3-0.5 range it is working. **If it is around 0.03, kill it** rather than paying
2.8 hours to confirm the fix did not take on the 4-client path.

**If you re-launch on the same machine, clear the four void rows from
`csv_<hostname>/` first.** The label fix changes no configuration key, so those
rows *should* fingerprint identically to the new runs and suppress them silently.
Measured, they do not - the CSV round trip writes `fedprox_mu` as `0.0` where the
generated config has `0`, and the fingerprint compares with `str()` - but that is
a formatting accident, not a design, so do not rely on it. On a fresh Colab clone
the question does not arise.

**Download the evidence before the session ends** - it is small, and it has been
lost once:

```python
from google.colab import files
!zip -r results.zip csv_* logs_* checkpoints_*
files.download('results.zip')
```

## The timing constant, no longer missing

It used to say here that nobody had measured how long one pass over the training
set takes on the target GPU. **Measured, 2026-08-29, from these very runs**,
which is what `early_stop_patience: 999` made possible - they all ran their full
`global_epoch`:

| config | clients | rounds | duration | **s / round** |
|---|---|---|---|---|
| gate, `num_custom_layers` 1 / 2 | 1 | 5 | 428 / 437 s | 85.7 / 87.5 |
| ladder, alpha 100 | 4 | 10 | 752 / 774 s | 75.2 / 77.4 |
| ladder, alpha 0.5 | 4 | 10 | 719 / 747 s | 71.9 / 74.7 |
| augmentation, off / on | 1 | 15 | 1188 / 1334 s | 79.2 / 88.9 |

**~85 s per round** at `local_epoch: 1`, and it barely moves with client count or
head size. Three cautions before multiplying it by anything: a round is about
**two** passes, not one (training forward+backward, then `validate('train')`
forward-only over the same images); a `local_epoch: 5` cell is therefore roughly
three times this, not five; and it is a measurement of one Colab GPU, of which
there have been three different ones so far. Details in `CLAUDE.md` **7W**.
