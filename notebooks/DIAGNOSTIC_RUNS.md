# Diagnostic runs - why did the four checkpoint models learn nothing?

The four checkpoint runs finished and produced four usable files. The models
inside them are at or below trivial baselines, which is a different problem from
"the results are low" and has to be settled before anything downstream is
measured.

## What was measured, and why it is not a matter of opinion

The partition of each of the four runs was reproduced locally, bit for bit -
same seed, same code, no GPU needed. The training totals come out at **23910**
and **23940** images, identical to the `aggregation_denominator` recorded in the
two FedAvg checkpoints, so the reproduction is the same partition the runs saw.
On that partition one can compute what a model that has learnt *nothing* would
score, and compare:

| run | accuracy | "always answer that client's most frequent class" | |
|---|---|---|---|
| FedAvg alpha=0.1 | 0.164 | 0.220 | **below** |
| FIPA alpha=0.1 | 0.237 | 0.220 | +1.7 points |
| FedAvg alpha=0.5 | 0.070 | 0.200 | **far below** |
| FIPA alpha=0.5 | 0.065 | 0.200 | **far below** |

The two `alpha=0.5` runs are below an even weaker baseline, *guess at random
among only the classes that client holds* (0.0775).

**The loss says the same thing by an independent route.** `ln(43) = 3.761` is
the cross-entropy of a model answering uniformly over 43 classes. The two
`alpha=0.5` runs sit at **5.92** and **5.59**, i.e. they give the true sign about
`e^-5.9 = 0.27%` against the 2.3% of a coin toss. Not ignorant - *systematically
wrong*, pushing the correct class down.

**And no difference between the four is significant.** The validation sets are
made of whole tracks - 2730 and 2700 images are **91** and **90** distinct
physical signs, 30 frames each - so the effective sample size is ~90, not ~2700
(`iot/metrics.py:119`). The 95% half-widths are 5 to 9 points:

```
alpha=0.1:  FIPA - FedAvg = +7.3 points   threshold +-11.6   -> NOT significant
alpha=0.5:  FIPA - FedAvg = -0.6 points   threshold  +-7.3   -> NOT significant
```

So "FIPA beats FedAvg at alpha=0.1" is not a claim these runs support.

## What was ruled out first

Checked before hypothesising, so the search space is smaller:

- **the backbone really is pretrained** - `model_manager.py:162`,
  `ResNet18_Weights.IMAGENET1K_V1`;
- **the head has the right shape** - `Linear(512->256) -> ReLU -> Dropout(0.4)
  -> Linear(256->43)`, 142379 parameters, matching the shapes in the `.pkl`;
- **the checkpoints are sound** - no NaN, weight standard deviation 0.033
  against the 0.044 of a fresh `Linear(512,256)`, so the descaling was applied
  correctly;
- **all four clients train every round** (`MIN_NUM_WORKERS = 4`) and start from
  the same initial weights (`aggregator.py:18`);
- **the partition is reproducible.**

The failure is not in the infrastructure. It is in the learning.

## The two anomalies that point somewhere

**1. `alpha=0.5` is 2.3x worse than `alpha=0.1`, and it should be the other way
round.** `alpha` is the Dirichlet concentration: larger means labels are spread
more evenly across clients, so the problem is *easier*. At 0.5 the clients see
27/31/28/42 classes; at 0.1 they see 15/18/22/40. The easier case does worse.

**2. The smoke test of 2026-08-26 did better with a tenth of the data and two
rounds** - accuracy 0.39, loss 2.24, below `ln(43)`, so it was learning. The
configuration difference is `num_custom_layers`: **1** there, **2** here.

## The two hypotheses these runs separate

**H1 - the two-layer head.** With `num_custom_layers: 1` the head is a single
`Linear(512->43)`. With 2 there is a hidden layer of 256 units. FedAvg averages
weights coordinate by coordinate, and a hidden layer has *permutation symmetry*:
client A's hidden unit 5 and client B's hidden unit 5 encode different features,
because nothing aligned them. Averaging them does not interpolate, it destroys
both. With a single linear layer the problem does not exist - averaging linear
classifiers is well defined. This is the hypothesis that also explains anomaly 2.

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

The two hypotheses make different predictions, and that is what the ladder below
tests: **H1 does not care about the skew** - averaging misaligned hidden units
goes wrong on IID data too. **H2 lives on the skew** - clients whose data look
alike develop statistics that look alike, so it should nearly vanish at a large
alpha.

## The ladder

Two config files, in sequence. The second is only worth launching if the first
passes.

### 1. The gate - `grid_search_config_diagnostic_central.json`

**2 runs, ~10 passes over the training set, roughly 10 minutes.**

`num_clients: 1`, so there is no aggregation at all: the server sums one
update weighted by that client's size and the client divides by the same total,
which is the identity. `global_epoch: 5` with `local_epoch: 1` is therefore
plain centralised training for 5 epochs. What remains under test is exactly
what every run shares - backbone, transform, head, learning rate.
`num_custom_layers` is crossed 1 against 2 so the question "is the two-layer
head trainable at all" is answered separately from "does averaging break it".

| outcome | reading |
|---|---|
| **both reach a high f1** (expect > 0.9; ResNet18 pretrained on GTSRB is an easy problem) | model, transform and learning rate are fine. The failure is in the federated part - go to step 2 |
| **neither learns** | the fault is upstream of federation entirely, and the four checkpoint runs say nothing about FedAvg or FIPA. Look at the transform, the learning rate, the label mapping |
| **`num_custom_layers: 1` learns and `2` does not** | the two-layer head is not trainable in this setup at all, which settles it without step 2 |

### 2. The ladder - `grid_search_config_diagnostic_federated.json`

**4 runs, 40 passes, roughly 35 minutes.** `num_clients: 4`, `global_epoch: 10`,
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
  itself be informative: it would mean the model *degrades* rather than never
  learning, and the per-round CSV shows exactly when.

## What to read, in both steps

`early_stop_patience` is set to **999**, i.e. disabled on purpose, so the whole
trajectory is visible rather than being cut where the loss first rises five
times in a row. The per-round CSV in `csv_diagnostic_<hostname>/runs/` carries
`train_loss` and `test_loss` side by side, and the pair is the diagnosis:

| `train_loss` | `test_loss` | reading |
|---|---|---|
| falls | rises | the clients are learning and the aggregate is losing it - drift, so H1 or H2 |
| flat | flat | nothing is learning locally either: the head, the learning rate or the transform |
| falls | falls | that cell works, and the comparison with the cell that does not is the answer |

**Keep the per-round CSVs and the logs this time.** They are small, they are the
evidence, and the previous session expired before they were downloaded - which
is why the reading above had to be reconstructed from four checkpoint files.

## Everything is written to separate paths

`base_csv_path`, `base_log_path`, `base_plot_path`, `base_split_data_path` and
`base_checkpoint_path` all carry a `_diagnostic` suffix, so these six rows land
in **their own** results CSV. Two consequences, both wanted: they never pollute
the shared results file the real grid deduplicates against, and they are never
mistaken for grid work already done. Verified: the two files expand to exactly
**2** and **4** configurations, with 2 and 4 **distinct** fingerprints - no
silent collapse.

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

Three tests added, `tests/test_dirichlet_partition.py`; **210 pass locally**.

One cell of the full 45-run grid still has a small worst client -
`num_clients: 8, alpha: 0.1` leaves someone with 6 tracks over 4 classes - but
that is the partition doing what it was asked, not a fallback, and its weight in
the aggregate is proportionally small.

## Running it on Colab

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

# 4. THE GATE. Two runs, ~10 minutes. Read the result before going on.
!python federated_grid_search.py grid_search_config_diagnostic_central.json
!cat csv_diagnostic_*/runs/*.csv

# 5. Only if the gate passed. Four runs, ~35 minutes.
!python federated_grid_search.py grid_search_config_diagnostic_federated.json
```

Use `%cd`, not `!cd`: `!cd` opens a subshell that dies with the cell.

**Download the evidence before the session ends** - it is small, and last time it
was lost:

```python
from google.colab import files
!zip -r diagnostic.zip csv_diagnostic_* logs_diagnostic_*
files.download('diagnostic.zip')
```

## Time this, while it runs

The single constant behind every hour estimate for the full grid - how long one
pass over the training set takes on the target GPU - has still never been
measured. The gate run is 5 passes on the whole training set with no aggregation
overhead at all, which is the cleanest place there will ever be to measure it.
Divide the run's `total_duration` by 5.
