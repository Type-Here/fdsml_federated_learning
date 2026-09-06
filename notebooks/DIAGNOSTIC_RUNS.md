# Diagnostic runs — narrow configs that answer one question each

**Why they exist.** Five configurations, each stripping the pipeline down until only
one variable is left. They were built to explain why four checkpoint runs produced
models at or below trivial baselines. **The question is answered:** the cause was a
label bug, not the aggregation — `ImageFolder` numbers the class subdirectories *it
finds*, and under a Dirichlet partition a client does not hold all 43 classes in
both of its shares, so labels shifted differently for every client. The fix is
`class_mapping.py`.

The configs are kept because the method is reusable and two of them have never been
launched.

All five write to `*_diagnostic` paths (`base_csv_path`, `base_log_path`,
`base_plot_path`, `base_split_data_path`, `base_checkpoint_path`), so their rows land
in their own results CSV and never pollute the file the real grid deduplicates
against. `early_stop_patience: 999` — disabled on purpose, so the whole trajectory is
visible rather than being cut where the loss first rises.

| config | what it asks | runs | status |
|---|---|---|---|
| `diagnostics/..._central.json` | **the gate.** `num_clients: 1`, so there is no aggregation at all — what remains under test is only what every run shares: backbone, transform, head, learning rate. `num_custom_layers` crossed 1 x 2 | 2 | ran twice, pre- and post-fix. **This is the one that found the bug** |
| `diagnostics/..._federated.json` | **the ladder.** 4 clients, `num_custom_layers` x `dirichlet_alpha` (100 / 0.5) | 4 | ran once, **pre-fix — the results are void.** Re-run it, do not read it |
| `diagnostics/..._augmentation.json` | does geometric augmentation buy anything? Paired, off against on | 2 | ran, post-fix. Answer: **no** |
| `diagnostics/..._backbone.json` | is the starting model worth changing? 128 vs 224 px, and ResNet34 | 3 | never launched |
| `diagnostics/..._unfrozen.json` | `num_custom_layers: 0`, all 11.2 M parameters. Reference only, never a checkpoint candidate | 1 | never launched |
| `diagnostics/..._optimizer.json` | Adam vs SGD at two learning rates, to test whether the optimizer explains FIPA's plateau | 4 | ran. Answer: **it does not** |

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

# 4. Pick one. The gate is 2 runs / ~15 min; the ladder 4 runs / ~50 min.
!python federated_grid_search.py diagnostics/grid_search_config_diagnostic_central.json

# 5. Keep the evidence - it is small, and it has been lost once
from google.colab import files
!zip -r diagnostics.zip csv_*diagnostic* logs_*diagnostic* checkpoints_*diagnostic*
files.download('diagnostics.zip')
```

## What to read

The per-round CSV in `csv_diagnostic_<hostname>/runs/` carries `train_loss` and
`test_loss` side by side, and the pair **is** the diagnosis:

| `train_loss` | `test_loss` | reading |
|---|---|---|
| falls | rises | the clients are learning and the aggregate is losing it — drift |
| falls | **rises monotonically, well above `ln(43) = 3.761`** | **the labels disagree between the two loaders.** This is the signature that was missed the first time |
| flat | flat | nothing is learning locally either: the head, the learning rate or the transform |
| falls | falls | that cell works; the comparison with the cell that does not is the answer |

## What the gate established

| | `num_custom_layers: 1` | `num_custom_layers: 2` |
|---|---|---|
| before the label fix | f1 **0.040**, loss 8.88 | f1 **0.047**, loss 8.92 |
| after | f1 **0.609**, loss 1.61 | f1 **0.588**, loss 1.60 |

The cleanest single piece of evidence: after the fix the gate's `train_loss` is
**identical digit for digit** to the pre-fix run — 1.2582, 0.7290, 0.6177, 0.5611,
0.5343 — while `test_loss` moves from 6.724 to 1.465 and accuracy from 0.040 to
0.588. With one client the `train/` tree holds all 43 class directories, so training
did not change by a bit; it was the 40-directory `valid/` that was shifted. **It is
not a better model, it is the same run read with the right label.**

**0.6 is the ceiling, not a leftover defect.** This is a linear probe on frozen
ImageNet features and GTSRB is far from ImageNet — small, low-resolution signs, 18%
of them under 32 px. Two consequences: `global_epoch: 30` in the checkpoint runs is
generous (expect early stopping around round 6–9), and "the frozen backbone costs us
N points" is a sentence nobody has measured — `..._unfrozen.json` is the run that
would measure it.

## The timing constant

Measured from these runs, which is what `early_stop_patience: 999` made possible —
they all ran their full `global_epoch`:

| config | clients | rounds | duration | **s / round** |
|---|---|---|---|---|
| gate, `num_custom_layers` 1 / 2 | 1 | 5 | 428 / 437 s | 85.7 / 87.5 |
| ladder, alpha 100 | 4 | 10 | 752 / 774 s | 75.2 / 77.4 |
| ladder, alpha 0.5 | 4 | 10 | 719 / 747 s | 71.9 / 74.7 |
| augmentation, off / on | 1 | 15 | 1188 / 1334 s | 79.2 / 88.9 |

**~85 s per round** at `local_epoch: 1`, barely moving with client count or head
size. Three cautions before multiplying: a round is about **two** passes, not one
(training forward+backward, then `validate('train')` forward-only over the same
images); a `local_epoch: 5` cell is therefore roughly three times this, not five;
and it is one Colab GPU, of which there have been three different ones.
