# Checkpoint runs — the four models the inference-time work starts from

**Why it exists.** The product of these runs is not a number but a **file**: the
trained global model, which the test-time adaptation half consumes. Unlike the two
smoke tests these are **real runs** — full dataset, 30 rounds, results that count.

`grid_search_config_checkpoints.json` is a strict subset of
`grid_search_config.json` — same paths, same fixed parameters, every axis narrowed
to a value that file already contains — so the four rows land in the same shared
results CSV and the full grid later recognises them by fingerprint and skips them.
Nothing here is repeated work.

| | |
|---|---|
| model | ResNet18, `num_custom_layers: 2` (frozen backbone, head of 142 379 parameters) |
| algorithms | FedAvg, FIPA |
| `dirichlet_alpha` | 0.1, 0.5 |
| clients | 4, all sampled every round |
| rounds | `global_epoch: 30`, `local_epoch: 1` |
| encryption | `no_encryption` |
| **runs** | **2 x 2 = 4**, ~2.8 h at the measured ~85 s/round |

**Why these four.** ResNet18 only, because the adaptation stage adjusts BatchNorm
and `ConvNet` (HE-friendly, `x*x` activations) has none at all — a checkpoint from
it is not less useful, it is unusable. No encryption axis, because in encrypted mode
the server holds Paillier ciphertexts and has no private key, so a plaintext
checkpoint cannot come from such a run. No FedProx, because server-side it performs
the same arithmetic as FedAvg; the same GPU hours buy more on the alpha axis.

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

# 4. The four configurations, in sequence (num_parallel_executions: 1)
!python federated_grid_search.py grid_search_config_checkpoints.json

# 5. Inspect what came out, without unpickling anything
!ls -lh checkpoints_*/
!cat checkpoints_*/*.json | head -60
```

**Download the checkpoints before the session ends** — they are the only output that
cannot be recomputed cheaply:

```python
from google.colab import files
!zip -r checkpoints.zip checkpoints_* csv_*
files.download('checkpoints.zip')
```

**Watch round 2 of the first run.** If f1 is in the 0.3–0.5 range it is working; if
it is around 0.03, kill it rather than paying 2.8 hours.

## What "passed" looks like

1. **The process exits on its own**, printing
   `=== All parallel executions have finished. ===`.
2. **Four checkpoints in `checkpoints_<hostname>/`**, each a `.pkl` with a `.json`
   twin: `gtsrb_ResNet18_{FedAvg,FIPA}_a{0.1,0.5}_c4_le1_seed42_<timestamp>.pkl`.
3. **The server log confirms the rescale**, one line per run:
   `Saved global model checkpoint to ... (round 24, f1 0.8xxx, 142379 parameters, descaled by 20xxx.xxxx)`

   | algorithm | expected divisor | why |
   |---|---|---|
   | FedAvg | the round's total training size, **~20 000** | the server aggregates by summation and the clients divide; the checkpoint must do the same |
   | FIPA | **1.0** | FIPA does not produce an average — its result already is the model |

   A FedAvg checkpoint reporting `descaled by 1.0` is wrong by a factor of twenty
   thousand, loads without complaint and predicts noise. FIPA's `1.0` is correct.
   But FIPA's divisor is `1.0` only if its best round was a FIPA round — with
   `fipa_warmup_rounds: 3`, a best model from round 0–2 carries divisor `N`. Check
   `best_round` in the JSON before deciding anything is wrong.
4. **`num_parameters: 142379`** in every JSON — `512x256 + 256` plus `256x43 + 43`,
   i.e. the custom head and nothing else, so the backbone and every BatchNorm layer
   was frozen as intended.
5. **`"bn_stats_source": "imagenet"`** in every JSON. Not a defect: BatchNorm's
   `running_mean` / `running_var` are buffers, not parameters, so no round ever
   aggregated them and a loaded checkpoint gets a fresh backbone's statistics. The
   field makes that a known property; the recalibration pass is what fixes it.
6. **The results CSV has a `checkpoint_path` column**, so each row traces to its model.

## Loading one afterwards

```python
import pickle
from model_manager import ModelManager

with open('checkpoints_<hostname>/<name>.pkl', 'rb') as handle:
    checkpoint = pickle.load(handle)

meta = checkpoint['metadata']
config = {'model_name': meta['model_name'], 'num_classes': meta['num_classes'],
          'num_custom_layers': meta['num_custom_layers'],
          'image_size': meta['image_size'], 'device': 'cuda'}

manager = ModelManager(config=config, dataset_path='dataset/gtsrb_c')
manager.set_weights(checkpoint['weights'])

# Each corruption directory is an ImageFolder root, and _get_dataloader joins
# dataset_path with the split name - so the condition takes the split's place.
manager.validate(batch_size=64, split='fog_s3')
```

`set_weights` copies **positionally** into whatever `_get_trainable_parameters()`
returns, which is why the architecture travels in the metadata: built with a
different `num_custom_layers` or `num_classes`, the same arrays either raise on a
shape mismatch or, worse, fit and mean something else.
