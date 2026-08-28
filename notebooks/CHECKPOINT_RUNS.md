# Checkpoint runs - four models for the inference-time work

Four federated runs on GTSRB whose product is not a number but a **file**: the
trained global model, which everything on the inference side starts from. Until
now a finished run left only its metrics; the model lived in memory and the
process exited.

Unlike the two smoke tests, **these are real runs** - full dataset, 30 rounds,
results that count. `grid_search_config_checkpoints.json` is a strict subset of
`grid_search_config.json`: same paths, same fixed parameters, every axis narrowed
to a value that file already contains. So the four rows land in the **same**
shared results CSV, and when the full grid is launched later it recognizes them
by fingerprint and runs **41 instead of 45**. Nothing here is repeated work.

| | |
|---|---|
| model | ResNet18, `num_custom_layers: 2` (frozen backbone, trainable head of 142 379 parameters) |
| algorithms | FedAvg, FIPA |
| `dirichlet_alpha` | 0.1, 0.5 |
| clients | 4, all sampled every round |
| rounds | `global_epoch: 30`, `local_epoch: 1` |
| encryption | `no_encryption` |
| **runs** | **2 x 2 = 4** |

## Why these four and not others

**ResNet18 only.** The adaptation stage adjusts BatchNorm, and `ConvNet`
(`model_manager.py:15`) is the HE-friendly network with `x*x` activations - it
has no BatchNorm layer at all, and neither does torchvision's AlexNet. A
checkpoint from those is not less useful, it is unusable. ConvNet keeps its place
in the full grid, where it is the only non-transfer-learning entry.

**No encryption axis.** Two independent reasons: in encrypted mode
`current_weights` holds Paillier ciphertext dictionaries and the server has no
private key - by design, the Trusted Authority hands keys to clients only - so a
plaintext checkpoint cannot come from such a run; and the two modes were measured
to differ by less than two runs of the *same* mode do.

**No FedProx.** Server-side it performs the same arithmetic as FedAvg - a
summation weighted by `train_size`, the shared branch at `aggregator.py:59`. It
differs in local training, so the model is not identical, but it does not ask a
new question of the adaptation stage. The same GPU hours buy more on the alpha
axis, which changes the starting model much more.

**`local_epoch: 1`.** Cost is `global_epoch x local_epoch` passes over the
training set, so 5 would be five times the wall clock. The client-drift axis
belongs to the full grid, where it is the knob FIPA claims to correct; here the
model is the deliverable, not the comparison.

## Cost

`30 x 1 = 30` passes per run, **120 passes** in total - 3.4% of the full grid's
3510. The per-pass constant on a Colab GPU has **never been measured**, and every
hour estimate for the full grid scales off it, so take that measurement here:
time the first round and multiply. The rough expectation is 20-40 s per pass,
i.e. 40-80 minutes for all four.

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

# 4. Run the four configurations, in sequence (num_parallel_executions: 1)
!python federated_grid_search.py grid_search_config_checkpoints.json

# 5. Look at what came out, without unpickling anything
!ls -lh checkpoints_*/
!cat checkpoints_*/*.json | head -60
```

Use `%cd`, not `!cd`: `!cd` opens a subshell that dies with the cell. Step 3
takes a few minutes and has to be redone if the runtime disconnects.

**Download the checkpoints before the session ends.** They are the only output
that cannot be recomputed cheaply:

```python
from google.colab import files
!zip -r checkpoints.zip checkpoints_* csv_*
files.download('checkpoints.zip')
```

## What "passed" looks like

**1. The process exits on its own**, printing
`=== All parallel executions have finished. ===`.

**2. Four checkpoints on disk**, in `checkpoints_<hostname>/`, each with a `.pkl`
and a `.json` twin. The filenames say what they are:

```
gtsrb_ResNet18_FedAvg_a0.1_c4_le1_seed42_<timestamp>.pkl
gtsrb_ResNet18_FedAvg_a0.5_c4_le1_seed42_<timestamp>.pkl
gtsrb_ResNet18_FIPA_a0.1_c4_le1_seed42_<timestamp>.pkl
gtsrb_ResNet18_FIPA_a0.5_c4_le1_seed42_<timestamp>.pkl
```

**3. The server log confirms the rescale**, one line per run:

```
Saved global model checkpoint to ... (round 24, f1 0.8xxx, 142379 parameters, descaled by 20xxx.xxxx)
```

Read `descaled by` carefully, because it is the one number that decides whether
the file is usable:

| algorithm | expected divisor | what it means |
|---|---|---|
| FedAvg | the round's total training size, **~20000** | the server aggregates by summation and the clients divide; the checkpoint has to do the same division |
| FIPA | **1.0** | FIPA does not produce an average - its result already is the model |

A FedAvg checkpoint reporting `descaled by 1.0` would be wrong by a factor of
twenty thousand, and would load without complaint and predict noise. FIPA's
`1.0` is correct and is not a missing divisor.

Note that FIPA's divisor is `1.0` only if its best round was a FIPA round. With
`fipa_warmup_rounds: 3`, if round 0, 1 or 2 turned out to be the best, the
divisor is `N` - the checkpoint follows the round the weights came from, not the
one the run ended on. Check `best_round` in the JSON before deciding anything is
wrong.

**4. `num_parameters: 142379`** in every JSON. That is
`512x256 + 256` for the first layer plus `256x43 + 43` for the second, i.e. the
custom head and nothing else - so the backbone, and with it every BatchNorm
layer, was frozen as intended. A different number means `num_custom_layers`
was not 2, and that model cannot serve as the source for the adaptation work.

**5. `"bn_stats_source": "imagenet"`** in every JSON. Not a defect: BatchNorm's
`running_mean` / `running_var` are buffers rather than parameters, so
`get_weights` never saw them and no round ever aggregated them. Whoever loads the
checkpoint gets a freshly built backbone's statistics, which are ImageNet's. The
field is there so that this is a known property rather than a discovery, and it
is what the recalibration pass exists to fix.

**6. The results CSV has a `checkpoint_path` column**, so each row can be traced
to the model it produced.

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

`set_weights` copies **positionally** into whatever
`_get_trainable_parameters()` returns, which is why the architecture travels in
the metadata: built with a different `num_custom_layers` or `num_classes`, the
same arrays either raise on a shape mismatch or, worse, fit and mean something
else.
