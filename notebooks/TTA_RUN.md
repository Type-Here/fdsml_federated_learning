# TTA run - Part B on the FedAvg alpha=0.5 checkpoint

The inference half, executed. Everything in `iot/` is already written and the
torch-free half is covered by the local test suite; what has **never run** is
`stream_eval.py`, `source_model.py`, `bn_bank.py`, `bn_adapt.py`. Steps 5 and 6
exist for exactly that reason - do not skip them to save ten minutes.

The checkpoint:

    models/second_run_checkpoints/gtsrb_ResNet18_FedAvg_a0.5_c4_le1_seed42_20260829-185510.pkl

    ResNet18, num_custom_layers 2, 128 px, 43 classes, 142379 parameters
    best round 27:  f1 0.6446   acc 0.5844   loss 1.5841
    bn_stats: null   bn_stats_source: "imagenet"

f1 0.6446 is the ceiling a linear probe on a frozen ImageNet backbone reaches on
GTSRB, which the centralized diagnostic established independently. This is the
post-label-fix run, not one of the four void ones under
`models/first_run_checkpoint/`.

---

## The cells

```bash
# 1. Clone the working branch EXPLICITLY. On main, main() takes no argument and
#    the config on the command line is ignored. Use %cd, not !cd: !cd opens a
#    subshell that dies with the cell.
!git clone -b features/tta <repo-url>
%cd fdsml_federated_learning

# 2. Only what Colab is missing. Do NOT install requirements_gpu.txt as-is:
#    re-pinning numpy==1.26.4 breaks the preinstalled CUDA torch.
!pip install imagecorruptions thop "setuptools<81"

# 3. Clean GTSRB, BOTH splits - the two halves below need different ones:
#      dataset/gtsrb/test   -> what step 4 corrupts (26640 train / 12630 test,
#      dataset/gtsrb/train  -> what step 7 recalibrates on   disjoint archives)
!python datasets_prep/prepare_gtsrb.py
!ls dataset/gtsrb                    # must list both: train, test

# 4. GTSRB-C, regenerated rather than uploaded (2.1 GB; the build is ~2-5 min
#    and is seeded per (image, corruption, severity), so it is reproducible)
!python -m iot.gtsrb_c
!ls dataset/gtsrb_c | wc -l          # must be 50: 49 conditions + manifest.json
```

```python
# 5. self_check - the accumulation, against the same data in one batch.
#    No CLI exists for this one, so it is four lines here.
import itertools, torch
from iot.bn_bank import self_check
from iot.source_model import build_model, image_folder_loader, load_checkpoint

CKPT = ("models/second_run_checkpoints/"
        "gtsrb_ResNet18_FedAvg_a0.5_c4_le1_seed42_20260829-185510.pkl")

checkpoint = load_checkpoint(CKPT)
model, manager = build_model(checkpoint['metadata'], checkpoint['weights'])
loader = image_folder_loader('dataset/gtsrb/train', manager.transform_pipeline,
                             batch_size=128, num_workers=2)
# a few hundred images is enough, and self_check concatenates them all
print(self_check(model, list(itertools.islice(loader, 4)), manager.device))
```

```bash
# The path, once, so the cells below are copy-pasteable
%env CKPT=models/second_run_checkpoints/gtsrb_ResNet18_FedAvg_a0.5_c4_le1_seed42_20260829-185510.pkl

# 6. The branch walk. Same branches as the full run - fallback, bootstrap batch,
#    unseen corruptions, four arms - in minutes. Its numbers are meaningless.
!python -m iot.stream_eval --checkpoint $CKPT --max-batches 2 --out results/tta_smoke

# 7. Recalibration: ImageNet BatchNorm statistics -> GTSRB's. Written once and
#    kept, so the Source model is one file rather than something recomputed.
#    Writes <checkpoint>_bn.pkl beside the original.
!python -m iot.source_model --checkpoint $CKPT --data dataset/gtsrb/train

# 8. The study.
!python -m iot.stream_eval --checkpoint ${CKPT%.pkl}_bn.pkl \
    --gtsrb-c dataset/gtsrb_c --out results/tta \
    --batch-size 128 --num-workers 4
```

Then bring back the three small files - `results/tta/conditions.csv`,
`batches.csv`, `summary.json`. They are all the analysis needs.

---

## What "passed" looks like

| step | criterion |
|---|---|
| 5 `self_check` | returns `{'mean': ..., 'var': ...}` both **below 1e-6**, or raises. This catches the one bug that stays invisible otherwise: per-batch variances being averaged gives BatchNorm layers that are too narrow, and such a state still loads, still classifies, and only shows up as a disappointing number. |
| 6 branch walk | exits 0, `results/tta_smoke/summary.json` written, four arms in `accuracy_by_arm`, and `check_descriptor_independence` did **not** raise. |
| 7 recalibration | ~26640 images, ~4800 channels across the BatchNorm layers, `..._bn.pkl` written. Its `bn_stats_source` is no longer `imagenet`, and the **weights are unchanged bit for bit** - the pass moves buffers, never parameters. |
| 8 the study | `conditions.csv`, `batches.csv`, `summary.json`; bank of **13 states** (clean + 12 seen corruptions), ~460 KB; a finite threshold. |

---

## Traps

**Regenerate GTSRB-C here; do not upload it.** The dataset on the development
machine was built under numpy 1.26.4. Colab runs numpy 2, where `fog`
(`np.float_`) and `gaussian_blur` (`multichannel=`) fail - and they fail at
**call** time, so without `iot/corruption_shim.py` the build would quietly write
three empty conditions instead of stopping. Import `corrupt` through the shim,
never from the package, and check step 4's directory count.

**Two different splits, and swapping them is silent.** Step 4 corrupts
`dataset/gtsrb/test`; step 7 recalibrates on `dataset/gtsrb/train`. That is not a
slip. The recalibration stands in for the pass each client would run on **its own
clean training data** - one forward pass, no labels, no gradients - so the
training split is what it means. Recalibrating on `test` instead would compute
the model's normalisation statistics from the very images GTSRB-C is built out
of, so the Source arm would already carry the evaluation set's statistics and
every arm's number would be inflated by an amount nobody could separate from the
adaptation being measured. The two archives are disjoint (26640 / 12630), which
is what keeps that clean.

**The checkpoint pickle was written under numpy 2.** It references
`numpy._core.*`, so it cannot be unpickled by an environment pinned to
numpy 1.26.4 - which is what `requirements_dev.txt` creates on the development
machine. Colab is unaffected. Reading a `.pkl` brought back from here needs a
`numpy._core -> numpy.core` shim first.

**`--max-batches 2` cuts the recalibration and the bank build as well**, not just
the stream, and `evaluate` stamps `bn_stats_source` with `PARTIAL` to say so.
Step 6 walks branches; it does not produce data. Never read
`results/tta_smoke/*.csv` as a result.

**Recalibrate in step 7, not inline with `--recalibrate-on`.** Inline works, but
it redoes the 26640-image pass on every run and leaves no file that *is* the
Source model - so "what did the recalibration alone buy on clean images" stops
being answerable.

**Batch size 128, never 16.** Blind BN-adapt estimates its statistics from the
batch in front of it, and a small batch makes the baseline look worse than it is.
At 2000 images per condition this is ~16 batches, so the routing lag - the
decision for batch *i* is taken on batch *i-1*'s descriptor - costs one misrouted
batch per condition transition, about 6%. `batches.csv` records it rather than
hiding it.

---

## Reading the tables

One bar to clear before any difference means anything. The GTSRB test set is 421
physical signs at 30 frames each, so 2000 images per condition are not 2000
independent observations: `effective_sample_size` gives **n_eff ~ 419**, and the
95% half-width at accuracy 0.5 is **+-4.8 points, not +-2.2**.
`metrics.difference_is_significant` returns that half-width, so the write-up can
quote what a difference had to beat instead of asserting it.

Three numbers, in this order:

    blind - source      did adapting at all help
    oracle - blind      what the bank is worth
    routed - oracle     what the routing costs

The oracle arm exists to keep the last two apart: with only source / blind /
routed, "the bank contains useful states" and "the router finds them" are one
number. Report both or neither.

Expected and not a defect: the four **unseen** corruptions should mostly take the
fallback, and the `clean_again` visit at the end of the stream is where "what did
twelve corruptions cost on undegraded input" gets its answer.
