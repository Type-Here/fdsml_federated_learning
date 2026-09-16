# TTA run — the test-time adaptation experiment

**Why it exists.** The Part A checkpoint carries ImageNet's BatchNorm statistics
(buffers are never federated) and has only ever seen clean images. This notebook
recalibrates it on clean GTSRB, builds a corrupted test set, and runs four
adaptation arms over it — the experiment that produces every Part B number.

**The four arms.** `source` adapts nothing. `blind` normalises on the batch in hand.
`routed` reads a descriptor off `bn1`, finds the nearest of the bank's 13 states by
symmetric KL, and loads it — or refuses and falls back to `blind`. `oracle` is handed
the true corruption's state. The oracle exists so that *"the bank is worth
something"* and *"the routing finds it"* stay separate questions: report
`oracle − blind` and `routed − oracle`, or neither.

**The checkpoint** (post-label-fix, not one of the four void ones):

    gtsrb_ResNet18_FedAvg_a0.5_c4_le1_seed42_20260829-185510.pkl
    ResNet18, num_custom_layers 2, 128 px, 43 classes, 142379 parameters
    best round 27:  f1 0.6446   acc 0.5844   loss 1.5841
    bn_stats: null   bn_stats_source: "imagenet"

## Cells

```bash
# 1. Clone the working branch EXPLICITLY. On main, main() takes no argument and
#    the config on the command line is ignored. Use %cd, not !cd: !cd opens a
#    subshell that dies with the cell.
!git clone -b features/feddisco <repo-url>
%cd fdsml_federated_learning

# 2. Only what Colab is missing. Do NOT install requirements_gpu.txt as-is:
#    re-pinning numpy==1.26.4 breaks the preinstalled CUDA torch.
!pip install imagecorruptions thop "setuptools<81"

# 3. Clean GTSRB, BOTH splits - the two halves below need different ones:
#      dataset/gtsrb/test   -> what step 4 corrupts       (12630 images)
#      dataset/gtsrb/train  -> what step 6 recalibrates on (26640, disjoint)
!python datasets_prep/prepare_gtsrb.py
!ls dataset/gtsrb                    # must list both: train, test

# 4. GTSRB-C, regenerated rather than uploaded (2.1 GB, ~2-5 min, seeded per
#    (image, corruption, severity) so it is reproducible byte for byte)
!python -m iot.gtsrb_c
!ls dataset/gtsrb_c | wc -l          # must be 50: 49 conditions + manifest.json
```

```python
# 5. The BatchNorm state must be a FIXED POINT: measure it, load it, measure
#    again in plain eval(), and it must not move. A state that moves describes a
#    different network - every layer below the first was measured while the
#    layers above still carried the old statistics. This is what voided the
#    first run, and the three methods side by side ARE a result, not a debug aid.
import itertools
from iot.bn_bank import (assert_state_is_fixed_point, collect_bn_state,
                         state_residual)
from iot.source_model import build_model, image_folder_loader, load_checkpoint

CKPT = ("models/second_run_checkpoints/"
        "gtsrb_ResNet18_FedAvg_a0.5_c4_le1_seed42_20260829-185510.pkl")

checkpoint = load_checkpoint(CKPT)
model, manager = build_model(checkpoint['metadata'], checkpoint['weights'])
loader = image_folder_loader('dataset/gtsrb/train', manager.transform_pipeline,
                             batch_size=128, num_workers=2)
few = list(itertools.islice(loader, 8))

for method in ('as-is', 'batch-stats', 'sequential'):
    state = collect_bn_state(model, few, manager.device, method=method)
    residual = state_residual(model, state, few, manager.device)
    print(f"{method:12s} mean {residual['mean_shift']:9.6f} sigma   "
          f"var {residual['var_ratio'] * 100:9.3f}%   "
          f"worst {residual['var_layer']}")

# Only the last has to hold, and it is the one every state is built with.
exact = collect_bn_state(model, few, manager.device, method='sequential')
assert_state_is_fixed_point(model, exact, few, manager.device)
```

```bash
# The path, once, so the cells below are copy-pasteable
%env CKPT=models/second_run_checkpoints/gtsrb_ResNet18_FedAvg_a0.5_c4_le1_seed42_20260829-185510.pkl

# 6. Recalibration: ImageNet BatchNorm statistics -> GTSRB's. Writes
#    <checkpoint>_bn.pkl beside the original, so the Source model is one file
#    rather than something recomputed. Sweeps the 20 layers one at a time and
#    raises rather than writing a state that is not a fixed point.
!python -m iot.source_model --checkpoint $CKPT --data dataset/gtsrb/train

# 7. The branch walk: same branches as the full run - fallback, bootstrap batch,
#    unseen corruptions, four arms - in minutes. Its numbers are meaningless.
!python -m iot.stream_eval --checkpoint $CKPT --max-batches 2 --out results/tta_smoke

# 8. The study.
!python -m iot.stream_eval --checkpoint ${CKPT%.pkl}_bn.pkl \
    --gtsrb-c dataset/gtsrb_c --out results/tta \
    --batch-size 128 --num-workers 4

# 9. The control: the same stream on the ORIGINAL checkpoint (ImageNet
#    statistics), source arm only. This is what makes "the recalibration bought
#    N points on clean images" a measured sentence. Cheapest cell here; without
#    it a quietly broken Source baseline looks exactly like a weak one.
!python -m iot.stream_eval --checkpoint $CKPT \
    --gtsrb-c dataset/gtsrb_c --out results/tta_imagenet \
    --arms source --batch-size 128 --num-workers 4
```

Bring back `results/tta/conditions.csv`, `batches.csv`, `summary.json` — they are all
the analysis needs.

## What "passed" looks like

| step | criterion |
|---|---|
| 5 | `as-is` ~8.8 sigma (what voided the first run), `batch-stats` ~0.23, `sequential` at float noise ~1e-6. `assert_state_is_fixed_point` must not raise on the last. If `sequential` is not ~1e-6, stop — nothing below it is worth running |
| 6 | ~26 640 images, ~4800 channels, `..._bn.pkl` written, a `fixed point:` line printed, `bn_stats_source` no longer `imagenet`, and the **weights unchanged bit for bit** — the pass moves buffers, never parameters |
| 7 | exits 0, `summary.json` written, four arms in `accuracy_by_arm`, and `check_descriptor_independence` did **not** raise |
| 8 | `conditions.csv`, `batches.csv`, `summary.json`; bank of **13 states** (clean + 12 seen corruptions), ~460 KB; a finite threshold. One arithmetic check before reading anything: **`source` on clean must beat `blind` on clean** — pooled statistics over 26 640 images losing to a 128-image estimate is the fixed-point bug returning |
| 9 | `source` arm only; its `bn_fixed_point` residual is expected to be **large** — ImageNet's statistics genuinely do not describe GTSRB, and that is the size of the shift step 6 removes |

## Traps

**Two different splits, and swapping them is silent.** Step 4 corrupts
`dataset/gtsrb/test`; step 6 recalibrates on `dataset/gtsrb/train`. Recalibrating on
`test` would compute the normalisation statistics from the very images GTSRB-C is
built out of, inflating every arm by an amount nobody could separate from the
adaptation being measured. The two archives are disjoint (26 640 / 12 630).

**Regenerate GTSRB-C here; do not upload it.** Colab runs numpy 2, where `fog`
(`np.float_`) and `gaussian_blur` (`multichannel=`) fail — and they fail at **call**
time, so without `iot/corruption_shim.py` the build would quietly write three empty
conditions instead of stopping. Check step 4's directory count.

**The stream is shuffled, and that is not cosmetic.** `ImageFolder` walks the class
directories in order, so an unshuffled batch of 128 out of a 2000-image condition
holds two or three of the 43 sign types — and blind adaptation's entire estimate *is*
that batch. `--seed` keeps a shuffled run reproducible; `--no-shuffle` is a stress
case, a finding if reported as one and a confound if not.

**Batch size 128, not 16.** Blind adaptation estimates from the batch in front of it.
At 2000 images per condition this is ~16 batches, so the routing lag — the decision
for batch *i* is taken on batch *i−1*'s descriptor — costs one misrouted batch per
condition transition. `batches.csv` records it rather than hiding it.

**`--max-batches 2` cuts the recalibration and the bank build too**, not just the
stream, and `evaluate` stamps `bn_stats_source` with `PARTIAL` to say so. Never read
`results/tta_smoke/*.csv` as a result.

**The checkpoint pickle was written under numpy 2**, so it references
`numpy._core.*` and cannot be unpickled by an environment pinned to numpy 1.26.4.
Colab is unaffected; reading one locally needs a `numpy._core -> numpy.core` shim.

## Reading the tables

The GTSRB test set is **421 physical signs at 30 frames each**, so 2000 images per
condition are not 2000 independent observations: `effective_sample_size` gives
`n_eff ≈ 419`, and the 95% half-width at accuracy 0.5 is **±4.8 points, not ±2.2**.
`metrics.difference_is_significant` returns that half-width, so the write-up can
quote what a difference had to beat.

Three numbers, in this order:

    blind  - source     did adapting at all help
    oracle - blind      what the bank is worth
    routed - oracle     what the routing costs

Expected and not a defect: the four **unseen** corruptions mostly take the fallback,
and the `clean_again` visit at the end of the stream is where "what did twelve
corruptions cost on undegraded input" gets its answer.
