"""Turning predictions into the numbers the write-up can defend.

Torch-free on purpose, the same cut as `routing.py` against `bn_bank.py`: the
forward passes need a GPU, but deciding what a number means is arithmetic that
fails quietly, so it lives where it can be run and tested on any machine.

Three groups of things live here.

**Quality.** Accuracy and F1 per condition, and the comparison between two arms
of the experiment. The important function is not `classification_metrics` - it
is `difference_is_significant`, which exists because the obvious reading of this
dataset is wrong (see below).

**The efficiency budget**, which is what the IoT half is actually graded on:
how large the bank is in bytes, how long a batch takes, how much of that the
routing decision costs.

**The correctness of the routing itself**, kept apart from the accuracy it
produces. "The bank works" and "the routing works" are two claims, and a table
of accuracies alone confounds them - which is why the evaluation has an oracle
arm and why `routing_report` counts decisions rather than predictions.

---

**The one number that changes every conclusion: n is not the image count.**

GTSRB's test set is 12630 images, and every per-class count is an exact multiple
of 30 because it is 421 physical signs photographed 30 times each. Thirty
consecutive frames of the same sign are near-duplicates: they do not carry
thirty independent pieces of evidence about whether the model can classify that
sign. GTSRB shuffled the filenames, so the track identity is not recoverable and
the grouping cannot be undone - but it is still there, and a confidence interval
computed on 12630 is roughly five times too narrow.

    naive        +- 1.96 * sqrt(p(1-p)/12630)   ~ +- 0.9 points at p = 0.5
    with n_eff   +- 1.96 * sqrt(p(1-p)/421)     ~ +- 4.8 points at p = 0.5

So a two-point gap between two adaptation methods is **not** a result, and every
interval in this module is computed on `effective_sample_size`, not on the
number of files.
"""

import csv
from dataclasses import asdict, dataclass, field
from math import sqrt
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score

__all__ = [
    'GTSRB_TEST_IMAGES', 'GTSRB_TEST_TRACKS', 'FRAMES_PER_TRACK',
    'CORRUPTION_FAMILIES',
    'ConditionRow',
    'classification_metrics', 'effective_sample_size', 'wilson_interval',
    'accuracy_interval', 'difference_is_significant', 'retention',
    'bank_footprint', 'latency_summary', 'routing_report',
    'summarize_by_family', 'rows_to_csv',
]

# The structure of the held-out set, measured rather than assumed: every
# per-class count in `dataset/gtsrb/test` is a multiple of 30, and
# 12630 / 30 = 421.
GTSRB_TEST_IMAGES = 12630
GTSRB_TEST_TRACKS = 421
FRAMES_PER_TRACK = 30

# The ImageNet-C family grouping, so a table of 37 conditions can be reported as
# four lines without inventing a taxonomy at write-up time.
CORRUPTION_FAMILIES: Dict[str, str] = {
    'gaussian_noise': 'noise', 'shot_noise': 'noise', 'impulse_noise': 'noise',
    'speckle_noise': 'noise',
    'defocus_blur': 'blur', 'motion_blur': 'blur', 'zoom_blur': 'blur',
    'glass_blur': 'blur', 'gaussian_blur': 'blur',
    'snow': 'weather', 'frost': 'weather', 'fog': 'weather',
    'brightness': 'digital', 'contrast': 'digital', 'pixelate': 'digital',
    'jpeg_compression': 'digital', 'elastic_transform': 'digital',
    'spatter': 'weather', 'saturate': 'digital',
    'clean': 'clean',
}


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """Accuracy and both F1 averages for one condition under one arm.

    Two F1 averages rather than one, because they answer different questions on
    a set this imbalanced (GTSRB's rarest class has a tenth of the commonest):

        macro     every class counts the same. Falls when the model gives up on
                  the rare signs, which is exactly what corruption does first.
        weighted  every *image* counts the same. Comparable with Part A's
                  numbers, which `model_manager.validate` reports weighted.

    Accuracy is the headline anyway - it is what the -C benchmarks report and
    what `accuracy_interval` can put an honest interval around.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"{y_true.size} labels against {y_pred.size} predictions")
    if y_true.size == 0:
        raise ValueError("no predictions to score")

    correct = int(np.sum(y_true == y_pred))
    return {
        'n_images': int(y_true.size),
        'correct': correct,
        'accuracy': correct / y_true.size,
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }


def effective_sample_size(n_images: int,
                          n_tracks: int = GTSRB_TEST_TRACKS,
                          frames_per_track: int = FRAMES_PER_TRACK,
                          rho: float = 1.0) -> float:
    """How many independent observations `n_images` GTSRB test images are worth.

    The standard cluster-sampling correction. `rho` is the intra-track
    correlation - how much two frames of the same sign say the same thing:

        n_eff = n / (1 + (m - 1) * rho)        m = images per track in the sample

    `rho = 1` is the default and the conservative reading: thirty frames of one
    physical sign count as **one** observation, so `n_eff` collapses to the
    number of distinct tracks the sample touches. `rho = 0` would give back the
    image count. The truth is in between and is not measurable here, because the
    track ids were stripped from the test filenames - so the interval is
    reported at `rho = 1` and the assumption is stated rather than hidden.

    Args:
        n_images: images actually evaluated (2000 per condition by default).
        n_tracks: physical signs in the source set.
        frames_per_track: frames per sign, 30 for GTSRB.
        rho: intra-track correlation in [0, 1].

    Returns:
        The effective count, between 1 and `n_images`.
    """
    if n_images <= 0:
        raise ValueError(f"n_images must be positive, got {n_images}")
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}")

    total = n_tracks * frames_per_track
    # Expected number of distinct tracks touched when drawing n_images of the
    # total without replacement: each track contributes unless all of its frames
    # are missed. At n = 2000 of 12630 this is ~419 of 421, so subsampling costs
    # almost nothing in independent evidence - which is the whole argument for
    # 2000 images per condition rather than 12630.
    if n_images >= total:
        touched = float(n_tracks)
    else:
        miss = (1.0 - n_images / total) ** frames_per_track
        touched = n_tracks * (1.0 - miss)
    touched = max(touched, 1.0)

    images_per_track = n_images / touched
    design_effect = 1.0 + (images_per_track - 1.0) * rho
    return max(n_images / design_effect, 1.0)


def wilson_interval(successes: float, n: float, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a proportion.

    Wilson and not the textbook `p +- z*sqrt(p(1-p)/n)`: with n_eff around 421
    and accuracies that can sit near 0.95 under mild corruption or near 0.05
    under severe, the normal approximation runs off the end of [0, 1] and
    reports intervals that include impossible values. Wilson does not, and costs
    one more line.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = (z / denominator) * sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def accuracy_interval(accuracy: float, n_images: int, z: float = 1.96,
                      **effective_kwargs) -> Tuple[float, float]:
    """The interval around one accuracy, on the effective sample size.

    This is the function to quote from. Passing `n_images` straight to
    `wilson_interval` instead would produce the five-times-too-narrow interval
    the module docstring warns about.
    """
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError(f"accuracy must be in [0, 1], got {accuracy}")
    n_eff = effective_sample_size(n_images, **effective_kwargs)
    return wilson_interval(accuracy * n_eff, n_eff, z=z)


def difference_is_significant(accuracy_a: float, accuracy_b: float,
                              n_images_a: int, n_images_b: Optional[int] = None,
                              z: float = 1.96, **effective_kwargs) -> Dict[str, float]:
    """Is the gap between two arms bigger than the noise, on n_eff?

    The two arms are evaluated on the **same images** (one subsample shared by
    every condition, by construction in `gtsrb_c.py`), so this is a paired
    comparison and treating it as unpaired is conservative - it will call some
    real differences insignificant, never the reverse. Conservative is the right
    direction here: the claim being protected is "our method is better", and
    that claim should have to clear the higher bar.

    Returns a dict rather than a bool so the write-up can quote the half-width
    that the difference had to beat, which is more informative than the verdict.
    """
    n_images_b = n_images_a if n_images_b is None else n_images_b
    n_a = effective_sample_size(n_images_a, **effective_kwargs)
    n_b = effective_sample_size(n_images_b, **effective_kwargs)

    difference = accuracy_a - accuracy_b
    standard_error = sqrt(accuracy_a * (1 - accuracy_a) / n_a
                          + accuracy_b * (1 - accuracy_b) / n_b)
    half_width = z * standard_error
    return {
        'difference': difference,
        'half_width': half_width,
        'effective_n_a': n_a,
        'effective_n_b': n_b,
        'significant': bool(abs(difference) > half_width),
    }


def retention(accuracy_corrupted: float, accuracy_clean: float) -> float:
    """The share of clean accuracy that survives the corruption.

    Reported alongside the raw accuracy because the two answer different
    questions. A method that lifts a condition from 0.20 to 0.30 and one that
    lifts another from 0.80 to 0.90 both gain ten points; only the ratio says
    which of them recovered most of what was lost.
    """
    if accuracy_clean <= 0:
        raise ValueError("clean accuracy must be positive to divide by it")
    return accuracy_corrupted / accuracy_clean


# ---------------------------------------------------------------------------
# The efficiency budget
# ---------------------------------------------------------------------------

def bank_footprint(states: Iterable[Dict[str, Tuple[np.ndarray, np.ndarray]]],
                   dtype_bytes: int = 4) -> Dict[str, float]:
    """How much memory the bank costs on the device.

    `dtype_bytes = 4` (float32) is the deployment figure and the one to quote.
    The states are carried in float64 in this process because they are
    accumulated in float64 - see `bn_bank` - but a device stores them at the
    model's own precision, and quoting the float64 size would double the number
    for no reason.

    Returns the total, the per-state size and the channel count. For ResNet18
    that is 4800 channels, so ~38 KB per state and ~460 KB for a bank of twelve
    - a number small enough to state as an argument rather than an apology.
    """
    states = list(states)
    if not states:
        raise ValueError("an empty bank has no footprint worth reporting")

    channels = [sum(np.asarray(mean).size for mean, _ in state.values())
                for state in states]
    if len(set(channels)) > 1:
        raise ValueError(
            f"the states cover different channel counts ({sorted(set(channels))}); "
            f"they did not come from the same model")

    per_state = channels[0] * 2 * dtype_bytes   # a mean and a variance each
    return {
        'num_states': len(states),
        'channels_per_state': channels[0],
        'bytes_per_state': float(per_state),
        'kilobytes_per_state': per_state / 1024.0,
        'total_bytes': float(per_state * len(states)),
        'total_kilobytes': per_state * len(states) / 1024.0,
    }


def latency_summary(durations: Sequence[float], images: int) -> Dict[str, float]:
    """Wall-clock cost of a stream, per batch and per image.

    The durations cover everything an arm does per batch - the distance
    computation against the bank, the copy of a chosen state into the buffers,
    the descriptor read out of the forward - and not the forward alone, or every
    arm would cost the same and this table would be a measurement of the
    backbone rather than of the method.

    The median and the 95th percentile, not the mean alone: the first batch of
    an arm pays for CUDA context setup and cuDNN autotuning and is several times
    slower than the rest, so a mean over a short stream is mostly that one
    batch. A device's budget is about the typical batch and the bad batch.

    Args:
        durations: seconds per batch, in stream order.
        images: total images classified, for the per-image figure.
    """
    values = np.asarray(list(durations), dtype=np.float64)
    if values.size == 0:
        raise ValueError("no timings to summarize")
    if images <= 0:
        raise ValueError(f"images must be positive, got {images}")
    return {
        'num_batches': int(values.size),
        'total_seconds': float(values.sum()),
        'mean_ms_per_batch': float(values.mean() * 1e3),
        'median_ms_per_batch': float(np.median(values) * 1e3),
        'p95_ms_per_batch': float(np.percentile(values, 95) * 1e3),
        'first_batch_ms': float(values[0] * 1e3),
        'ms_per_image': float(values.sum() * 1e3 / images),
    }


# ---------------------------------------------------------------------------
# The routing, scored on its own
# ---------------------------------------------------------------------------

def routing_report(chosen: Sequence[Optional[str]],
                   truth: Sequence[Optional[str]],
                   distances: Optional[Sequence[float]] = None,
                   margins: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Score the decisions, not the predictions they lead to.

    A batch is one decision. `truth[i]` is the bank label the batch really
    deserves, or **None** when the condition is one of the unseen corruptions -
    for those there is no right entry, and the right answer is to refuse.

    So the two rates are not two views of one thing:

        hit_rate         over batches that HAVE a right answer: did it pick it?
        refusal_rate     over batches that DO NOT: did it fall back?

    A method that scored well on the first and badly on the second would be a
    router that is confidently wrong whenever it meets something new, and a
    single "accuracy" number would hide exactly that.

    `chosen[i]` is the label the router picked, or None for a fallback.
    """
    chosen = list(chosen)
    truth = list(truth)
    if len(chosen) != len(truth):
        raise ValueError(f"{len(chosen)} decisions against {len(truth)} truths")
    if not chosen:
        raise ValueError("no decisions to score")

    routable = [(c, t) for c, t in zip(chosen, truth) if t is not None]
    unroutable = [c for c, t in zip(chosen, truth) if t is None]

    report: Dict[str, float] = {
        'num_decisions': len(chosen),
        'fallback_rate': sum(c is None for c in chosen) / len(chosen),
        'num_routable': len(routable),
        'num_unroutable': len(unroutable),
    }
    if routable:
        report['hit_rate'] = sum(c == t for c, t in routable) / len(routable)
        report['fallback_rate_when_routable'] = (
            sum(c is None for c, _ in routable) / len(routable))
    if unroutable:
        report['refusal_rate'] = sum(c is None for c in unroutable) / len(unroutable)
    if distances is not None and len(distances):
        report['mean_distance'] = float(np.mean(np.asarray(distances, dtype=np.float64)))
    if margins is not None and len(margins):
        finite = np.asarray([m for m in margins if np.isfinite(m)], dtype=np.float64)
        if finite.size:
            report['mean_margin'] = float(finite.mean())
    return report


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

@dataclass
class ConditionRow:
    """One line of the results table: one arm on one condition.

    Flat and stringly-typed on purpose - it goes straight to CSV, and the
    analysis afterwards is a spreadsheet or pandas, not this module.
    """

    arm: str
    condition: str
    corruption: str
    severity: Optional[int]
    family: str
    seen: bool
    n_images: int
    correct: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    accuracy_low: float = 0.0
    accuracy_high: float = 0.0
    effective_n: float = 0.0
    retention: Optional[float] = None
    fallback_rate: Optional[float] = None
    hit_rate: Optional[float] = None
    ms_per_image: Optional[float] = None
    extra: Dict[str, float] = field(default_factory=dict)


def summarize_by_family(rows: Sequence[ConditionRow], arm: str) -> Dict[str, Dict[str, float]]:
    """Collapse 37 conditions into the four ImageNet-C families, per arm.

    The mean is weighted by image count, so a family with more conditions in the
    table does not get more weight per image. `clean` is reported as its own
    family and never folded into an average of corruptions - averaging it in
    would flatter every method by the same amount and hide the one comparison
    that matters, which is each condition against clean.
    """
    grouped: Dict[str, List[ConditionRow]] = {}
    for row in rows:
        if row.arm != arm:
            continue
        grouped.setdefault(row.family, []).append(row)

    summary = {}
    for family, members in grouped.items():
        total = sum(r.n_images for r in members)
        summary[family] = {
            'num_conditions': len(members),
            'n_images': total,
            'accuracy': sum(r.correct for r in members) / total,
            'macro_f1': sum(r.macro_f1 * r.n_images for r in members) / total,
        }
    return summary


def rows_to_csv(rows: Sequence[ConditionRow], path: str) -> str:
    """Write the table, flattening `extra` into its own columns.

    Every arm and condition ends up in one file so the comparison is a
    `groupby`, not a join across four files that can silently disagree about
    which subsample they used.
    """
    if not rows:
        raise ValueError("nothing to write")

    extra_keys = sorted({key for row in rows for key in row.extra})
    base_keys = [k for k in asdict(rows[0]) if k != 'extra']

    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=base_keys + extra_keys)
        writer.writeheader()
        for row in rows:
            record = {k: v for k, v in asdict(row).items() if k != 'extra'}
            record.update({key: row.extra.get(key) for key in extra_keys})
            writer.writerow(record)
    return path
