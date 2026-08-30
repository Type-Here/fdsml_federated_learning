"""The experiment: a deployed model meeting corrupted input, four ways.

This is where the inference half finally produces numbers. A single machine
loads the checkpoint Part A produced, and classifies the corrupted test set as a
**stream** - conditions arriving one after another, no labels, no gradients, no
second chances - under four arms that differ only in what they do to the
normalisation layers:

    source    nothing. The model as Part A left it. The floor.
    blind     BN-adapt: every BatchNorm normalises on the batch in front of it.
              Knows nothing, needs nothing, and is the baseline to beat.
    routed    read the batch's descriptor, pick a state from the bank, or refuse
              and fall back to `blind`. The method.
    oracle    the bank, with the true corruption handed over for free. The
              ceiling the routing could reach if its decisions were perfect.

**Four arms and not three, and the fourth is the important one.** With only
source / blind / routed, "the bank contains useful states" and "the router finds
them" are one number and cannot be separated. If routed beats blind, which of
the two worked? The oracle answers it: `oracle - blind` is what the bank is
worth, `routed - oracle` is what the routing costs. Reporting the first without
the second is the mistake this arm exists to prevent.

---

**The protocol is continuous** - the stream crosses from one condition to the
next without a reset, and whatever adaptation the model has accumulated carries
over. That is the realistic edge scenario, and it is also the only version in
which "how much did it forget about clean images after seeing twelve
corruptions" is a question with an answer, which is why `clean` is visited again
at the end by default.

**The routing decision is taken on the *previous* batch's descriptor.** It has
to be: choosing a normalisation state for the current batch would require having
already pushed the current batch through the network, which is the forward pass
the choice is supposed to configure. One batch of lag is the price of one
forward pass instead of two, and on a stream where the condition changes every
few hundred batches it costs exactly one misrouted batch per transition - which
the results record rather than hide.

**The descriptor is read from `bn1` alone here, not from `bn1` + `layer1`.**
This is the one place where the offline bank and the online stream disagree, and
the reason is structural rather than a preference. `bn1`'s input is `conv1`'s
output: `conv1` is frozen, so the same batch always produces the same numbers
there, whatever state is loaded. `layer1`'s inputs are downstream of `bn1` and
are therefore already normalised by whichever state the router picked last time
- a descriptor built from them would move when the decision moves, which is the
circularity that makes a router oscillate between identical batches without ever
raising. `bn_bank.check_descriptor_independence` turns that into an error, and
is called before every stream.

Run it:

    python -m iot.stream_eval --checkpoint checkpoints/<run>.pkl \\
                              --gtsrb-c dataset/gtsrb_c \\
                              --out results/tta
"""

import argparse
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from iot import metrics
from iot.bn_adapt import adapt_batchnorm
from iot.bn_bank import (
    DescriptorProbe,
    build_bank,
    state_residual,
    check_descriptor_independence,
    current_bn_state,
    descriptor_layer_names,
    load_bn_state,
)
from iot.gtsrb_c import (
    CLEAN_CONDITION,
    CORRUPTIONS_SEEN,
    REVISIT_SUFFIX,
    condition_name,
    parse_condition,
)
from iot.routing import (
    BNDescriptor,
    BNState,
    blend,
    calibrate_threshold,
    l2_distance,
    route,
    symmetric_kl,
)
from iot.source_model import (
    bn_stats_from_checkpoint,
    build_model,
    image_folder_loader,
    load_checkpoint,
    recalibrate,
)

__all__ = [
    'ARMS', 'ONE_PASS_DESCRIPTOR_PREFIXES', 'BANK_SEVERITY',
    'ArmResult', 'BatchRecord',
    'REVISIT_SUFFIX',
    'bank_label_for_condition', 'severity_of', 'condition_is_seen',
    'build_bank_from_disk', 'run_arm', 'evaluate', 'main',
]

ARMS: Tuple[str, ...] = ('source', 'blind', 'routed', 'oracle')

# See the module docstring: the only BatchNorm whose input no loaded state can
# reach. The offline bank may use more, because nothing is being swapped while
# it is built; the stream may not.
ONE_PASS_DESCRIPTOR_PREFIXES: Tuple[str, ...] = ('bn1',)

# The bank is indexed by corruption, not by (corruption, severity): twelve
# states rather than thirty-six, and a method that does not presume to know how
# intense the degradation is. Severity 3 is the middle one it is built from.
BANK_SEVERITY = 3

# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def bank_label_for_condition(condition: str) -> str:
    """Which bank entry a stream condition should be looked up under.

    `fog_s5` -> `fog`: the bank is built at one severity and indexed by
    corruption, so a method that meets `fog_s5` is not assumed to know how thick
    the fog is. `parse_condition` does the string work.
    """
    return parse_condition(condition)[0]


def severity_of(condition: str) -> Optional[int]:
    """The severity in a condition name, or None for `clean`."""
    return parse_condition(condition)[1]


def condition_is_seen(condition: str, bank_labels: Sequence[str]) -> bool:
    """Is there a right answer for this condition in the bank?

    False for the four unseen corruptions, and that is not a gap in the
    experiment - it is the control. For those the correct routing decision is to
    refuse, and `metrics.routing_report` scores refusals separately from hits
    precisely so the two cannot be averaged into one flattering number.
    """
    return bank_label_for_condition(condition) in set(bank_labels)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class BatchRecord:
    """One batch of the stream, under one arm."""

    condition: str
    arm: str
    n_images: int
    correct: int
    seconds: float
    chosen: Optional[str] = None      # None on a fallback or an unrouted arm
    fallback: bool = False
    distance: float = float('nan')
    margin: float = float('nan')
    state_loaded: bool = False
    # False for the one batch at the head of the stream, where there is no
    # previous descriptor to decide from. Counting it as a fallback would put a
    # refusal in the table that the router never made - and on the `clean` row,
    # which is only sixteen batches long, one phantom refusal is six percent.
    decided: bool = True


@dataclass
class ArmResult:
    """Everything one arm produced over the whole stream."""

    arm: str
    records: List[BatchRecord] = field(default_factory=list)
    y_true: Dict[str, List[int]] = field(default_factory=dict)
    y_pred: Dict[str, List[int]] = field(default_factory=dict)
    num_state_loads: int = 0

    def conditions(self) -> List[str]:
        """Visit order, with a repeat visit kept distinct from the first."""
        seen, order = set(), []
        for record in self.records:
            if record.condition not in seen:
                seen.add(record.condition)
                order.append(record.condition)
        return order


# ---------------------------------------------------------------------------
# The bank, from what is on disk
# ---------------------------------------------------------------------------

def build_bank_from_disk(model, transform, root: str,
                         corruptions: Sequence[str] = CORRUPTIONS_SEEN,
                         severity: int = BANK_SEVERITY,
                         batch_size: int = 128, num_workers: int = 2,
                         max_batches_per_condition: Optional[int] = None,
                         percentile: float = 95.0,
                         shuffle: bool = True, seed: int = 42,
                         device: Optional[torch.device] = None):
    """One state per known corruption, plus the threshold, from GTSRB-C on disk.

    `clean` is a bank entry and not an afterthought: a bank of corruptions only
    would route a device that is looking at undegraded images to the nearest
    corruption and leave it worse off than doing nothing. `build_bank` refuses a
    bank without it.

    The threshold is calibrated from the intra-condition distances the bank
    build measures - the distance from a single batch of fog to the fog state,
    and so on - at the 95th percentile. That gives the fallback rule a sentence
    instead of a constant: *if you are further away than 95% of batches are from
    their own corruption, I do not recognise you.*

    The loaders are **shuffled**, with a fixed seed, for the same reason the
    stream is. The pooled state does not care - it sees every image either way -
    but the intra-condition distances that set the threshold are measured on
    single batches, and `ImageFolder` walks the class directories in order, so
    an unshuffled batch of 128 holds two or three sign types out of 43. A
    threshold calibrated on class-homogeneous batches does not describe the
    class-mixed batches the router is then asked to judge.

    Returns:
        `(entries, threshold, intra_distances)`.
    """
    device = device or next(model.parameters()).device
    loaders: "OrderedDict[str, object]" = OrderedDict()
    loaders[CLEAN_CONDITION] = image_folder_loader(
        os.path.join(root, CLEAN_CONDITION), transform, batch_size,
        shuffle=shuffle, num_workers=num_workers, seed=seed)
    for corruption in corruptions:
        directory = os.path.join(root, condition_name(corruption, severity))
        if not os.path.isdir(directory):
            raise FileNotFoundError(
                f"{directory} does not exist; generate GTSRB-C first with "
                f"`python -m iot.gtsrb_c`")
        loaders[corruption] = image_folder_loader(
            directory, transform, batch_size, shuffle=shuffle,
            num_workers=num_workers, seed=seed)

    entries, intra = build_bank(model, loaders, device,
                                descriptor_prefixes=ONE_PASS_DESCRIPTOR_PREFIXES,
                                max_batches=max_batches_per_condition)
    return entries, calibrate_threshold(intra, percentile), intra


# ---------------------------------------------------------------------------
# The stream
# ---------------------------------------------------------------------------

def _synchronize(device: torch.device) -> None:
    """Make a wall-clock measurement mean something on a GPU.

    CUDA kernels are queued asynchronously, so without this every batch would
    appear to take the time it costs to *enqueue* the work, and the whole
    latency table would be a measurement of Python.
    """
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def run_arm(model, arm: str, conditions: "OrderedDict[str, object]",
            device: torch.device,
            bank=None, threshold: Optional[float] = None,
            source_state: Optional[BNState] = None,
            distance=symmetric_kl, alpha: float = 1.0,
            descriptor_layers: Optional[Sequence[str]] = None,
            max_batches_per_condition: Optional[int] = None) -> ArmResult:
    """Run the whole stream once, under one arm.

    The model is put back on `source_state` before the arm starts, so the four
    arms are independent measurements of the same starting point. Without that
    the second arm would inherit whatever the first left in the buffers - blind
    BN-adapt overwrites them as it runs - and the comparison would depend
    silently on the order the arms were evaluated in.

    Args:
        model: the network, on `device`, already recalibrated.
        arm: one of `ARMS`.
        conditions: label -> DataLoader, in the order the stream visits them.
        bank: the `BankEntry` list, required for `routed` and `oracle`.
        threshold: fallback distance for `routed`. None disables the fallback,
            which is worth running once to measure what the fallback is worth.
        source_state: what to restore before the arm. Defaults to whatever the
            model is carrying when this is called.
        distance: `symmetric_kl` (default) or `l2_distance`.
        alpha: interpolation between the chosen state and the source, 1.0 being
            a plain swap.
        descriptor_layers: BN layers the descriptor is read from. Defaults to
            `bn1` alone - see the module docstring.
        max_batches_per_condition: cut every condition short, for a smoke run.

    Returns:
        An `ArmResult`.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm '{arm}', expected one of {list(ARMS)}")
    if arm in ('routed', 'oracle') and not bank:
        raise ValueError(f"the '{arm}' arm needs a bank to choose from")

    source_state = source_state or current_bn_state(model)
    load_bn_state(model, source_state)
    model.eval()

    result = ArmResult(arm=arm)
    needs_descriptor = (arm == 'routed')

    bank_descriptors = [entry.descriptor for entry in bank] if bank else []
    bank_by_label = {entry.label: entry for entry in bank} if bank else {}

    if needs_descriptor:
        layers = list(descriptor_layers) if descriptor_layers is not None else \
            descriptor_layer_names(model, ONE_PASS_DESCRIPTOR_PREFIXES)
        # Every BatchNorm is swappable in this design, so the descriptor has to
        # be upstream of all of them. Checked rather than assumed: getting it
        # wrong produces a working, oscillating router.
        check_descriptor_independence(model, layers,
                                      list(current_bn_state(model).keys()))
        probe = DescriptorProbe(model, layers)
    else:
        probe = None

    previous_descriptor: Optional[BNDescriptor] = None
    loaded_label: Optional[str] = None

    try:
        for condition, loader in conditions.items():
            true_label = bank_label_for_condition(condition)
            result.y_true.setdefault(condition, [])
            result.y_pred.setdefault(condition, [])

            for index, (images, labels) in enumerate(loader):
                if (max_batches_per_condition is not None
                        and index >= max_batches_per_condition):
                    break
                images = images.to(device, non_blocking=True)

                # The clock starts before the decision, not after it. Everything
                # the arm does to earn its accuracy has to be inside: the
                # distance computation against the bank, the copy of the chosen
                # state into the buffers, the descriptor read out of the pass.
                # Timing only the forward would make every arm cost the same and
                # turn the efficiency table - the thing the IoT half is graded on
                # - into a measurement of the backbone.
                _synchronize(device)
                started = time.perf_counter()

                blind = (arm == 'blind')
                chosen: Optional[str] = None
                fallback = False
                best_distance = float('nan')
                margin = float('nan')
                target_label: Optional[str] = None
                decided = (arm in ('routed', 'oracle'))

                if arm == 'routed':
                    if previous_descriptor is None:
                        # Nothing has been seen yet, so there is nothing to
                        # decide from. Starting on the source state rather than
                        # on an arbitrary bank entry is the conservative choice,
                        # and it costs exactly one batch per stream.
                        target_label = None
                        decided = False
                    else:
                        decision = route(previous_descriptor, bank_descriptors,
                                         distance=distance, threshold=threshold)
                        best_distance, margin = decision.distance, decision.margin
                        fallback = decision.fallback
                        if fallback:
                            blind = True
                        else:
                            chosen = decision.label
                            target_label = decision.label
                elif arm == 'oracle':
                    if true_label in bank_by_label:
                        chosen = true_label
                        target_label = true_label
                    else:
                        # An unseen corruption: the bank has nothing right to
                        # offer, so the oracle gets the fallback too. Anything
                        # else would be an oracle over a bank that does not
                        # exist, and would inflate the ceiling.
                        fallback = True
                        blind = True

                state_loaded = False
                if target_label is not None and target_label != loaded_label:
                    entry = bank_by_label[target_label]
                    state = (entry.state if alpha == 1.0
                             else blend(entry.state, source_state, alpha))
                    load_bn_state(model, state)
                    loaded_label = target_label
                    state_loaded = True
                    result.num_state_loads += 1
                elif blind:
                    # Blind mode overwrites the running buffers as it goes, so
                    # whatever was loaded is gone and the next routed batch must
                    # load again rather than assume.
                    loaded_label = None

                if probe is not None:
                    probe.reset()
                with adapt_batchnorm(model, enabled=blind, keep_running_stats=False):
                    logits = model(images)
                predictions = logits.argmax(dim=1)
                if probe is not None:
                    previous_descriptor = probe.descriptor(condition)
                _synchronize(device)
                elapsed = time.perf_counter() - started

                truth = labels.numpy()
                predicted = predictions.cpu().numpy()
                result.y_true[condition].extend(truth.tolist())
                result.y_pred[condition].extend(predicted.tolist())
                result.records.append(BatchRecord(
                    condition=condition, arm=arm, n_images=int(truth.size),
                    correct=int(np.sum(truth == predicted)), seconds=elapsed,
                    chosen=chosen, fallback=fallback,
                    distance=best_distance, margin=margin,
                    state_loaded=state_loaded, decided=decided))
    finally:
        if probe is not None:
            probe.close()
        load_bn_state(model, source_state)

    return result


# ---------------------------------------------------------------------------
# Assembling the results
# ---------------------------------------------------------------------------

def _rows_for_arm(result: ArmResult, bank_labels: Sequence[str],
                  clean_condition: str = CLEAN_CONDITION) -> List[metrics.ConditionRow]:
    """One `ConditionRow` per condition for one arm, intervals included."""
    clean_accuracy = None
    if clean_condition in result.y_true:
        truth = np.asarray(result.y_true[clean_condition])
        predicted = np.asarray(result.y_pred[clean_condition])
        clean_accuracy = float(np.mean(truth == predicted)) if truth.size else None

    by_condition: Dict[str, List[BatchRecord]] = {}
    for record in result.records:
        by_condition.setdefault(record.condition, []).append(record)

    rows = []
    for condition, records in by_condition.items():
        scores = metrics.classification_metrics(result.y_true[condition],
                                                result.y_pred[condition])
        low, high = metrics.accuracy_interval(scores['accuracy'], scores['n_images'])
        corruption = bank_label_for_condition(condition)
        seen = condition_is_seen(condition, bank_labels)

        decided = [r for r in records if r.decided]
        routing = metrics.routing_report(
            chosen=[r.chosen for r in decided],
            truth=[corruption if seen else None for _ in decided],
            distances=[r.distance for r in decided if np.isfinite(r.distance)],
            margins=[r.margin for r in decided if np.isfinite(r.margin)],
        ) if result.arm in ('routed', 'oracle') and decided else {}

        latency = metrics.latency_summary([r.seconds for r in records],
                                          scores['n_images'])
        rows.append(metrics.ConditionRow(
            arm=result.arm,
            condition=condition,
            corruption=corruption,
            severity=severity_of(condition),
            family=metrics.CORRUPTION_FAMILIES.get(corruption, 'other'),
            seen=seen,
            n_images=scores['n_images'],
            correct=scores['correct'],
            accuracy=scores['accuracy'],
            macro_f1=scores['macro_f1'],
            weighted_f1=scores['weighted_f1'],
            accuracy_low=low,
            accuracy_high=high,
            effective_n=metrics.effective_sample_size(scores['n_images']),
            retention=(metrics.retention(scores['accuracy'], clean_accuracy)
                       if clean_accuracy else None),
            fallback_rate=routing.get('fallback_rate'),
            hit_rate=routing.get('hit_rate'),
            ms_per_image=latency['ms_per_image'],
            extra={
                'median_ms_per_batch': latency['median_ms_per_batch'],
                'p95_ms_per_batch': latency['p95_ms_per_batch'],
                'num_batches': latency['num_batches'],
                'refusal_rate': routing.get('refusal_rate'),
                'mean_distance': routing.get('mean_distance'),
                'mean_margin': routing.get('mean_margin'),
                'state_loads': sum(r.state_loaded for r in records),
            },
        ))
    return rows


def _model_cost(model, image_size: int, device: torch.device) -> Dict[str, float]:
    """Multiply-accumulates and parameters for one forward pass.

    `thop` is optional - it lives in `requirements_gpu.txt` and not in the
    development set - so a missing import degrades to the parameter count rather
    than stopping an evaluation that has already run.

    The number this table is really for is the *comparison*: the adaptation adds
    no multiply-accumulates at all, because it changes buffers and not the graph.
    The whole cost of the method is the descriptor and the distance computation,
    which is why they are timed separately rather than folded in here.
    """
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cost = {'parameters': float(parameters), 'trainable_parameters': float(trainable)}
    try:
        import thop
    except ImportError:
        cost['macs'] = float('nan')
        cost['macs_note'] = 'thop not installed'
        return cost
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)
    macs, _ = thop.profile(model, inputs=(dummy,), verbose=False)
    cost['macs'] = float(macs)
    return cost


def evaluate(checkpoint_path: str, gtsrb_c_root: str, output_dir: str,
             arms: Sequence[str] = ARMS,
             batch_size: int = 128,
             severities: Sequence[int] = (1, 3, 5),
             bank_severity: int = BANK_SEVERITY,
             distance_name: str = 'symmetric_kl',
             alpha: float = 1.0,
             percentile: float = 95.0,
             revisit_clean: bool = True,
             recalibrate_on: Optional[str] = None,
             num_workers: int = 2,
             shuffle_stream: bool = True,
             seed: int = 42,
             max_batches_per_condition: Optional[int] = None,
             device: Optional[str] = None) -> Dict:
    """The whole experiment: build the bank, run every arm, write the tables.

    Writes `conditions.csv` (one row per arm per condition), `batches.csv` (one
    row per batch, which is what the routing behavior has to be read from) and
    `summary.json` (the bank, the threshold, the efficiency budget).

    `recalibrate_on` is the clean training set. If the checkpoint does not
    already carry GTSRB statistics and this is not given, the run still happens
    but every arm starts from an ImageNet-normalized backbone, and the numbers
    then mix "we fixed ImageNet -> GTSRB" with "we fixed clean -> corrupted".
    The summary records which of the two situations produced it.

    `shuffle_stream` is on, with a fixed `seed`, so the run stays reproducible
    while each batch is a mixed sample of the 43 classes. Unshuffled,
    `ImageFolder` hands out the class directories in order and a 128-image batch
    covers two or three sign types - which handicaps blind BN-adapt, whose whole
    estimate is that batch, and puts semantics into a descriptor whose job is to
    describe the degradation. The protocol is unchanged either way: batch *i-1*
    still decides for batch *i*.
    """
    checkpoint = load_checkpoint(checkpoint_path)
    metadata = checkpoint['metadata']
    model, manager = build_model(metadata, checkpoint['weights'], device=device)
    torch_device = manager.device
    transform = manager.transform_pipeline
    image_size = int(metadata.get('image_size', 128))

    # 1. The normalisation the whole comparison starts from.
    bn_source = metadata.get('bn_stats_source', 'imagenet')
    carried = bn_stats_from_checkpoint(metadata)
    clean_loader = None
    if recalibrate_on:
        clean_loader = image_folder_loader(recalibrate_on, transform, batch_size,
                                           shuffle=shuffle_stream,
                                           num_workers=num_workers, seed=seed)
    if carried is not None:
        load_bn_state(model, carried)
    elif clean_loader is not None:
        recalibrate(model, loader=clean_loader, device=torch_device,
                    max_batches=max_batches_per_condition)
        bn_source = 'gtsrb-train-pooled (this run)'
        if max_batches_per_condition is not None:
            bn_source += f' - PARTIAL, {max_batches_per_condition} batches only'
    source_state = current_bn_state(model)

    # Is the state the whole comparison starts from a fixed point of the model
    # carrying it? One extra pass answers it, and the answer belongs in the
    # summary rather than in somebody's memory: a state that moves when it is
    # loaded describes a different network, and every arm built on it then
    # classifies at chance without anything raising. Reported, not enforced -
    # ImageNet's statistics legitimately move on GTSRB data, and that number is
    # itself the size of the shift the recalibration exists to remove.
    fixed_point = None
    if clean_loader is not None:
        fixed_point = state_residual(model, source_state, clean_loader,
                                     torch_device,
                                     max_batches=max_batches_per_condition)
        print(f"  starting statistics ({bn_source}): input mean moves "
              f"{fixed_point['mean_shift']:.4f} sigma, variance "
              f"{fixed_point['var_ratio'] * 100:.2f}%")

    # 2. The bank, and the threshold read off its own intra-condition spread.
    entries, threshold, intra = build_bank_from_disk(
        model, transform, gtsrb_c_root, severity=bank_severity,
        batch_size=batch_size, num_workers=num_workers,
        max_batches_per_condition=max_batches_per_condition,
        percentile=percentile, shuffle=shuffle_stream, seed=seed,
        device=torch_device)
    load_bn_state(model, source_state)
    bank_labels = [entry.label for entry in entries]

    # 3. The stream: clean, then every condition on disk, then clean again so
    #    that what the adaptation cost on undegraded input is measurable.
    conditions: "OrderedDict[str, object]" = OrderedDict()
    available = sorted(d for d in os.listdir(gtsrb_c_root)
                       if os.path.isdir(os.path.join(gtsrb_c_root, d)))
    ordered = [CLEAN_CONDITION] + [c for c in available if c != CLEAN_CONDITION
                                   and severity_of(c) in severities]
    for condition in ordered:
        conditions[condition] = image_folder_loader(
            os.path.join(gtsrb_c_root, condition), transform, batch_size,
            shuffle=shuffle_stream, num_workers=num_workers, seed=seed)
    if revisit_clean:
        conditions[CLEAN_CONDITION + REVISIT_SUFFIX] = image_folder_loader(
            os.path.join(gtsrb_c_root, CLEAN_CONDITION), transform, batch_size,
            shuffle=shuffle_stream, num_workers=num_workers, seed=seed)

    distance = symmetric_kl if distance_name == 'symmetric_kl' else l2_distance

    # 4. The arms.
    os.makedirs(output_dir, exist_ok=True)
    rows: List[metrics.ConditionRow] = []
    batch_records: List[BatchRecord] = []
    arm_results: Dict[str, ArmResult] = {}
    for arm in arms:
        started = time.time()
        result = run_arm(model, arm, conditions, torch_device,
                         bank=entries, threshold=threshold,
                         source_state=source_state, distance=distance,
                         alpha=alpha,
                         max_batches_per_condition=max_batches_per_condition)
        arm_results[arm] = result
        rows.extend(_rows_for_arm(result, bank_labels))
        batch_records.extend(result.records)
        print(f"  {arm:8s} {time.time() - started:6.1f} s, "
              f"{len(result.records)} batches, {result.num_state_loads} state loads")

    # 5. The tables.
    conditions_csv = metrics.rows_to_csv(rows, os.path.join(output_dir, 'conditions.csv'))
    batches_csv = os.path.join(output_dir, 'batches.csv')
    import csv as _csv
    with open(batches_csv, 'w', newline='') as handle:
        writer = _csv.DictWriter(handle, fieldnames=list(vars(batch_records[0])))
        writer.writeheader()
        for record in batch_records:
            writer.writerow(vars(record))

    summary = {
        'checkpoint': checkpoint_path,
        'checkpoint_metadata': {k: v for k, v in metadata.items() if k != 'bn_stats'},
        'bn_stats_source': bn_source,
        'bn_fixed_point': fixed_point,
        'gtsrb_c_root': gtsrb_c_root,
        'arms': list(arms),
        'batch_size': batch_size,
        'severities': list(severities),
        'shuffle_stream': shuffle_stream,
        'seed': seed,
        'bank': {
            'labels': bank_labels,
            'severity': bank_severity,
            'threshold': threshold,
            'threshold_percentile': percentile,
            'intra_distance_median': float(np.median(intra)),
            'footprint': metrics.bank_footprint([e.state for e in entries]),
        },
        'descriptor': {
            'prefixes': list(ONE_PASS_DESCRIPTOR_PREFIXES),
            'channels': entries[0].descriptor.num_channels,
            'distance': distance_name,
            'alpha': alpha,
        },
        'cost': _model_cost(model, image_size, torch_device),
        'accuracy_by_arm': {
            arm: sum(r.correct for r in rows if r.arm == arm)
                 / max(sum(r.n_images for r in rows if r.arm == arm), 1)
            for arm in arms
        },
        'families': {arm: metrics.summarize_by_family(rows, arm) for arm in arms},
        'files': {'conditions': conditions_csv, 'batches': batches_csv},
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w') as handle:
        json.dump(summary, handle, indent=2, default=str)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a federated checkpoint on GTSRB-C, four arms.")
    parser.add_argument('--checkpoint', required=True,
                        help="the .pkl written by the aggregator")
    parser.add_argument('--gtsrb-c', default='dataset/gtsrb_c',
                        help="root of the corrupted dataset")
    parser.add_argument('--out', default='results/tta',
                        help="where the tables go")
    parser.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS))
    parser.add_argument('--batch-size', type=int, default=128,
                        help="64-128, not 16: blind BN-adapt estimates its "
                             "statistics from this batch and a small one makes "
                             "the baseline look worse than it is")
    parser.add_argument('--severities', type=int, nargs='+', default=[1, 3, 5])
    parser.add_argument('--bank-severity', type=int, default=BANK_SEVERITY)
    parser.add_argument('--distance', default='symmetric_kl',
                        choices=['symmetric_kl', 'l2'])
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="1.0 swaps the state outright; below 1 interpolates "
                             "with the source state")
    parser.add_argument('--percentile', type=float, default=95.0)
    parser.add_argument('--recalibrate-on', default=None,
                        help="clean training set, e.g. dataset/gtsrb/train. "
                             "Skipped if the checkpoint already carries GTSRB "
                             "statistics")
    parser.add_argument('--no-revisit-clean', action='store_true',
                        help="do not visit clean again at the end; the cost of "
                             "adaptation on undegraded input then goes unmeasured")
    parser.add_argument('--max-batches', type=int, default=None,
                        help="cut the recalibration, the bank build and every "
                             "condition short. For a smoke run only: it walks "
                             "every branch in minutes, and every number it "
                             "produces is meaningless")
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42,
                        help="fixes the stream order, so a shuffled run is "
                             "still reproducible batch for batch")
    parser.add_argument('--no-shuffle', action='store_true',
                        help="visit the images in ImageFolder order, i.e. class "
                             "by class. A batch then holds two or three of the "
                             "43 sign types, which handicaps blind BN-adapt and "
                             "puts semantics in the routing descriptor - a "
                             "stress case worth reporting, not the default")
    parser.add_argument('--device', default=None)
    args = parser.parse_args(argv)

    summary = evaluate(
        checkpoint_path=args.checkpoint, gtsrb_c_root=args.gtsrb_c,
        output_dir=args.out, arms=args.arms, batch_size=args.batch_size,
        severities=args.severities, bank_severity=args.bank_severity,
        distance_name=('symmetric_kl' if args.distance == 'symmetric_kl' else 'l2'),
        alpha=args.alpha, percentile=args.percentile,
        revisit_clean=not args.no_revisit_clean,
        recalibrate_on=args.recalibrate_on, num_workers=args.num_workers,
        shuffle_stream=not args.no_shuffle, seed=args.seed,
        max_batches_per_condition=args.max_batches, device=args.device)

    print(f"\nBN statistics: {summary['bn_stats_source']}")
    if summary.get('bn_fixed_point'):
        residual = summary['bn_fixed_point']
        print(f"  fixed point: mean {residual['mean_shift']:.4f} sigma, "
              f"variance {residual['var_ratio'] * 100:.2f}% - a large residual "
              f"means the states describe a different network")
    print(f"Bank: {len(summary['bank']['labels'])} states, "
          f"{summary['bank']['footprint']['total_kilobytes']:.0f} KB, "
          f"threshold {summary['bank']['threshold']:.4g}")
    print("\naccuracy over the whole stream")
    for arm, accuracy in summary['accuracy_by_arm'].items():
        print(f"  {arm:8s} {accuracy:.4f}")
    print(f"\n-> {summary['files']['conditions']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
