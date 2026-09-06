"""Building the bank: one BatchNorm state per known corruption.

This is the torch half of the routing work. `iot/routing.py` decides *which*
state to load; this module produces the states and the descriptors it decides
between, by pushing corrupted images through the frozen backbone and watching
what its normalization layers see.

Nothing here trains anything. There is no loss, no backward, no optimizer and
no label: a BatchNorm layer's statistics are means, not fitted parameters, so
they are learned by looking. That is the whole reason the adaptation stage is
cheap enough to belong on a device.

    build_bank(...)  ->  for each condition, one forward pass over its images
                         -> BNState      the running_mean / running_var to load
                         -> BNDescriptor the summary the routing compares
                         -> intra-condition distances, to calibrate the
                            fallback threshold from data instead of guessing

**How the statistics are accumulated, and the trap it avoids.** Over a whole
condition the state has to describe *all* its images, but the images arrive in
batches. The tempting route - average each batch's mean and variance - is wrong
for the variance: the mean of the within-batch variances ignores how much the
batch means differ from each other, so it underestimates, and the resulting BN
layers are too narrow and saturate. The identity that repairs it is the same one
the federated recalibration needs (total variance = mean of within-group
variances + variance of the group means).

This module sidesteps that entirely by accumulating **raw moments** - the count,
the sum and the sum of squares per channel - and forming the mean and variance
only once, at the end. There are no per-batch variances to average, so there is
no identity to get wrong. It is also less code.

Note that PyTorch's own accumulation does not do this: in `train()` mode
BatchNorm updates its buffers with an exponential moving average that depends on
the order of the batches, and even with `momentum=None` its cumulative average
averages the per-batch variances. So the buffers are written here explicitly
rather than harvested from a `train()` pass.

**A set of statistics has to be self-consistent, and measuring them all at once
does not make them so.** This is the trap that costs a whole run and raises
nothing. A layer's `running_mean` / `running_var` describe *its input*, and that
input is produced by the layers above it - which are themselves normalizing with
whatever statistics are loaded at the time. Measure all twenty of a ResNet18's
layers in one `eval()` pass and only the first is exact: `bn1` is fed by `conv1`,
which is frozen, but `layer1.0.bn2` is fed by `layer1.0.bn1`, whose output moves
the moment its state is replaced. Loading the whole set then puts every layer
after the first on statistics describing an input distribution that no longer
exists, and the mismatch compounds with depth.

    measure (eval, old buffers)          load all 20 at once
    conv1 -> [bn1]      exact              bn1 now normalizes differently
          -> [l1.0.bn1] stale              ...so this input moved
          -> [l1.0.bn2] worse              ...and this one moved more

That is one Jacobi step away from the old statistics, not a fixed point. The way
out is `collect_bn_state(..., method='sequential')`: measure one layer, write it,
move to the next, so every layer is measured with everything upstream already at
its final value. Exact by construction, and the argument is in `_sweep_bn_state`.
`method='batch-stats'` is the cheaper approximation - one pass with the layers
normalizing on the batch in hand - which lands about 38x closer than doing
nothing and is still not a fixed point, because during that pass the upstream
layers use per-batch statistics while the finished state uses pooled ones, and
the gap accumulates with depth. Iterating it does not help: it converges to the
batch-statistics fixed point, which is not the one `eval()` runs on.

The accumulation is the raw-moment one throughout, so the pooled variance is
always exact; what the three methods differ in is the distribution the moments
are taken over.

`state_residual` / `assert_state_is_fixed_point` are the invariant that catches
it: load a state, measure again, and require it not to move. `self_check` cannot
- both of its sides share one collection method, so it checks additivity, not
self-consistency.
"""

from collections import OrderedDict
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from iot.routing import BNDescriptor, BNState, descriptor_from_state, symmetric_kl

__all__ = [
    'DEFAULT_DESCRIPTOR_PREFIXES', 'BankEntry',
    'bn_modules', 'descriptor_layer_names', 'collect_bn_state',
    'load_bn_state', 'current_bn_state', 'build_bank', 'self_check',
    'restrict_state', 'check_descriptor_independence', 'DescriptorProbe',
    'state_residual', 'assert_state_is_fixed_point', 'COLLECT_METHODS',
]

# Which layers the descriptor is read from: the early stages only. They respond
# to noise, contrast and frequency content - what the degradation is. The deep
# stages respond to semantics, i.e. which sign is in the image, which is exactly
# what the routing has to be invariant to.
DEFAULT_DESCRIPTOR_PREFIXES: Tuple[str, ...] = ('bn1', 'layer1')

# Accumulators are float64 whatever the model's dtype: a sum of squares over
# ~30M values in float32 loses low-order bits, and the variance is a difference
# of two large numbers where exactly those bits live.
_ACCUM_DTYPE = torch.float64


def bn_modules(model: nn.Module) -> "OrderedDict[str, nn.Module]":
    """Every BatchNorm module, keyed by its dotted name, in definition order.

    For ResNet18 that is 20 modules and 4800 channels: `bn1` 64, `layer1` 256,
    `layer2` 640, `layer3` 1280, `layer4` 2560 (the deeper stages include the
    downsample BN of their first block).
    """
    found = OrderedDict()
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            found[name] = module
    if not found:
        raise ValueError(
            "this model has no BatchNorm layer, so there is nothing to adapt - "
            "ConvNet and AlexNet are in that position and cannot be used here")
    return found


def descriptor_layer_names(model: nn.Module,
                           prefixes: Sequence[str] = DEFAULT_DESCRIPTOR_PREFIXES
                           ) -> List[str]:
    """The BN layers the descriptor is built from, in a fixed order.

    The order is part of the contract between the bank and the runtime: the same
    list must be used on both sides or the distances compare different channels.
    """
    names = [name for name in bn_modules(model)
             if any(name == p or name.startswith(p + '.') for p in prefixes)]
    if not names:
        raise ValueError(
            f"no BatchNorm layer matches {list(prefixes)}; the model has "
            f"{list(bn_modules(model))[:5]}...")
    return names


def check_descriptor_independence(model: nn.Module,
                                 descriptor_layers: Sequence[str],
                                 adapted_layers: Sequence[str]) -> None:
    """Refuse a descriptor whose input depends on a state the router will swap.

    This is the subtle half of the "read the input, never the output" rule, and
    it is the one that survives a careless reading of it. Reading `bn1`'s input
    is safe: it is `conv1`'s output, and `conv1` is frozen, so the same batch
    always produces the same numbers there. Reading `layer1.0.bn2`'s input is
    **not** safe if `layer1.0.bn1` is one of the layers the bank overwrites -
    that input has already been normalized by whichever state is loaded, so the
    descriptor would move when the routing decision moves.

    What that failure looks like: the bank's descriptors were measured under the
    source state, the query's under the state chosen for the previous batch, and
    the two are no longer comparable. Distances inflate, the fallback fires on
    conditions the bank does contain, and the decision oscillates between
    identical batches. Nothing raises, and the run just looks disappointing.

    Two ways out, and this project takes the first:

        descriptor = ('bn1',)                adapt everything, describe from the
                                             one layer no state can reach
        adapt from 'layer2' onward           describe from `bn1` + `layer1`, and
                                             give up adapting the first stage

    Ordering uses `named_modules`, which for a ResNet is execution order and,
    where it is not (a downsample branch), is conservative - it can refuse a
    combination that would have been fine, never accept one that is not.

    Raises:
        ValueError: if any adapted BatchNorm precedes any descriptor layer.
    """
    order = {name: index for index, name in enumerate(bn_modules(model))}
    unknown = [n for n in list(descriptor_layers) + list(adapted_layers) if n not in order]
    if unknown:
        raise KeyError(f"not BatchNorm layers of this model: {unknown}")

    adapted = set(adapted_layers)
    for name in descriptor_layers:
        upstream = sorted(a for a in adapted if order[a] < order[name])
        if upstream:
            raise ValueError(
                f"descriptor layer '{name}' sits downstream of adapted layer(s) "
                f"{upstream[:3]}, so its input changes with the state that gets "
                f"loaded and the descriptor would depend on the decision it "
                f"drives; either describe from '{list(order)[0]}' alone or stop "
                f"adapting the layers before it")


def restrict_state(state: BNState, names: Sequence[str]) -> BNState:
    """The part of a state covering `names` only.

    Used to hold the first stage fixed while adapting the rest: a bank entry is
    collected over every BatchNorm layer, and this is what selects the subset
    that is actually written back into the model.
    """
    missing = [n for n in names if n not in state]
    if missing:
        raise KeyError(f"the state does not cover {missing}")
    return {name: state[name] for name in names}


class _StopForward(Exception):
    """The pass has what it was after; there is no reason to finish the network.

    Raised out of a hook and caught by the collection loop. Only used when the
    layers being measured are a prefix of the network - measuring `bn1` means
    running `conv1` and stopping, not running all of ResNet18 - which is what
    makes the layer-by-layer sweep affordable.
    """


class _InputMoments:
    """Raw per-channel moments of what each BatchNorm layer is fed.

    Hooks read the layer's **input**, never its output. The output is already
    normalized by whichever state is currently loaded, so a descriptor built
    from it would depend on the choice it is meant to drive - the routing would
    oscillate between identical batches, and nothing would raise.
    """

    def __init__(self, model: nn.Module, names: Optional[Sequence[str]] = None,
                 stop_when_complete: bool = False):
        modules = bn_modules(model)
        self.names = list(names) if names is not None else list(modules)
        self.stop_when_complete = stop_when_complete
        self._wanted = set(self.names)
        self._seen_this_batch: set = set()
        self._handles = []
        self._count: Dict[str, float] = {}
        self._sum: Dict[str, torch.Tensor] = {}
        self._sum_sq: Dict[str, torch.Tensor] = {}

        for name in self.names:
            module = modules[name]
            self._handles.append(
                module.register_forward_pre_hook(self._make_hook(name)))

    def begin_batch(self) -> None:
        """Tell the early exit that a new forward is starting."""
        self._seen_this_batch = set()

    def _make_hook(self, name: str):
        def hook(_module, inputs):
            x = inputs[0].detach().to(_ACCUM_DTYPE)
            # (N, C, H, W) -> per-channel; also tolerate (N, C) for a 1-D BN.
            dims = [0] + list(range(2, x.dim()))
            total = x.sum(dim=dims)
            total_sq = (x * x).sum(dim=dims)
            n = x.numel() / x.shape[1]
            if name in self._sum:
                self._sum[name] += total
                self._sum_sq[name] += total_sq
                self._count[name] += n
            else:
                self._sum[name] = total
                self._sum_sq[name] = total_sq
                self._count[name] = n
            if self.stop_when_complete:
                self._seen_this_batch.add(name)
                if self._wanted <= self._seen_this_batch:
                    raise _StopForward
        return hook

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def state(self) -> BNState:
        """mean = S/n, var = S2/n - mean^2, computed once at the end."""
        if not self._sum:
            raise RuntimeError("no batch was ever passed through the model")
        result: BNState = {}
        for name in self.names:
            n = self._count[name]
            mean = self._sum[name] / n
            var = self._sum_sq[name] / n - mean * mean
            # A tiny negative value here is float cancellation on a near-constant
            # channel, not a bug in the data.
            var = torch.clamp(var, min=0.0)
            result[name] = (mean.cpu().numpy(), var.cpu().numpy())
        return result


class DescriptorProbe:
    """The current batch's descriptor, read out of the pass that classifies it.

    This is what makes the runtime cost of routing honest. `collect_bn_state`
    would give the same numbers, but by running the network a second time - and
    a method whose selling point is "one forward pass, no gradient, no label"
    cannot afford a second forward pass to decide what to do. The hooks stay
    registered for the whole stream and the accumulators are reset between
    batches, so the descriptor is a by-product of work that had to happen
    anyway.

        with DescriptorProbe(model, layers) as probe:
            for images in stream:
                probe.reset()
                logits = model(images)
                descriptor = probe.descriptor()

    The layers it reads must be **upstream of everything the router swaps** -
    see `check_descriptor_independence`, which the caller is expected to have
    called first.
    """

    def __init__(self, model: nn.Module, layers: Sequence[str]):
        self._moments = _InputMoments(model, layers)
        self._layers = list(layers)

    def __enter__(self) -> "DescriptorProbe":
        return self

    def __exit__(self, *exception) -> None:
        self.close()

    def reset(self) -> None:
        """Forget the previous batch. Call before every forward."""
        self._moments._count.clear()
        self._moments._sum.clear()
        self._moments._sum_sq.clear()

    def descriptor(self, label: str = 'batch') -> BNDescriptor:
        """What the last forward pass saw, as a comparable descriptor."""
        return descriptor_from_state(self._moments.state(), self._layers, label)

    def close(self) -> None:
        self._moments.remove()


@contextmanager
def _batch_statistics(model: nn.Module, enabled: bool = True):
    """Normalize on the batch in hand while the moments are being measured.

    Only the BatchNorm modules switch, so the head's Dropout stays off and the
    pass remains a deterministic function of the data. The buffers are restored
    on exit: in this mode BatchNorm overwrites them with a moving average as it
    goes, and those are exactly the numbers being measured - letting the pass
    edit its own starting point would make the result depend on batch order.

    Why the measurement wants this at all is the module docstring's subject: a
    layer measured while the layers above it still carry the *old* statistics is
    describing an input distribution that will not exist once the new ones are
    loaded.
    """
    if not enabled:
        yield model
        return

    modules = bn_modules(model)
    saved = current_bn_state(model)
    previous_modes = {name: module.training for name, module in modules.items()}
    for module in modules.values():
        module.train()
    try:
        yield model
    finally:
        for name, module in modules.items():
            module.train(previous_modes[name])
        load_bn_state(model, saved)


COLLECT_METHODS: Tuple[str, ...] = ('as-is', 'batch-stats', 'sequential')


@torch.no_grad()
def collect_bn_state(model: nn.Module, loader, device: torch.device,
                     names: Optional[Sequence[str]] = None,
                     max_batches: Optional[int] = None,
                     method: str = 'as-is',
                     progress=None) -> BNState:
    """Measure what each BatchNorm layer is fed, and return it as a state.

    Dropout is off throughout - train mode would make each forward a different
    random function - and the model's own buffers are left as they were found.

    **Three methods, and the choice is a correctness one.** The difficulty is
    that the twenty layers are not twenty independent measurements: each one's
    input is produced by the layers above it, which are normalising with
    whatever is loaded at the time. See the module docstring for what that
    breaks.

        'as-is'        One pass, everything measured under the statistics the
                       model is carrying now. Correct only for layers fed by
                       frozen modules - `bn1`, whose input is `conv1`'s output.
                       This is the right choice for a **descriptor**, and it is
                       the default for that reason. Using it for a state that
                       gets loaded is the bug this module's docstring is about.

        'batch-stats'  One pass with the layers normalising on the batch in
                       hand, so each input is produced by upstream layers
                       already adapted to this data. Gets close - measured on a
                       ResNet18, it takes the residual from ~8.8 sigma to ~0.23
                       - but it is not exact: during the pass the upstream
                       layers use *per-batch* statistics while the finished
                       state uses *pooled* ones, and that gap accumulates with
                       depth. It converges to the batch-statistics fixed point,
                       which is not the one eval() runs on, so **iterating it
                       does not help**.

        'sequential'   Exact, by construction. One layer at a time in execution
                       order: measure it, write it, move on - so each layer is
                       measured with everything upstream already at its final
                       value, and nothing downstream can affect it. Costs one
                       pass per layer, but each pass stops as soon as the layer
                       it wants has been seen, so twenty layers cost about ten
                       full passes rather than twenty. **The right choice for
                       any state that gets loaded into the model.**

    Args:
        model: the network, already on `device`.
        loader: yields `(images, labels)`; the labels are ignored, because this
            stage has none to use. Must be re-iterable for 'sequential'.
        device: where to run.
        names: which BN layers to record. None records all of them, which is
            what a bank state needs; a descriptor uses a subset.
        max_batches: stop early, for a smoke check. Under 'sequential' the
            batches are snapshotted once and reused for every layer - a shuffled
            loader reshuffles on each pass, and layers measured on different
            subsets would not be a fixed point of anything.
        method: one of `COLLECT_METHODS`, above.
        progress: optional `f(done, total, layer_name)`, called by 'sequential'
            after each layer - it is the one method slow enough to want it.

    Returns:
        `{layer name: (mean, var)}` as numpy arrays.
    """
    if method not in COLLECT_METHODS:
        raise ValueError(f"unknown method {method!r}, expected one of "
                         f"{list(COLLECT_METHODS)}")
    if method == 'sequential':
        return _sweep_bn_state(model, loader, device, names=names,
                               max_batches=max_batches, progress=progress)

    was_training = model.training
    model.eval()
    moments = _InputMoments(model, names)
    try:
        with _batch_statistics(model, enabled=(method == 'batch-stats')):
            for index, batch in enumerate(loader):
                if max_batches is not None and index >= max_batches:
                    break
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                model(images.to(device, non_blocking=True))
    finally:
        moments.remove()
        model.train(was_training)
    return moments.state()


@torch.no_grad()
def _one_layer_moments(model: nn.Module, name: str, loader,
                       device: torch.device,
                       max_batches: Optional[int] = None) -> BNState:
    """One layer's input moments, running only as much of the network as needed.

    The hook raises `_StopForward` the moment it has recorded the batch, so a
    measurement of `bn1` costs `conv1` and nothing else. That is what makes a
    twenty-layer sweep cost about ten full passes instead of twenty.
    """
    was_training = model.training
    model.eval()
    moments = _InputMoments(model, [name], stop_when_complete=True)
    try:
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            moments.begin_batch()
            try:
                model(images.to(device, non_blocking=True))
            except _StopForward:
                pass
    finally:
        moments.remove()
        model.train(was_training)
    return moments.state()


def _sweep_bn_state(model: nn.Module, loader, device: torch.device,
                    names: Optional[Sequence[str]] = None,
                    max_batches: Optional[int] = None,
                    progress=None) -> BNState:
    """The exact fixed point: one layer at a time, in execution order.

    Why this is exact where one simultaneous pass is not. A layer's input
    depends on the layers **above** it and on nothing else - not on its own
    statistics, not on anything downstream. So if every upstream layer already
    carries its final value when the layer is measured, that measurement is the
    one the finished network will reproduce, and it stays true as the sweep
    continues past it. After one sweep every layer satisfies that, which is what
    "fixed point" means. Loading the whole set changes nothing, and
    `state_residual` comes back at float noise.

    `named_modules` order is used, which for a ResNet is execution order. A
    downsample branch is the one place it is not, and it is harmless: that
    branch is *parallel* to the two convolutions beside it and its input is the
    block's input, which is upstream of all three and finalised before any of
    them. Nothing in the block depends on the order the three are visited in.

    The model's buffers are restored on exit; the caller decides whether to
    apply the result.
    """
    modules = bn_modules(model)
    wanted = list(modules) if names is None else [n for n in modules if n in set(names)]
    unknown = [n for n in (names or []) if n not in modules]
    if unknown:
        raise KeyError(f"not BatchNorm layers of this model: {unknown}")

    if max_batches is not None:
        # Every layer must see the SAME data, or the sweep is a fixed point of
        # nothing. A shuffled loader reshuffles on each pass, so with a cut-off
        # the layers would each get a different subset. Snapshot once instead;
        # it is a smoke-sized number of batches by construction.
        loader = [batch for _, batch in zip(range(max_batches), loader)]
        max_batches = None

    saved = current_bn_state(model)
    result: BNState = {}
    try:
        for position, name in enumerate(wanted):
            measured = _one_layer_moments(model, name, loader, device,
                                          max_batches=max_batches)
            result[name] = measured[name]
            # Commit before measuring the next one - that is the whole method.
            load_bn_state(model, {name: measured[name]})
            if progress is not None:
                progress(position + 1, len(wanted), name)
    finally:
        load_bn_state(model, saved)
    return result


def current_bn_state(model: nn.Module) -> BNState:
    """The statistics the model is carrying right now.

    For a checkpoint straight out of federated training these are ImageNet's:
    BatchNorm's buffers are not parameters, so no aggregation round ever touched
    them. That state is the "source" the blending in `routing.blend` mixes
    against, and the baseline every adapted result is compared to.
    """
    return {name: (module.running_mean.detach().cpu().numpy().copy(),
                   module.running_var.detach().cpu().numpy().copy())
            for name, module in bn_modules(model).items()
            if module.running_mean is not None}


def load_bn_state(model: nn.Module, state: BNState, strict: bool = True) -> None:
    """Write a state into the model's BatchNorm buffers, in place.

    With the model in `eval()` this is the whole of the adaptation: the layers
    then normalize with these numbers instead of the ones they were shipped
    with. The trainable head is untouched, and has no BatchNorm of its own, so
    the two halves of the project never collide.
    """
    modules = bn_modules(model)
    for name, (mean, var) in state.items():
        if name not in modules:
            if strict:
                raise KeyError(f"the model has no BatchNorm layer '{name}'")
            continue
        module = modules[name]
        target = module.running_mean
        if target is None:
            # track_running_stats=False: the layer has no buffers to write into
            # and always uses batch statistics. Nothing to load, and silently
            # skipping it would make the bank look applied when it was not.
            raise ValueError(
                f"layer '{name}' keeps no running statistics, so a bank state "
                f"cannot be loaded into it")
        if target.shape != torch.Size(mean.shape):
            raise ValueError(
                f"layer '{name}': state has {mean.shape} channels, model wants "
                f"{tuple(target.shape)} - these came from different models")
        module.running_mean.copy_(torch.as_tensor(mean, dtype=target.dtype,
                                                  device=target.device))
        module.running_var.copy_(torch.as_tensor(var, dtype=target.dtype,
                                                 device=target.device))


@torch.no_grad()
def state_residual(model: nn.Module, state: BNState, loader,
                   device: torch.device,
                   max_batches: Optional[int] = None) -> Dict[str, float]:
    """How far a state moves when you load it and measure again.

    This is the invariant that `self_check` structurally cannot provide. Load
    the state, run the same data through in plain `eval()` - the regime the
    state will actually be used in - and see what each layer's input looks like
    *now*. A state that describes the network it is loaded into does not move.
    One that was measured under different statistics upstream does, and the
    further down the network the worst, which is the shape of the failure.

    Two numbers, chosen so they mean something rather than being relative errors
    on quantities that straddle zero:

        mean_shift   |mu_new - mu| / sigma, i.e. how far the layer's input mean
                     moved **in units of that channel's own spread**. This is
                     the scale BatchNorm itself normalizes on, so 0.05 really is
                     "a twentieth of a standard deviation".
        var_ratio    max(v_new/v, v/v_new) - 1, the worst relative widening or
                     narrowing. 0.10 is "one variance is 10% off the other".

    Returns:
        `{'mean_shift', 'var_ratio', 'mean_layer', 'var_layer'}` - the two worst
        values and the layers they were found in. The layer names matter: a
        residual concentrated in the deep stages is the compounding described in
        the module docstring, while one spread evenly is more likely to be the
        data simply differing from what the state was built on.
    """
    saved = current_bn_state(model)
    try:
        load_bn_state(model, state)
        again = collect_bn_state(model, loader, device, names=list(state),
                                 max_batches=max_batches)
    finally:
        load_bn_state(model, saved)

    worst = {'mean_shift': 0.0, 'var_ratio': 0.0,
             'mean_layer': '', 'var_layer': ''}
    for name, (mean, var) in state.items():
        new_mean, new_var = again[name]
        sigma = np.sqrt(np.maximum(var, 1e-12))
        shift = float(np.max(np.abs(new_mean - mean) / sigma))
        floor = 1e-12
        ratio = np.maximum(new_var, floor) / np.maximum(var, floor)
        ratio = float(np.max(np.maximum(ratio, 1.0 / ratio))) - 1.0
        if shift > worst['mean_shift']:
            worst['mean_shift'], worst['mean_layer'] = shift, name
        if ratio > worst['var_ratio']:
            worst['var_ratio'], worst['var_layer'] = ratio, name
    return worst


def assert_state_is_fixed_point(model: nn.Module, state: BNState, loader,
                                device: torch.device,
                                mean_tolerance: float = 0.05,
                                var_tolerance: float = 0.10,
                                max_batches: Optional[int] = None
                                ) -> Dict[str, float]:
    """`state_residual`, raising if the state does not describe its own model.

    Cheap - one extra pass - and it is the check that turns the silent failure
    of the module docstring into a stop. Run it once on the recalibration and
    once on a bank entry; if those hold, the rest of the bank was built the same
    way.

    Raises:
        AssertionError: if the state moves by more than the tolerances.
    """
    worst = state_residual(model, state, loader, device, max_batches=max_batches)
    if worst['mean_shift'] > mean_tolerance or worst['var_ratio'] > var_tolerance:
        raise AssertionError(
            f"this state is not a fixed point of the model it was loaded into: "
            f"the input mean of '{worst['mean_layer']}' moves by "
            f"{worst['mean_shift']:.3f} sigma (tolerance {mean_tolerance}) and "
            f"the variance of '{worst['var_layer']}' by "
            f"{worst['var_ratio'] * 100:.1f}% (tolerance {var_tolerance * 100:.0f}%). "
            f"Collect it with method='sequential', which measures one layer at "
            f"a time with everything upstream already final; anything else "
            f"leaves every layer below the first describing an input "
            f"distribution that no longer exists")
    return worst


class BankEntry:
    """One known condition: what to load, and how to recognize it."""

    __slots__ = ('label', 'state', 'descriptor')

    def __init__(self, label: str, state: BNState, descriptor: BNDescriptor):
        self.label = label
        self.state = state
        self.descriptor = descriptor

    def __repr__(self) -> str:
        return f"BankEntry({self.label!r}, {self.descriptor.num_channels} channels)"


def build_bank(model: nn.Module, loaders: "OrderedDict[str, object]",
               device: torch.device,
               descriptor_prefixes: Sequence[str] = DEFAULT_DESCRIPTOR_PREFIXES,
               calibration_batches: int = 8,
               max_batches: Optional[int] = None,
               method: str = 'sequential',
               progress=None
               ) -> Tuple[List[BankEntry], List[float]]:
    """Build one entry per condition, plus the distances to calibrate on.

    `loaders` maps a condition label to its DataLoader, and **must include
    `clean`**. A bank of corruptions only would send a device that is looking at
    clean images to the nearest corruption and leave it worse off than doing
    nothing at all.

    The second return value is the intra-condition distances: for every
    condition, the descriptor of a few individual batches measured against that
    condition's own stored state. They are the spread of "a batch that really
    does belong here", and `routing.calibrate_threshold` turns them into the
    fallback threshold - so the threshold comes from data rather than taste.

    Args:
        max_batches: cut every condition short. For a smoke run only - a state
            built from two batches describes two batches, and the threshold
            calibrated beside it is meaningless. Never set it for a bank that
            will produce a number.
        method: how the entry states are measured. 'sequential' is the only
            exact setting for a state that gets loaded back into the model - see
            `collect_bn_state`. 'batch-stats' is a cheaper approximation and
            'as-is' reproduces the earlier, broken behavior; both remain
            reachable so the difference can be measured rather than asserted.
        progress: optional `f(done, total, layer)`, passed to each condition's
            sweep - the slow part of a bank build.

    Returns:
        `(entries, intra_distances)`.
    """
    if 'clean' not in loaders:
        raise ValueError(
            "the bank has no 'clean' entry; without it a device looking at "
            "undegraded images is routed to the nearest corruption")

    layer_names = descriptor_layer_names(model, descriptor_prefixes)
    entries: List[BankEntry] = []
    intra: List[float] = []

    for label, loader in loaders.items():
        state = collect_bn_state(model, loader, device, max_batches=max_batches,
                                 method=method, progress=progress)
        descriptor = descriptor_from_state(state, layer_names, label)
        entries.append(BankEntry(label, state, descriptor))

        # A handful of single-batch descriptors against the entry just built.
        # One batch is exactly what the router will see at runtime, so this
        # measures the right thing rather than a smoothed version of it.
        #
        # Deliberately 'as-is', for the same reason the runtime descriptor is:
        # these layers are `bn1`, fed by the frozen `conv1`, so no method
        # changes their numbers - and reading them exactly the way the stream
        # reads them is what keeps the two comparable.
        for index, batch in enumerate(loader):
            if index >= calibration_batches:
                break
            batch_state = collect_bn_state(model, [batch], device, names=layer_names)
            batch_descriptor = descriptor_from_state(batch_state, layer_names, label)
            intra.append(symmetric_kl(batch_descriptor, descriptor))

    return entries, intra


@torch.no_grad()
def self_check(model: nn.Module, loader, device: torch.device,
              tolerance: float = 1e-6) -> Dict[str, float]:
    """Verify the accumulation against the same statistics computed in one go.

    The batched accumulation is the one piece here that can be wrong while
    looking right - a state that is quietly too narrow still loads, still
    classifies, and only shows up as a disappointing number. This runs the same
    data twice, once batch by batch and once as a single concatenated batch, and
    the two must agree. Cheap enough to run on a few hundred images before
    trusting a whole bank.

    **What it catches, and what it cannot.** Both sides go through the same
    `_InputMoments`, so this is a check on *additivity*, not on the formula:

        caught      per-batch variances being averaged (the D5-shaped bug this
                    exists for: the one-batch reference has nothing to average,
                    so the two sides diverge immediately); any return to
                    PyTorch's own `train()` accumulation, whose exponential
                    moving average depends on batch order; float32 accumulators
                    losing the low-order bits the variance is a difference of;
                    accumulator state leaking between batches
        NOT caught   a wrong reduction axis, a wrong element count, a wrong
                    variance identity - all three would be wrong *identically*
                    on both sides and the check would pass

    That is acceptable because the second group fails loudly elsewhere - a wrong
    axis gives a state whose shape does not match the layer and `load_bn_state`
    raises - while the first group is the family that fails silently.

    Returns the largest relative discrepancy found, per statistic.
    """
    batches = [b[0] if isinstance(b, (tuple, list)) else b for b in loader]
    if not batches:
        raise ValueError("the loader yielded nothing")

    incremental = collect_bn_state(model, [(b,) for b in batches], device)
    at_once = collect_bn_state(model, [(torch.cat(batches, dim=0),)], device)

    worst = {'mean': 0.0, 'var': 0.0}
    for name, (mean, var) in incremental.items():
        reference_mean, reference_var = at_once[name]
        scale_mean = np.maximum(np.abs(reference_mean), 1e-8)
        scale_var = np.maximum(np.abs(reference_var), 1e-8)
        worst['mean'] = max(worst['mean'],
                            float(np.max(np.abs(mean - reference_mean) / scale_mean)))
        worst['var'] = max(worst['var'],
                           float(np.max(np.abs(var - reference_var) / scale_var)))

    if worst['mean'] > tolerance or worst['var'] > tolerance:
        raise AssertionError(
            f"batched accumulation disagrees with the single-pass reference: "
            f"mean {worst['mean']:.2e}, var {worst['var']:.2e} (tolerance {tolerance:.0e})")
    return worst
