"""The baselines: adapting normalization without a bank, or with a chosen state.

Two ways of running the model at test time, both a single forward pass with no
gradient and no label:

    adapt_batchnorm(model)     blind BN-adapt. The layers normalize on the
                               statistics of the batch in front of them,
                               whatever it is. Knows nothing, needs nothing.

    load_bn_state(model, s)    load a state from the bank (in `bn_bank`), then
                               run in plain eval(). This is what the routing
                               decision turns into.

Blind BN-adapt is the baseline the bank has to beat. It is also the fallback the
router falls back *to* when nothing in the bank is close enough, which is why
the two live in the same module.

**Why the head is safe.** With `num_custom_layers > 0` the trainable classifier
is `Linear -> ReLU -> Dropout -> ... -> Linear` and contains no BatchNorm, while
every BatchNorm sits in the frozen backbone. So federated training moves
parameters this module never touches, and this module moves buffers federated
training never touched. The two halves of the project compose without
interfering - which is a claim worth making, not an accident.
"""

from contextlib import contextmanager
from typing import Iterator, Optional

import torch
import torch.nn as nn

from iot.bn_bank import bn_modules, current_bn_state, load_bn_state

__all__ = ['adapt_batchnorm', 'bn_adapt_forward']


@contextmanager
def adapt_batchnorm(model: nn.Module, enabled: bool = True,
                    keep_running_stats: bool = False) -> Iterator[nn.Module]:
    """Put **only** the BatchNorm layers into batch-statistics mode.

    Three things this gets right that a plain `model.train()` does not:

    1. **Only the BN modules switch.** `model.train()` would also re-enable the
       Dropout in the classifier head, so the baseline would be measuring that
       noise as well and would look worse than it is.
    2. **The buffers are restored on exit if asked.** In train mode BatchNorm
       overwrites `running_mean` / `running_var` as it goes. For a continuous
       stream that is wanted - the model is supposed to carry its adaptation
       forward. For an episodic protocol, or for measuring an oracle against an
       untouched model, it is contamination: the next condition would start from
       the previous one's statistics and the comparison would silently depend on
       the order conditions were evaluated in.
    3. **The previous mode is restored**, so a caller cannot leak train mode
       into a later evaluation.

    Args:
        model: the network.
        enabled: False makes this a no-op, so the Source baseline can be run
            through exactly the same code path as the adapted ones. Anything
            that differs between the two arms then really is the adaptation.
        keep_running_stats: restore the buffers on exit. False (the default)
            matches the continuous protocol, where the adaptation persists.

    Note that BatchNorm needs more than one sample to estimate a variance, and
    with a small batch the estimate is noisy enough to make this baseline look
    worse than it is. This is the reason the evaluation uses 64 or 128 rather than the
    16 of federated training.
    """
    if not enabled:
        yield model
        return

    modules = bn_modules(model)
    saved = current_bn_state(model) if keep_running_stats else None
    previous_modes = {name: module.training for name, module in modules.items()}

    for module in modules.values():
        module.train()
    try:
        yield model
    finally:
        for name, module in modules.items():
            module.train(previous_modes[name])
        if saved is not None:
            load_bn_state(model, saved)


@torch.no_grad()
def bn_adapt_forward(model: nn.Module, images: torch.Tensor,
                     state: Optional[dict] = None,
                     blind: bool = False,
                     keep_running_stats: bool = False) -> torch.Tensor:
    """One adapted forward pass - the three arms of the comparison, in one call.

        state=None,  blind=False   Source: no adaptation at all
        state=None,  blind=True    blind BN-adapt on the current batch
        state=given, blind=False   load that state, then a plain eval() pass

    Returns the logits. No gradient is computed and no label is used, which is
    the property that makes any of this affordable on a device: one forward,
    nothing else.
    """
    if state is not None and blind:
        raise ValueError(
            "a state and blind adaptation are alternatives: blind BN-adapt "
            "normalises on the batch and would ignore the state entirely")

    if state is not None:
        load_bn_state(model, state)

    was_training = model.training
    model.eval()
    try:
        with adapt_batchnorm(model, enabled=blind,
                             keep_running_stats=keep_running_stats):
            return model(images)
    finally:
        model.train(was_training)
