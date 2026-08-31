"""Compatibility layer for `imagecorruptions` on a modern numpy / scikit-image.

`imagecorruptions` 1.1.2 was written against numpy 1.x and scikit-image 0.19.
Three of its corruptions call names that no longer exist:

    fog             `np.float_`, the alias of float64 that numpy 2.0 removed
                    (corruptions.py:46, inside the plasma-fractal helper)
    glass_blur      `skimage.filters.gaussian(..., multichannel=True)`, renamed
    gaussian_blur   to `channel_axis=` and removed in scikit-image 0.23
                    (corruptions.py:188, 198, 210)

The failures happen at *call* time, not at import, so without this module a
corrupted dataset is generated with holes in it and nothing complains until
much later.

Both problems are pure renames - the arithmetic behind them is unchanged - so
patching restores the original behavior rather than approximating it. That is
worth the twenty lines: the alternative, dropping the three, would cost us
`fog`, and weather is the corruption family this project is actually about.

Which failures you hit depends on the environment, and both are environments we
use:

    numpy 1.26 + scikit-image 0.26   (this repo's pinned local environment)
        `np.float_` still exists, so only the two blurs break
    numpy 2.x  + scikit-image 0.26   (what a fresh `pip install` gives, Colab)
        all three break

The shim covers both and is a no-op wherever nothing is broken.

One thing it deliberately does NOT fix, because it cannot: `imagecorruptions`
imports `pkg_resources`, which setuptools removed in version 81. That one is an
install-time constraint (`pip install "setuptools<81"`), not a runtime patch.

Import this module instead of `imagecorruptions` - patching runs on import, so
there is no way to forget it:

    from iot.corruption_shim import corrupt, corruption_names
"""

from typing import Dict, List

import numpy as np

__all__ = ['corrupt', 'corruption_names', 'patch_report', 'MIN_SIDE_PX']

# `imagecorruptions` refuses anything smaller than this on either side. It is
# not a soft limit we could argue with: several corruptions index into fixed
# 32x32 kernels. GTSRB matters here - about one test image in five is smaller
# than this natively - which is why gtsrb_c.py resizes before it corrupts.
MIN_SIDE_PX = 32

_report: Dict[str, bool] = {}


def _patch_numpy_float_alias() -> bool:
    """Restore `np.float_` on numpy 2.x. Returns True if it was missing."""
    if hasattr(np, 'float_'):
        return False
    np.float_ = np.float64
    return True


def _patch_skimage_gaussian() -> bool:
    """Accept the removed `multichannel=` keyword by translating it.

    Returns True if the translation was installed, False if a previous call
    already did it.
    """
    import skimage.filters as skimage_filters

    original = skimage_filters.gaussian
    if getattr(original, '_translates_multichannel', False):
        return False

    def gaussian(image, *args, **kwargs):
        # `multichannel=True` meant "the last axis is color, do not smooth
        # across it", which is exactly what `channel_axis=-1` means now.
        # `multichannel=False` meant the opposite and is the default, so
        # popping it without setting anything is already correct.
        if kwargs.pop('multichannel', None):
            kwargs.setdefault('channel_axis', -1)
        return original(image, *args, **kwargs)

    gaussian._translates_multichannel = True
    skimage_filters.gaussian = gaussian

    # `imagecorruptions.corruptions` did `from skimage.filters import gaussian`
    # at import time, so it holds its own reference to the original function
    # that rebinding the attribute above does not reach.
    import imagecorruptions.corruptions as _corruptions
    _corruptions.gaussian = gaussian
    return True


def _apply() -> Dict[str, bool]:
    return {
        'numpy_float_alias': _patch_numpy_float_alias(),
        'skimage_gaussian_multichannel': _patch_skimage_gaussian(),
    }


_report = _apply()

# Imported only after the patches are in place. The import itself is harmless
# either way, but keeping the order explicit documents the dependency.
from imagecorruptions import corrupt, get_corruption_names  # noqa: E402


def corruption_names() -> List[str]:
    """Every corruption the installed package exposes (19 at version 1.1.2).

    That is the classic 15 of the ImageNet-C / CIFAR-10-C benchmarks plus four
    extras: speckle_noise, gaussian_blur, spatter, saturate.
    """
    return list(get_corruption_names('all'))


def patch_report() -> Dict[str, bool]:
    """What this environment actually needed patching, for the run manifest."""
    return dict(_report)
