"""What the training augmentation is, and what it deliberately is not.

Torch-free: this module decides the *parameters* and refuses the dangerous
ones, so both decisions are unit tested on a machine without a GPU.
`ExtendedModelManager._get_dataloader` turns the spec into a torchvision
pipeline - that half needs torch and is three lines.

Why augment at all
------------------

With a frozen pre-trained backbone the only trainable part is the classifier
head, and the head overfits almost immediately: measured centrally on the whole
training set, `train_loss` falls 1.26 -> 0.53 over five epochs while
`test_loss` rises from its minimum at epoch 1 and accuracy plateaus. Nothing in
the received pipeline counteracts that - `ModelManager._get_transforms` is a
resize, a tensor conversion and a normalisation, with no augmentation at all.

Why only geometry, and why this geometry
----------------------------------------

The inference-time half of this project evaluates the model under corruption,
so any augmentation that resembles a corruption would train on the test
distribution and turn a measured recovery into a circular result. The corruption
catalogue (`iot/gtsrb_c.py`) is noise, blur, weather and digital artefacts, plus
four unseen ones. So:

    rotation, translation, scale   -> kept: nothing in the catalogue is a rigid
                                      geometric transform, and all three are
                                      physically real (camera tilt, a sign seen
                                      off-centre, a sign seen from further away)
    brightness, contrast, blur,    -> refused: `brightness` and `contrast` are
    noise, sharpness                  literally two of the evaluated conditions

One partial overlap remains and must be stated rather than hidden:
`elastic_transform`, one of the four unseen corruptions, is a geometric warp.
It differs from what is done here - it is a *local* displacement field, while
rotation, translation and scale are *global* rigid transforms - but a model
trained to tolerate the second is partly more tolerant of the first. It is the
one condition whose number has to be read knowing that augmentation was on.

Why never a flip
----------------

Several GTSRB classes are each other's mirror image:

    19 curve left        <-> 20 curve right
    33 turn right ahead  <-> 34 turn left ahead
    36 straight or right <-> 37 straight or left
    38 keep right        <-> 39 keep left

A horizontal flip takes an image of class 33 and produces the *appearance* of
class 34 while keeping the label 33. On eight of the 43 classes it teaches the
model that left and right are the same thing. A vertical flip is simply outside
the domain: no traffic sign is ever seen upside down. So there is no flip
option, and asking for one raises rather than being quietly ignored.
"""

from typing import Dict, NamedTuple, Optional, Tuple

__all__ = [
    'AugmentationSpec',
    'augmentation_spec',
    'REFUSED_KEYS',
    'MIRROR_CLASS_PAIRS',
]

# Present only to be refused, each with the reason it is refused.
REFUSED_KEYS: Dict[str, str] = {
    'augmentation_horizontal_flip':
        "GTSRB has mirror-image class pairs (33/34, 36/37, 38/39, 19/20), so a "
        "horizontal flip relabels one class as another",
    'augmentation_vertical_flip':
        "a traffic sign is never seen upside down",
    'augmentation_brightness':
        "'brightness' is one of the corruptions this project evaluates, so "
        "training on it would make the measured recovery circular",
    'augmentation_contrast':
        "'contrast' is one of the corruptions this project evaluates",
    'augmentation_blur':
        "the corruption catalogue contains three blur families",
    'augmentation_noise':
        "the corruption catalogue contains three noise families",
}

MIRROR_CLASS_PAIRS: Tuple[Tuple[int, int], ...] = (
    (19, 20),   # dangerous curve left / right
    (33, 34),   # turn right / left ahead
    (36, 37),   # go straight or right / left
    (38, 39),   # keep right / left
)


class AugmentationSpec(NamedTuple):
    """The parameters of a single `RandomAffine`, already validated.

    Applied **after** the resize, so the geometry is the same for every image
    whatever its source resolution - GTSRB images range from under 32 pixels to
    over 200 - and so the interpolation happens once, at the working size.

    Attributes:
        degrees: maximum absolute rotation, in degrees. Sampled uniformly in
            [-degrees, +degrees].
        translate: maximum shift as a fraction of the image side, both axes.
        scale: (min, max) zoom factors.
    """
    degrees: float
    translate: float
    scale: Tuple[float, float]


def augmentation_spec(config: Dict) -> Optional[AugmentationSpec]:
    """Read the augmentation parameters out of a run configuration.

    Off unless asked for, so an existing configuration keeps the received
    behaviour exactly - which also means the checkpoint runs already executed
    stay comparable with any run that leaves the flag alone.

    Config keys, all optional:

        train_augmentation              bool,  default False
        augmentation_rotation_degrees   float, default 10.0
        augmentation_translate          float, default 0.1
        augmentation_scale              [min, max], default [0.9, 1.1]

    Returns:
        The spec, or None when augmentation is off.

    Raises:
        ValueError: for a parameter outside its meaningful range, or for any key
            in `REFUSED_KEYS` - see this module's docstring for why refusing is
            better than ignoring.
    """
    for key, reason in REFUSED_KEYS.items():
        if key in config:
            raise ValueError(f"'{key}' is not available: {reason}.")

    if not config.get('train_augmentation', False):
        return None

    degrees = float(config.get('augmentation_rotation_degrees', 10.0))
    translate = float(config.get('augmentation_translate', 0.1))
    scale = tuple(float(value) for value in
                  config.get('augmentation_scale', (0.9, 1.1)))

    if not 0.0 <= degrees < 180.0:
        raise ValueError(
            f"augmentation_rotation_degrees must be in [0, 180), got {degrees}. "
            f"A traffic sign is roughly upright; 10 to 15 degrees covers camera "
            f"tilt without inventing a pose the domain does not contain.")
    if not 0.0 <= translate < 1.0:
        raise ValueError(
            f"augmentation_translate is a fraction of the image side and must "
            f"be in [0, 1), got {translate}.")
    if len(scale) != 2:
        raise ValueError(
            f"augmentation_scale must be a (min, max) pair, got {scale}.")
    if not 0.0 < scale[0] <= scale[1]:
        raise ValueError(
            f"augmentation_scale must satisfy 0 < min <= max, got {scale}.")

    return AugmentationSpec(degrees=degrees, translate=translate,
                            scale=(scale[0], scale[1]))
