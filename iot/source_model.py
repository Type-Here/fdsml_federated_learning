"""The checkpoint, turned back into a network - and the pass that fixes its
normalization statistics.

Part A ends with a pickle holding the trainable parameters of the global model
and a metadata block describing the architecture they belong to. This module
turns that back into something that classifies, and then performs the one
correction it needs before it can be used as a baseline at all.

    load_checkpoint(path)          -> {'weights', 'metadata'}
    build_model(metadata)          -> (model, ModelManager) on the right device
    recalibrate(model, loader)     -> BatchNorm statistics from GTSRB, not ImageNet
    save_recalibrated(...)         -> a second checkpoint that says so

All four in one command, which is the way to run it once and keep the result:

    python -m iot.source_model --checkpoint checkpoints/<run>.pkl \
                               --data dataset/gtsrb/train

**Why the model is rebuilt through `ModelManager` and not by hand.** `set_weights`
copies positionally into whatever `_get_trainable_parameters()` returns, so a
model assembled with a different `num_custom_layers` either raises on a shape
mismatch or - the bad case - fits and is silently wrong. Going through the same
class the training used makes that impossible, and hands over
`_get_transforms()` as well, so the images reach the network through exactly the
resize and normalisation they were trained under. Two different interpolations
between the two halves of the project would show up as a difference nobody could
attribute.

---

**The recalibration, and why it is not optional.**

BatchNorm's `running_mean` and `running_var` are buffers, not parameters.
`get_weights` returns `[p for p in model.parameters() if p.requires_grad]`, so no
federated round ever saw them, and a model rebuilt from a checkpoint carries the
statistics a freshly constructed backbone was shipped with - **ImageNet's**. The
checkpoint says so itself, in `metadata['bn_stats_source']`.

Left there, the "Source" baseline of the inference half is already mismatched on
*clean* images, and any recovery measured afterwards mixes two effects:

    "we fixed ImageNet -> GTSRB"      not the project's claim
    "we fixed clean -> corrupted"     the project's claim

Only the second is being argued for, so the first has to be done once, up front,
for every arm equally.

**One forward pass over the clean training data, no labels, no gradient.** The
federated version of this would be an extra round - each client runs the pass on
its own data, the server merges with `w_k = n_k / N`:

    mu     = sum_k w_k mu_k
    sigma2 = sum_k w_k (sigma2_k + mu_k^2) - mu^2

that is, total variance = mean of the within-client variances **plus** the
variance of the client means. Averaging the `sigma2_k` alone underestimates,
because it ignores how much the clients differ from each other, and under a
Dirichlet partition that term is large - the BatchNorm layers would come out too
narrow and saturate.

Doing the pass offline on pooled data is not an approximation of that. The
batches are then drawn homogeneously, so the between-client term **vanishes by
construction**: the pooled pass computes directly the number the federated
version reassembles from pieces. The operation is linear and would even survive
Paillier; the checkpoint run is plaintext anyway, so there is nothing to gain by
federating it.
"""

import argparse
import json
import os
import pickle
import time
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from class_mapping import assert_canonical_labels
from iot.bn_bank import bn_modules, collect_bn_state, load_bn_state
from iot.routing import BNState

__all__ = [
    'load_checkpoint', 'build_model', 'image_folder_loader',
    'recalibrate', 'save_recalibrated', 'bn_stats_from_checkpoint', 'main',
]

# Where a recalibrated checkpoint says its statistics came from. The string is
# part of the file's contract: `'imagenet'` means "nobody has fixed this yet".
GTSRB_BN_SOURCE = 'gtsrb-train-pooled'


def load_checkpoint(path: str) -> Dict:
    """Read a checkpoint written by the aggregator.

    The file is `{'weights': [ndarray, ...], 'metadata': {...}}`. The weights
    are already on the scale a model is used on - the server aggregates by
    summation and the aggregator divides by the denominator of the round the
    best model came from before writing - so nothing here rescales anything.
    """
    with open(path, 'rb') as handle:
        checkpoint = pickle.load(handle)
    for key in ('weights', 'metadata'):
        if key not in checkpoint:
            raise ValueError(
                f"{path} is not a checkpoint from this project: no '{key}' key")
    return checkpoint


def build_model(metadata: Dict, weights: Optional[Sequence[np.ndarray]] = None,
                device: Optional[str] = None, dataset_path: str = ''):
    """Rebuild the network the checkpoint came from, and load its weights.

    `dataset_path` is unused for construction - `ModelManager` only reads it when
    it builds a DataLoader - but it is on the signature because the class asks
    for it.

    Returns:
        `(model, manager)`. The manager carries `transform_pipeline` and
        `device`, which the evaluation needs and must not reinvent.
    """
    # Imported here rather than at module scope so that a machine without torch
    # can still import the rest of `iot`; `model_manager` pulls in torchvision.
    from model_manager import ModelManager

    config = {
        'model_name': metadata.get('model_name', 'ResNet18'),
        'num_custom_layers': metadata.get('num_custom_layers', 2),
        'num_classes': metadata.get('num_classes', 43),
        'image_size': metadata.get('image_size', 128),
        'device': device or ('cuda' if torch.cuda.is_available() else 'cpu'),
        'convnet_hidden1': metadata.get('convnet_hidden1', 64),
        'convnet_hidden2': metadata.get('convnet_hidden2', 32),
    }
    manager = ModelManager(config, dataset_path)

    if weights is not None:
        expected = [tuple(shape) for shape in metadata.get('weights_shapes', [])]
        actual = [tuple(np.asarray(w).shape) for w in weights]
        if expected and expected != actual:
            raise ValueError(
                f"the weights do not match the metadata: {actual[:3]}... against "
                f"{expected[:3]}... - this file was written by a different model")
        manager.set_weights([np.asarray(w) for w in weights])

    try:
        bn_modules(manager.model)
    except ValueError as error:
        raise ValueError(
            f"{config['model_name']} has no BatchNorm layer, so there is nothing "
            f"for the inference-time adaptation to work on - the checkpoint has "
            f"to come from a ResNet run") from error

    manager.model.eval()
    return manager.model, manager


def image_folder_loader(root: str, transform, batch_size: int = 128,
                        shuffle: bool = False, num_workers: int = 2,
                        seed: Optional[int] = None) -> DataLoader:
    """A DataLoader over one `ImageFolder` root - a condition, or the train set.

    Every condition of GTSRB-C is written as its own `ImageFolder` tree exactly
    so that this is all the loading code the inference half needs.

    `shuffle` is False by default: a stream is evaluated in a fixed order so two
    runs produce the same table, and the routing decision on batch *i* depends on
    batch *i-1*, which makes the order part of the experiment rather than an
    implementation detail.
    """
    dataset = ImageFolder(root=root, transform=transform)
    # ImageFolder numbers the class directories it finds, from 0, so a tree
    # missing one shifts every label after it and the accuracy below it becomes
    # meaningless without raising. Every condition is written with all 43 by
    # construction; this is what keeps that true.
    assert_canonical_labels(dataset, root)
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, generator=generator)


def bn_stats_from_checkpoint(metadata: Dict) -> Optional[BNState]:
    """The recalibrated statistics a checkpoint carries, or None.

    None means the model still has ImageNet's - `metadata['bn_stats_source']`
    says which, and it is the field to read before calling any result a property
    of "the federated model".
    """
    stats = metadata.get('bn_stats')
    if stats is None:
        return None
    return {name: (np.asarray(value['mean']), np.asarray(value['var']))
            for name, value in stats.items()}


@torch.no_grad()
def recalibrate(model, loader, device: torch.device,
                max_batches: Optional[int] = None,
                apply: bool = True) -> BNState:
    """Replace the model's ImageNet statistics with GTSRB's, from clean data.

    One forward pass, `eval()` throughout, over **every** BatchNorm layer - not
    the descriptor subset: this is the state the whole network runs on, and
    leaving the deep stages on ImageNet's numbers would be a stranger model than
    either alternative.

    The accumulation is `bn_bank.collect_bn_state`'s: raw moments per channel,
    combined once at the end. Averaging per-batch variances instead would
    underestimate by exactly the between-batch term - the same identity the
    federated version needs, avoided here by never forming a per-batch variance.

    Args:
        model: the rebuilt network, on `device`.
        loader: clean GTSRB training images. Labels are ignored; there are none
            to use.
        device: where to run.
        max_batches: stop early. For a smoke check only - a partial pass gives a
            partial state and the whole point is that it describes the data.
        apply: write the result into the model's buffers as well as returning it.

    Returns:
        The full `{layer: (mean, var)}` state.
    """
    state = collect_bn_state(model, loader, device, names=None,
                             max_batches=max_batches)
    if apply:
        load_bn_state(model, state)
    return state


def save_recalibrated(checkpoint: Dict, state: BNState, path: str,
                      source: str = GTSRB_BN_SOURCE,
                      recalibrated_on: Optional[str] = None,
                      num_images: Optional[int] = None) -> str:
    """Write a second checkpoint carrying the statistics, and saying so.

    A separate file rather than an edit in place: the original is the artefact
    Part A produced and the two are worth being able to compare, since "how much
    did the recalibration alone buy on clean images" is one of the numbers the
    write-up needs.

    The JSON twin gets a summary in the `bn_stats` slot instead of 9600 floats,
    so a directory of checkpoints stays readable; the arrays live in the pickle.
    """
    metadata = dict(checkpoint['metadata'])
    metadata['bn_stats'] = {name: {'mean': np.asarray(mean).tolist(),
                                   'var': np.asarray(var).tolist()}
                            for name, (mean, var) in state.items()}
    metadata['bn_stats_source'] = source
    metadata['bn_stats_recalibrated_on'] = recalibrated_on
    metadata['bn_stats_num_images'] = num_images
    metadata['bn_stats_num_layers'] = len(state)
    metadata['bn_stats_num_channels'] = int(
        sum(np.asarray(mean).size for mean, _ in state.values()))
    metadata['bn_stats_written'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'wb') as handle:
        pickle.dump({'weights': checkpoint['weights'], 'metadata': metadata},
                    handle, protocol=pickle.HIGHEST_PROTOCOL)

    readable = dict(metadata)
    readable['bn_stats'] = (
        f"{metadata['bn_stats_num_channels']} channels over "
        f"{metadata['bn_stats_num_layers']} layers, in the .pkl beside this file")
    json_path = os.path.splitext(path)[0] + '.json'
    with open(json_path, 'w') as handle:
        json.dump(readable, handle, indent=2, default=str)
    return path


def main(argv=None) -> int:
    """Recalibrate one checkpoint and write the corrected copy beside it.

        python -m iot.source_model --checkpoint checkpoints/<run>.pkl \
                                   --data dataset/gtsrb/train

    Worth doing once and keeping, rather than letting `stream_eval` redo the
    pass in memory every run: the recalibrated file is what makes "how much did
    the recalibration alone buy on clean images" answerable, and it is the same
    Source model for every later experiment instead of one recomputed per run.
    """
    parser = argparse.ArgumentParser(
        description="Replace a checkpoint's ImageNet BatchNorm statistics with "
                    "GTSRB's, from one label-free pass over clean training data.")
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='dataset/gtsrb/train',
                        help="clean training images - the same split Part A "
                             "trained on. Not the corrupted set: this pass "
                             "moves the statistics from ImageNet to GTSRB, it "
                             "is not the bank")
    parser.add_argument('--out', default=None,
                        help="output path; defaults to <checkpoint>_bn.pkl")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--device', default=None)
    args = parser.parse_args(argv)

    checkpoint = load_checkpoint(args.checkpoint)
    already = checkpoint['metadata'].get('bn_stats_source', 'imagenet')
    if already != 'imagenet':
        print(f"note: this checkpoint already carries statistics from "
              f"'{already}'; recalibrating again from {args.data}")

    model, manager = build_model(checkpoint['metadata'], checkpoint['weights'],
                                 device=args.device)
    loader = image_folder_loader(args.data, manager.transform_pipeline,
                                 args.batch_size, num_workers=args.num_workers)
    num_images = len(loader.dataset)

    started = time.time()
    state = recalibrate(model, loader, manager.device)
    elapsed = time.time() - started

    out = args.out or (os.path.splitext(args.checkpoint)[0] + '_bn.pkl')
    path = save_recalibrated(checkpoint, state, out,
                             recalibrated_on=args.data, num_images=num_images)

    channels = sum(np.asarray(mean).size for mean, _ in state.values())
    print(f"{num_images} images, {len(state)} BatchNorm layers, {channels} channels, "
          f"{elapsed:.1f} s")
    print(f"-> {path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
