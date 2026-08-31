"""Client-side extension of `ModelManager`: the FIPA gradient collection pass.

Needs Torch. The maths half - the SVD, the truncation, the
explained variance - lives in `fipa.py`, which imports no torch and is therefore
unit tested on the development machine. What is left here is the part that
genuinely needs a network and a dataset: producing the matrix `G` of collected
gradients.

Why a subclass instead of editing `model_manager.py`. `train` and
`_run_training_epoch` are received code, and the collection pass is additive:
it needs no state from training and changes nothing about it. A subclass keeps
the diff against the lab's code at exactly one line - the instantiation in
`federated_client.py`.
"""

import logging
import os
import re
import zlib
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import fipa
from model_manager import ModelManager


def shuffle_seed(config: dict, dataset_path: str) -> int:
    """A seed that belongs to one client and to nobody else.

    Derived from the run's `seed` and from the client's own data directory name
    (`client_0`, `client_1`, ...), so it is stable across processes and across
    restarts - unlike Python's `hash`, which is randomised per process unless
    PYTHONHASHSEED is pinned before the interpreter starts.

    Args:
        config: the run configuration; only `seed` is read.
        dataset_path: this client's data directory.

    Returns:
        A non-negative int below 2^31, which is what `torch.Generator` wants.
    """
    base = int(config.get('seed', 42))
    name = os.path.basename(os.path.normpath(dataset_path))
    trailing_number = re.search(r'(\d+)$', name)
    offset = (int(trailing_number.group(1)) if trailing_number
              else zlib.crc32(name.encode('utf-8')))
    return (base + offset) % (2 ** 31)


class ExtendedModelManager(ModelManager):
    """`ModelManager` plus what FIPA needs from the client.

    Behaves exactly like the base class unless `collect_gradient_factors` is
    called, so it is safe to use for every algorithm - FedAvg runs will simply
    never call it.

    It also gives the training loader a shuffle generator of its own; see
    `_get_dataloader`.
    """

    def __init__(self, config: dict, dataset_path: str):
        # Set before super().__init__ so that a base class which ever builds a
        # loader while constructing still finds it.
        self._shuffle_generator = torch.Generator()
        self._shuffle_generator.manual_seed(shuffle_seed(config, dataset_path))
        super().__init__(config, dataset_path)

    def _get_dataloader(self, split: str, batch_size: int) -> DataLoader:
        """The base loader, with the training shuffle driven by a private RNG.

        The received version passes no `generator`, so `shuffle=True` draws from
        the **global** Torch RNG. That is reproducible in a single-threaded
        program and not here: the clients of a run are simulated as concurrent
        threads in one process, so seeding fixes the sequence of random numbers
        but not how several threads divide it between them. Two runs of the same
        configuration then see different batch orders - which is how two runs
        that should have been identical came out with three of 330 validation
        images classified differently.

        One generator per instance, not one per call: a fresh generator reseeded
        on every call would hand every round the *same* batch order, which is a
        different bug. Created once, it carries on where the previous round left
        it, deterministically, and no other client can disturb it.

        This does not make a run bit-reproducible on its own. The classifier
        head is still initialised from the global RNG inside the base
        constructor, in those same concurrent threads, and with a frozen
        pre-trained backbone that head is the only randomly initialised thing in
        the model - so it remains the larger source. Removing it was a
        deliberate decision to leave the received training scheme alone.
        """
        data_path = os.path.join(self.dataset_path, split)
        if not os.path.isdir(data_path):
            raise FileNotFoundError(f"Dataset directory not found for split '{split}': {data_path}")
        dataset = ImageFolder(root=data_path, transform=self.transform_pipeline)
        if split != 'train':
            return DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          generator=self._shuffle_generator)

    def collect_gradient_factors(self, batch_size: int, rank: int,
                                 max_batches: Optional[int] = None,
                                 random_state: int = 42,
                                 logger: Optional[logging.Logger] = None
                                 ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Estimate this client's curvature, and return it in low-rank form.

        What it does, in one line: run over the local training data collecting
        the gradient of the loss for each mini-batch, then keep the `rank`
        directions along which those gradients are largest.

            forward + backward, no step        -> g_i for each mini-batch
            stack them                         -> G, shape (n, p)
            randomized SVD, top-r              -> U_m (p, r), L_m (r)

        `U_m` are the directions of parameter space this client's data constrain
        most; `L_m` says how strongly. The server uses them to decide, direction
        by direction, how much of this client's update to trust. See `fipa.py`
        for what happens to them next.

        Three deliberate choices, each of which would be a bug if made the other
        way:

        1. **`model.eval()`, not `model.train()`.** Two reasons. Dropout would
           make the "loss" a different random function on every batch, so the
           collected gradients would measure the dropout mask as much as the
           data. And BatchNorm in train mode updates its running statistics on
           every forward pass: this pass would silently shift the normalization
           statistics the exported global model carries, on data seen in a
           different order than during training.

        2. **No `optimizer.step()`, and gradients cleared afterwards.** This
           pass must observe the model the client is about to send, not move it.
           A single stray step would make the reported `theta_m` inconsistent
           with the curvature reported alongside it.

        3. **The plain task loss**, `self.criterion` - not the FedProx proximal
           term nor the FedLC calibration. The Fisher information is about how
           the *data* constrain the parameters; the proximal term is a pull
           towards the global model and constrains them for a reason that has
           nothing to do with this client's data.

        Cost: one extra forward+backward pass over (a prefix of) the local
        training set, per FIPA round. `max_batches` is the dial - and note the
        honesty caveat, which belongs in the report: PyTorch hands back the mean
        gradient of a mini-batch, not one gradient per sample, so `n` counts
        batches. Larger batches average more, the collected gradients differ
        less from one another, and the estimated spectrum flattens. Smaller
        batches give a better-conditioned estimate at the same cost per sample.

        Args:
            batch_size: batch size for the collection loader. Sensible to keep
                it equal to the training one, so `n` is predictable.
            rank: `r`, config key `fipa_rank`. Clamped inside `top_r_factors` if
                larger than the number of collected gradients.
            max_batches: stop after this many mini-batches. `None` walks the
                whole training set. This caps both time and the memory `G`
                takes: `n * p * 4` bytes, i.e. 36 MB for 64 batches at
                p = 142379.
            random_state: seed for the randomized SVD, so two runs of the same
                configuration produce the same factors.
            logger: optional; the explained-variance ratio is logged through it,
                because that number is a result, not a diagnostic - it is what
                justifies `fipa_rank` in the write-up.

        Returns:
            `(directions, curvature, explained_variance)`:
              - `directions`: `U_m`, shape (p, r), float32, orthonormal columns;
              - `curvature`: `L_m`, shape (r,), float32, non-negative decreasing;
              - `explained_variance`: the fraction of the gradients' variance the
                r kept directions account for, in [0, 1].

            float32 on purpose: these travel over the socket, and `U_m` is the
            bulk of the FIPA payload (2.85 MB at p = 142379, r = 5, against
            5.7 MB in float64). The server casts back to float64 before the QR,
            where precision actually matters; the round trip costs about 5e-8 of
            relative error, measured in `tests/test_fipa.py`.

        Raises:
            RuntimeError: if the training set yields no batch at all. Returning
                empty factors would let the round proceed with a client that
                silently contributes nothing.
        """
        parameters = self._get_trainable_parameters()
        loader = self._get_dataloader('train', batch_size)

        # eval(): no dropout, and no BatchNorm buffer drift. See point 1 above.
        was_training = self.model.training
        self.model.eval()

        gradients = []
        try:
            for batch_index, (inputs, labels) in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.model.zero_grad(set_to_none=True)
                loss = self.criterion(self.model(inputs), labels)
                loss.backward()

                # Flattened in `_get_trainable_parameters` order - the same
                # order `get_weights` uses, which is what makes this gradient
                # vector live in the same R^p as the delta the server will
                # project onto it. A parameter with no gradient (it should not
                # happen for a trainable one, but a frozen head layer would do
                # it) contributes zeros rather than breaking the alignment.
                flat = torch.cat([
                    (p.grad.detach().reshape(-1) if p.grad is not None
                     else torch.zeros(p.numel(), device=self.device))
                    for p in parameters
                ])
                gradients.append(flat.cpu().numpy())
        finally:
            # Leave no gradients behind: the next `train()` builds a fresh Adam,
            # but `loss.backward()` accumulates into `.grad` and a leftover
            # would be added to the first real batch of the next round.
            self.model.zero_grad(set_to_none=True)
            if was_training:
                self.model.train()

        if not gradients:
            raise RuntimeError(
                "No gradients collected: the training dataloader yielded no "
                "batch. A client with no data cannot contribute a curvature "
                "estimate."
            )

        G = np.stack(gradients)
        directions, curvature = fipa.top_r_factors(G, rank, random_state=random_state)
        explained = fipa.explained_variance_ratio(G, curvature)

        if logger is not None:
            logger.info(
                "FIPA factors: %d gradients of %d parameters, rank %d, "
                "explained variance %.4f.",
                G.shape[0], G.shape[1], curvature.shape[0], explained,
            )

        return (directions.astype(np.float32), curvature.astype(np.float32),
                float(explained))