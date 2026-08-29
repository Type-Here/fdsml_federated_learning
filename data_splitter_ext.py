"""Non-IID partitioning of the source dataset among the clients.

The received `DatasetSplitter` partitions with `StratifiedKFold`, which by
construction gives every client the *same* label distribution - an IID split.
Part A of this project is about what happens when that assumption breaks, so we
need a knob that produces label skew on demand.

Wrapping rather than editing. `DatasetSplitter` already does everything that is
independent of *how* the partition is chosen: scanning the class directories,
building `self.dataframe`, mapping class index back to directory name, and
copying files into `client_i/{train,valid}/<class_dir>/`. Only the choice of
"which sample goes to which client" changes, so this module subclasses and
overrides exactly that. `_copy_images`, `_clear_existing_split` and
`_build_dataframe_from_folders` are inherited untouched.

Two orthogonal choices:

    partition_strategy : 'stratified'  the inherited IID split (default)
                       : 'dirichlet'   label-skew partition, controlled by alpha

    partition_unit     : 'image'       distribute single images
                       : 'track'       distribute whole tracks (default)

Why the unit matters, for GTSRB specifically. Its filenames are
`<track>_<frame>.png`, and every physical road sign was photographed 30 times in
a row: 26640 images are only 888 distinct signs. Distributing single images puts
near-identical frames of one sign in several clients at once, and on both sides
of the internal train/valid split - so the validation score measures "have I
seen this exact sign before", not generalization. With `partition_unit='track'`
the 30 frames of a sign travel together and stay on one side.

The price of tracks is granularity: the rarest GTSRB class has 5 tracks, so with
10 clients at most 5 of them can hold that class at all, in blocks of 30 images.
That is a real coarsening of the Dirichlet draw and it should be stated in the
write-up, not hidden.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from aggregation_policy import label_distribution_discrepancy
from data_splitter import DatasetSplitter
from seeding import DEFAULT_SEED

STRATIFIED = "stratified"
DIRICHLET = "dirichlet"
PARTITION_STRATEGIES = (STRATIFIED, DIRICHLET)

UNIT_IMAGE = "image"
UNIT_TRACK = "track"
PARTITION_UNITS = (UNIT_IMAGE, UNIT_TRACK)

# A client needs at least two units, otherwise its internal train/valid split
# has nothing to put on one of the two sides.
MIN_UNITS_PER_CLIENT = 2

PARTITION_REPORT_NAME = "partition_report.csv"


def track_of(filename: str) -> str:
    """The track a GTSRB frame belongs to, from its filename.

    `00007_00013.png` -> `00007`: frame 13 of track 7. The trailing group of
    digits is the frame counter, everything before it names the physical sign.

    A filename with no `_<digits>` suffix is treated as its own track, so this
    degrades gracefully to per-image behavior on a dataset that has no track
    structure.
    """
    stem = os.path.splitext(filename)[0]
    head, separator, tail = stem.rpartition("_")
    if separator and tail.isdigit():
        return head
    return stem


class PartitionedDatasetSplitter(DatasetSplitter):
    """`DatasetSplitter` with a selectable partition strategy and unit."""

    def __init__(self,
                 output_base_dir: str,
                 source_images_dir: str,
                 num_clients: int,
                 partition_strategy: str = STRATIFIED,
                 dirichlet_alpha: float = 0.5,
                 partition_unit: str = UNIT_TRACK,
                 min_samples_per_client: int = 10,
                 max_resample_attempts: int = 50,
                 max_units_per_class: Optional[int] = None,
                 seed: int = DEFAULT_SEED):
        """
        Args:
            output_base_dir: where the `client_i/` trees are created.
            source_images_dir: dataset root, one subdirectory per class.
            num_clients: how many partitions to produce. Unlike the inherited
                stratified split, the Dirichlet path always produces exactly
                this many, never fewer.
            partition_strategy: 'stratified' or 'dirichlet'.
            dirichlet_alpha: the concentration parameter. Small alpha (0.1) means
                each class is concentrated on few clients - strong skew. Large
                alpha (100) approaches the uniform, near-IID split. It is the
                single dial of the non-IID experiments.
            partition_unit: 'track' or 'image'; see the module docstring.
            min_samples_per_client: reject a draw that leaves any client with
                fewer images than this, and draw again.
            max_resample_attempts: how many draws to try before giving up.
            max_units_per_class: cap the units taken from each class. `None`
                (the default) uses the whole dataset; a small value is the
                smoke-test lever, and must never be set for a real run.
            seed: seeds the Dirichlet draws and the train/valid splits, so the
                same configuration always produces the same partition.
        """
        super().__init__(output_base_dir, source_images_dir, num_clients)

        if partition_strategy not in PARTITION_STRATEGIES:
            raise ValueError(
                f"Unknown partition_strategy '{partition_strategy}'. "
                f"Expected one of {PARTITION_STRATEGIES}."
            )
        if partition_unit not in PARTITION_UNITS:
            raise ValueError(
                f"Unknown partition_unit '{partition_unit}'. "
                f"Expected one of {PARTITION_UNITS}."
            )
        if dirichlet_alpha <= 0:
            raise ValueError(
                f"dirichlet_alpha must be > 0, got {dirichlet_alpha}. "
                f"The Dirichlet distribution is undefined at 0; for a near-IID "
                f"split use a large alpha (e.g. 100) instead."
            )

        self.partition_strategy = partition_strategy
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.partition_unit = partition_unit
        self.min_samples_per_client = int(min_samples_per_client)
        self.max_resample_attempts = int(max_resample_attempts)
        self.max_units_per_class = (None if max_units_per_class is None
                                    else int(max_units_per_class))
        self.seed = int(seed)

        # Filled in by the Dirichlet path: one row per client, one column per
        # class. This is the evidence that the skew is what we asked for.
        self.partition_report: Optional[pd.DataFrame] = None

    @classmethod
    def from_config(cls,
                    config: Dict,
                    output_base_dir: Optional[str] = None,
                    source_images_dir: Optional[str] = None) -> "PartitionedDatasetSplitter":
        """Build a splitter from a grid-search configuration.

        Every key is optional and defaults to the received behaviour, so an old
        configuration that knows nothing about partitioning keeps producing the
        stratified IID split. All of them are recorded in the results CSV for
        free, because `Aggregator.save_results` copies the whole config into
        `run_summary`.
        """
        return cls(
            output_base_dir=output_base_dir or config["splitting_dir"],
            source_images_dir=source_images_dir or config["dataset_path"],
            num_clients=config["num_clients"],
            partition_strategy=config.get("partition_strategy", STRATIFIED),
            dirichlet_alpha=float(config.get("dirichlet_alpha", 0.5)),
            partition_unit=config.get("partition_unit", UNIT_TRACK),
            min_samples_per_client=int(config.get("min_samples_per_client", 10)),
            max_resample_attempts=int(config.get("max_resample_attempts", 50)),
            max_units_per_class=config.get("max_units_per_class"),
            seed=int(config.get("seed", DEFAULT_SEED)),
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def split_dataset(self, validation_split_size: float = 0.1) -> None:
        """Partition the dataset and materialise the per-client directories."""
        if self.partition_strategy == STRATIFIED:
            print("Partition strategy 'stratified': using the inherited IID split.")
            super().split_dataset(validation_split_size)
            return

        print(f"Partition strategy 'dirichlet' (alpha={self.dirichlet_alpha}, "
              f"unit='{self.partition_unit}', seed={self.seed}).")
        self._clear_existing_split()

        unit_members, unit_labels = self._build_units()
        print(f"  {len(self.dataframe)} images grouped into {len(unit_members)} "
              f"{self.partition_unit}(s).")

        unit_members, unit_labels = self._limit_units_per_class(unit_members, unit_labels)

        client_units = self._partition_units_dirichlet(unit_members, unit_labels)
        self._materialise_clients(client_units, unit_members, unit_labels,
                                  validation_split_size)

        self.partition_report = self._build_partition_report(client_units, unit_members)
        self._save_partition_report()

    # ------------------------------------------------------------------
    # Step 1 - decide what the atoms of the partition are
    # ------------------------------------------------------------------
    def _build_units(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """Group the dataframe rows into the atoms that will be distributed.

        Returns:
            unit_members: for each unit, the *positions* of its rows in
                `self.dataframe`. Positions, not index labels: the caller turns
                them into labels with `self.dataframe.index[...]`, mirroring what
                the received `split_dataset` does.
            unit_labels: the class of each unit, aligned with `unit_members`.

        A unit is entirely within one class, which is what makes the per-class
        Dirichlet draw well defined.
        """
        classes = self.dataframe["class"].to_numpy()

        if self.partition_unit == UNIT_IMAGE:
            unit_members = [np.array([position]) for position in range(len(self.dataframe))]
            return unit_members, classes.copy()

        # The track id is only unique *within* a class directory: `00000_*.png`
        # exists under class 00000 and under class 00001 and they are different
        # signs. The grouping key must therefore be (class, track).
        filenames = self.dataframe["filename"].to_numpy()
        groups: Dict[Tuple[int, str], List[int]] = {}
        for position, (filename, class_index) in enumerate(zip(filenames, classes)):
            groups.setdefault((int(class_index), track_of(filename)), []).append(position)

        # Sorted so the unit order does not depend on `os.listdir` order, which
        # is filesystem dependent and would break reproducibility.
        keys = sorted(groups)
        unit_members = [np.array(groups[key]) for key in keys]
        unit_labels = np.array([key[0] for key in keys])
        return unit_members, unit_labels

    def _limit_units_per_class(self,
                               unit_members: Sequence[np.ndarray],
                               unit_labels: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """Keep at most `max_units_per_class` units of each class.

        A smoke-test lever, not an experimental one. The full GTSRB training set
        is 26640 images, and copying all of them into per-client directories
        before a two-round sanity check wastes more time than the check itself.
        With `max_units_per_class=2` and tracks as units the run sees 2 x 30 x 43
        = 2580 images, enough to exercise every code path.

        The subset is drawn with the run's seed, so it is random but
        reproducible; taking the first N units of each class instead would bias
        the sample towards low track ids.

        Returns the inputs unchanged when the option is not set, which is the
        default for every real run.
        """
        if self.max_units_per_class is None:
            return list(unit_members), unit_labels

        rng = np.random.default_rng(self.seed)
        kept: List[int] = []
        for class_index in np.unique(unit_labels):
            units_of_class = np.where(unit_labels == class_index)[0]
            if len(units_of_class) > self.max_units_per_class:
                units_of_class = rng.choice(
                    units_of_class, size=self.max_units_per_class, replace=False
                )
            kept.extend(int(unit) for unit in units_of_class)

        kept.sort()
        limited_members = [unit_members[unit] for unit in kept]
        limited_labels = unit_labels[kept]
        kept_images = sum(len(members) for members in limited_members)
        print(f"  Subsampled to at most {self.max_units_per_class} "
              f"{self.partition_unit}(s) per class: {len(kept)} unit(s), "
              f"{kept_images} images. THIS IS NOT A FULL RUN.")
        return limited_members, limited_labels

    # ------------------------------------------------------------------
    # Step 2 - the Dirichlet draw
    # ------------------------------------------------------------------
    def _partition_units_dirichlet(self,
                                   unit_members: Sequence[np.ndarray],
                                   unit_labels: np.ndarray) -> List[np.ndarray]:
        """Assign every unit to exactly one client, with label skew.

        The algorithm, per class c:

            p ~ Dirichlet(alpha * ones(K))      p in R^K, sums to 1
            shuffle the units of class c
            cut them into K consecutive chunks whose sizes follow p

        Symbols:
            K     : number of clients.
            alpha : concentration. alpha -> 0 makes p almost one-hot, so class c
                    lands on a single client; alpha -> infinity makes p uniform,
                    so class c is shared evenly.
            p[k]  : the fraction of class c that goes to client k.

        What it says: the skew is *per class*, drawn independently for each
        class. That is the standard label-skew partition, and it is why the
        result is non-IID in the labels while the images themselves are
        untouched.

        Sample conservation is exact by construction: the units of a class are a
        permutation cut into consecutive chunks, so every unit lands in exactly
        one chunk. The explicit check below guards against a future edit
        breaking that.

        Rejection sampling: a small alpha can leave a client with nothing at all,
        which would crash the train/valid split and hand `ModelManager` an empty
        dataset. Draws that do are rejected and repeated. That biases the
        distribution slightly away from the pure Dirichlet - the standard
        trade-off in every FL implementation of this partition, worth one line
        in the write-up.
        """
        rng = np.random.default_rng(self.seed)
        unit_sizes = np.array([len(members) for members in unit_members])
        classes = np.unique(unit_labels)
        num_units = len(unit_labels)

        for attempt in range(1, self.max_resample_attempts + 1):
            buckets: List[List[int]] = [[] for _ in range(self.num_clients)]

            for class_index in classes:
                units_of_class = np.where(unit_labels == class_index)[0]
                rng.shuffle(units_of_class)

                proportions = rng.dirichlet(np.full(self.num_clients, self.dirichlet_alpha))
                # cumsum gives the running fraction; scaling by the number of
                # units turns it into cut positions. The last cut is dropped:
                # np.split makes the final chunk out of whatever is left, which
                # is what guarantees nothing is lost to rounding.
                cut_points = (np.cumsum(proportions)[:-1] * len(units_of_class)).astype(int)

                for client_index, chunk in enumerate(np.split(units_of_class, cut_points)):
                    buckets[client_index].extend(int(unit) for unit in chunk)

            units_per_client = np.array([len(bucket) for bucket in buckets])
            images_per_client = np.array([
                int(unit_sizes[bucket].sum()) if bucket else 0 for bucket in buckets
            ])

            if (units_per_client.min() >= MIN_UNITS_PER_CLIENT
                    and images_per_client.min() >= self.min_samples_per_client):
                assigned = int(units_per_client.sum())
                if assigned != num_units:
                    raise RuntimeError(
                        f"Partition lost or duplicated units: assigned {assigned} "
                        f"of {num_units}. This is a bug in _partition_units_dirichlet."
                    )
                print(f"  Dirichlet draw accepted at attempt {attempt}. "
                      f"Images per client: min {images_per_client.min()}, "
                      f"max {images_per_client.max()}.")
                return [np.array(sorted(bucket)) for bucket in buckets]

        raise ValueError(
            f"Could not find a Dirichlet partition leaving every client at least "
            f"{MIN_UNITS_PER_CLIENT} {self.partition_unit}(s) and "
            f"{self.min_samples_per_client} images, in {self.max_resample_attempts} "
            f"attempts. alpha={self.dirichlet_alpha} is too small for "
            f"num_clients={self.num_clients} at this granularity "
            f"({len(unit_labels)} {self.partition_unit}s available). Raise alpha, "
            f"lower num_clients, or switch partition_unit to '{UNIT_IMAGE}'."
        )

    # ------------------------------------------------------------------
    # Step 3 - write the client directories
    # ------------------------------------------------------------------
    def _materialise_clients(self,
                             client_units: Sequence[np.ndarray],
                             unit_members: Sequence[np.ndarray],
                             unit_labels: np.ndarray,
                             validation_split_size: float) -> None:
        """Split each client's share into train/valid and copy the files.

        This is the part that mirrors the tail of the received `split_dataset`,
        with one difference that is the whole point of `partition_unit='track'`:
        the train/valid split happens on *units*, not on images. Splitting
        images here would put frames 1..27 of a sign in train and frames 28..30
        in valid, and the validation score would be near-meaningless.
        """
        for client_index, units in enumerate(client_units):
            client_dir = os.path.join(self.output_base_dir, f"client_{client_index}")
            os.makedirs(client_dir, exist_ok=True)
            print(f"Processing data for client_{client_index}...")

            train_units, valid_units = self._split_units_train_valid(
                units, unit_labels, validation_split_size
            )
            train_positions = self._positions_of(train_units, unit_members)
            valid_positions = self._positions_of(valid_units, unit_members)

            self._copy_images(self.dataframe.index[train_positions], client_dir, "train")
            self._copy_images(self.dataframe.index[valid_positions], client_dir, "valid")

            print(f"  - {len(units)} {self.partition_unit}(s) -> "
                  f"train {len(train_positions)} / valid {len(valid_positions)} images.")

    def _split_units_train_valid(self,
                                 units: np.ndarray,
                                 unit_labels: np.ndarray,
                                 validation_split_size: float) -> Tuple[np.ndarray, np.ndarray]:
        """Hold out a validation share of one client's units.

        Stratified by class when possible, so validation covers the classes the
        client actually holds. Stratification has **two** requirements, and
        missing the second one is what made this function dangerous:

        1. at least 2 units per class, so each class can appear on both sides;
        2. at least as many validation units as there are classes, because a
           stratified split must place one unit of every class in the
           validation share and cannot do so with fewer slots than classes.

        The second is the perverse one. A client holding all 43 classes over 249
        tracks asks for `0.1 * 249 = 25` validation tracks, which is fewer than
        43, so `train_test_split` raises - while a *more* skewed client holding
        14 classes never even attempts to stratify and is fine. The better a
        client's class coverage, the more likely the failure.

        What that cost before this guard existed: the sole fallback was
        `units[:-1], units[-1:]`, a validation set of **one** track - 30 frames
        of one physical sign, one class. The client then reports an accuracy on
        a one-class problem, the server folds it into the round's weighted mean
        (`aggregator.py:99`), and the run's metrics are quietly part nonsense.
        Nothing raises, and the CSV looks ordinary.

        So the retry is unstratified rather than degenerate: a random 10% still
        covers most classes at these sizes, and it keeps the validation set the
        size it was asked to be. The one-unit split survives only as the last
        resort it was meant to be, for a client too small to divide at all.
        """
        labels = unit_labels[units]
        class_counts = pd.Series(labels).value_counts()
        # Mirrors what train_test_split does with a fractional test_size.
        n_valid_units = int(np.ceil(len(units) * validation_split_size))
        can_stratify = (len(units) >= 10
                        and class_counts.min() >= 2
                        and n_valid_units >= len(class_counts))

        try:
            train_units, valid_units = train_test_split(
                units,
                test_size=validation_split_size,
                random_state=self.seed,
                stratify=labels if can_stratify else None,
            )
        except ValueError as error:
            print(f"  Warning: stratified train/valid split failed ({error}); "
                  f"retrying without stratification.")
            try:
                train_units, valid_units = train_test_split(
                    units,
                    test_size=validation_split_size,
                    random_state=self.seed,
                )
            except ValueError as second_error:
                print(f"  Warning: train/valid split fallback ({second_error}).")
                train_units, valid_units = units[:-1], units[-1:]

        return train_units, valid_units

    @staticmethod
    def _positions_of(units: Sequence[int],
                      unit_members: Sequence[np.ndarray]) -> np.ndarray:
        """Flatten a list of units into the dataframe positions they contain."""
        if len(units) == 0:
            return np.array([], dtype=int)
        return np.concatenate([unit_members[unit] for unit in units])

    # ------------------------------------------------------------------
    # Step 4 - the evidence
    # ------------------------------------------------------------------
    def _build_partition_report(self,
                                client_units: Sequence[np.ndarray],
                                unit_members: Sequence[np.ndarray]) -> pd.DataFrame:
        """The client x class matrix, plus `d_k` for each client.

        `d_k` is the label-distribution discrepancy FedDisco is built on - the
        distance between the client's label distribution and a uniform one.
        Computing it here means the partition report already answers "did the
        skew actually happen?" in a single column, before any training runs.
        """
        classes = self.dataframe["class"].to_numpy()
        num_classes = int(classes.max()) + 1

        rows = []
        for client_index, units in enumerate(client_units):
            positions = self._positions_of(units, unit_members)
            counts = np.bincount(classes[positions], minlength=num_classes)

            row = {
                "client": f"client_{client_index}",
                "n_units": len(units),
                "n_images": int(counts.sum()),
                "n_classes_present": int((counts > 0).sum()),
                "d_k": round(label_distribution_discrepancy(counts), 4),
            }
            row.update({f"class_{c}": int(counts[c]) for c in range(num_classes)})
            rows.append(row)

        return pd.DataFrame(rows)

    def _save_partition_report(self) -> None:
        """Write the report next to the client directories and summarise it."""
        if self.partition_report is None:
            return

        report_path = os.path.join(self.output_base_dir, PARTITION_REPORT_NAME)
        self.partition_report.to_csv(report_path, index=False)

        summary = self.partition_report[["client", "n_units", "n_images",
                                         "n_classes_present", "d_k"]]
        print("--- Partition report ---")
        print(summary.to_string(index=False))
        print(f"  d_k: min {summary['d_k'].min():.4f}, "
              f"mean {summary['d_k'].mean():.4f}, max {summary['d_k'].max():.4f} "
              f"(0 = perfectly balanced client)")
        print(f"  Full client x class matrix written to {report_path}")
        print("------------------------")
