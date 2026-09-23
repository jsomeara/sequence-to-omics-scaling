from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
from torch.utils.data import Dataset


# A/C/G/T -> one-hot float32. Unknown bases (e.g. N) remain zeros.
_ONE_HOT = np.zeros((256, 4), dtype=np.float32)
_ONE_HOT[ord("A")] = (1.0, 0.0, 0.0, 0.0)
_ONE_HOT[ord("C")] = (0.0, 1.0, 0.0, 0.0)
_ONE_HOT[ord("G")] = (0.0, 0.0, 1.0, 0.0)
_ONE_HOT[ord("T")] = (0.0, 0.0, 0.0, 1.0)
_ONE_HOT[ord("a")] = (1.0, 0.0, 0.0, 0.0)
_ONE_HOT[ord("c")] = (0.0, 1.0, 0.0, 0.0)
_ONE_HOT[ord("g")] = (0.0, 0.0, 1.0, 0.0)
_ONE_HOT[ord("t")] = (0.0, 0.0, 0.0, 1.0)


class H5EnformerDataset(Dataset):
    """
    High-throughput Enformer dataset for repacked HDF5 targets.

    IMPORTANT:
    This is designed for HDF5 files whose target dataset is chunked like:

        (1, 896, tracks)

    i.e. one complete training example per HDF5 chunk.

    HDF5 and FASTA handles are opened lazily inside each DataLoader worker.
    """

    def __init__(
        self,
        *,
        h5_path: str | Path,
        bed_path: str | Path,
        genome_path: str | Path,
        seqlen: int = 131072,
        shift_aug: bool = False,
        rc_aug: bool = False,
        target_key: str = "targets",
    ) -> None:
        super().__init__()

        self.h5_path = Path(h5_path)
        self.bed_path = Path(bed_path)
        self.genome_path = Path(genome_path)

        self.seqlen = int(seqlen)
        self.shift_aug = bool(shift_aug)
        self.rc_aug = bool(rc_aug)
        self.target_key = target_key

        for path in (
            self.h5_path,
            self.bed_path,
            self.genome_path,
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"Required dataset file does not exist: {path}"
                )

        # Read BED once at construction time, then convert to plain NumPy arrays
        # so __getitem__ does not use pandas at all.
        bed = pd.read_csv(
            self.bed_path,
            sep="\t",
            usecols=["chrom", "start", "end"],
        )

        self.chroms = bed["chrom"].astype(str).to_numpy(copy=True)
        self.starts = bed["start"].to_numpy(
            dtype=np.int64,
            copy=True,
        )
        self.ends = bed["end"].to_numpy(
            dtype=np.int64,
            copy=True,
        )

        with h5py.File(self.h5_path, "r") as h5_file:
            if self.target_key not in h5_file:
                available = ", ".join(h5_file.keys())
                raise KeyError(
                    f"Expected HDF5 dataset '{self.target_key}' "
                    f"in {self.h5_path}; available keys: {available}"
                )

            ds = h5_file[self.target_key]

            self._length = int(ds.shape[0])
            self.target_shape = tuple(
                int(size)
                for size in ds.shape[1:]
            )

            if len(self.target_shape) != 2:
                raise ValueError(
                    f"Expected targets shape "
                    f"(examples, bins, tracks); got "
                    f"{ds.shape} in {self.h5_path}"
                )

        if len(self.chroms) != self._length:
            raise ValueError(
                f"BED rows ({len(self.chroms)}) do not match "
                f"HDF5 targets ({self._length}) for {self.h5_path}"
            )

        # Lazy worker-local handles.
        self._h5_file: h5py.File | None = None
        self._targets_ds: h5py.Dataset | None = None
        self._fasta: Fasta | None = None

    def __len__(self) -> int:
        return self._length

    def _targets(self) -> h5py.Dataset:
        if self._targets_ds is None:
            # Keep this intentionally simple.
            # For the repacked files, one example == one HDF5 chunk,
            # so the default HDF5 cache is sufficient and avoids wasting RAM
            # per worker.
            self._h5_file = h5py.File(
                self.h5_path,
                "r",
            )
            self._targets_ds = self._h5_file[self.target_key]

        return self._targets_ds

    def _genome(self) -> Fasta:
        if self._fasta is None:
            self._fasta = Fasta(
                str(self.genome_path),
                as_raw=True,
                sequence_always_upper=True,
            )

        return self._fasta

    def _sequence_window(
        self,
        chrom: str,
        start: int,
        end: int,
    ) -> str:
        chromosome = self._genome()[chrom]
        chromosome_length = len(chromosome)

        midpoint = (start + end) // 2
        window_start = midpoint - self.seqlen // 2
        window_end = window_start + self.seqlen

        clipped_start = max(
            0,
            window_start,
        )
        clipped_end = min(
            chromosome_length,
            window_end,
        )

        left_padding = max(
            0,
            -window_start,
        )
        right_padding = max(
            0,
            window_end - chromosome_length,
        )

        sequence = str(
            chromosome[
                clipped_start:clipped_end
            ]
        )

        if left_padding or right_padding:
            sequence = (
                "N" * left_padding
                + sequence
                + "N" * right_padding
            )

        return sequence

    @staticmethod
    def _one_hot(
        sequence: str,
    ) -> np.ndarray:
        encoded = np.frombuffer(
            sequence.encode("ascii"),
            dtype=np.uint8,
        )

        return _ONE_HOT[encoded]

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:

        # One optimized HDF5 chunk read.
        #
        # Do NOT use batched/fancy HDF5 indexing here.
        # With one-example-per-chunk files, direct scalar indexing is the
        # cheapest and most predictable access pattern.
        target = np.asarray(
            self._targets()[index],
            dtype=np.float32,
        )

        chrom = self.chroms[index]
        start = int(self.starts[index])
        end = int(self.ends[index])

        if self.shift_aug:
            shift = int(
                np.random.randint(-3, 4)
            )
            start += shift
            end += shift

        sequence = self._sequence_window(
            chrom,
            start,
            end,
        )

        one_hot = self._one_hot(
            sequence
        )

        if (
            self.rc_aug
            and np.random.random() < 0.5
        ):
            one_hot = one_hot[
                ::-1,
                ::-1,
            ].copy()

            target = target[
                ::-1
            ].copy()

        return {
            "x": torch.from_numpy(
                one_hot
            ),
            "labels": torch.from_numpy(
                target
            ),
        }

    def close(self) -> None:
        self._targets_ds = None

        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

        if self._fasta is not None:
            self._fasta.close()
            self._fasta = None

    def __getstate__(self):
        # Ensure live handles are never serialized into DataLoader workers.
        state = self.__dict__.copy()

        state["_h5_file"] = None
        state["_targets_ds"] = None
        state["_fasta"] = None

        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class MultiSpeciesEnformerDataset(Dataset):
    """
    Pair human and mouse examples for joint Enformer training.

    The smaller species is repeated so that each epoch covers every example
    from the larger species.
    """

    def __init__(
        self,
        *,
        human_h5_path: str | Path,
        human_bed_path: str | Path,
        human_genome_path: str | Path,
        mouse_h5_path: str | Path,
        mouse_bed_path: str | Path,
        mouse_genome_path: str | Path,
        seqlen: int = 131072,
        shift_aug: bool = False,
        rc_aug: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.human = H5EnformerDataset(
            h5_path=human_h5_path,
            bed_path=human_bed_path,
            genome_path=human_genome_path,
            seqlen=seqlen,
            shift_aug=shift_aug,
            rc_aug=rc_aug,
        )

        self.mouse = H5EnformerDataset(
            h5_path=mouse_h5_path,
            bed_path=mouse_bed_path,
            genome_path=mouse_genome_path,
            seqlen=seqlen,
            shift_aug=shift_aug,
            rc_aug=rc_aug,
        )

        if (
            len(self.human) == 0
            or len(self.mouse) == 0
        ):
            raise ValueError(
                "Human and mouse datasets must both contain examples."
            )

        if len(self.human) >= len(self.mouse):
            self._larger_species = "human"
            smaller_length = len(self.mouse)
        else:
            self._larger_species = "mouse"
            smaller_length = len(self.human)

        larger_length = max(
            len(self.human),
            len(self.mouse),
        )

        rng = np.random.default_rng(
            seed
        )

        repeats, remainder = divmod(
            larger_length,
            smaller_length,
        )

        repeated_indices = np.tile(
            np.arange(
                smaller_length,
                dtype=np.int64,
            ),
            repeats,
        )

        if remainder:
            remainder_indices = rng.choice(
                smaller_length,
                remainder,
                replace=False,
            ).astype(
                np.int64,
                copy=False,
            )

            repeated_indices = np.concatenate(
                [
                    repeated_indices,
                    remainder_indices,
                ]
            )

        self._smaller_indices = (
            repeated_indices.astype(
                np.int64,
                copy=False,
            )
        )

    def __len__(self) -> int:
        return len(
            self._smaller_indices
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:

        if self._larger_species == "human":
            human_index = index
            mouse_index = int(
                self._smaller_indices[index]
            )
        else:
            mouse_index = index
            human_index = int(
                self._smaller_indices[index]
            )

        human_item = self.human[
            human_index
        ]

        mouse_item = self.mouse[
            mouse_index
        ]

        return {
            "human_x": human_item["x"],
            "human_labels": human_item["labels"],
            "mouse_x": mouse_item["x"],
            "mouse_labels": mouse_item["labels"],
        }

    def close(self) -> None:
        human = getattr(
            self,
            "human",
            None,
        )

        mouse = getattr(
            self,
            "mouse",
            None,
        )

        if human is not None:
            human.close()

        if mouse is not None:
            mouse.close()

    def __getstate__(self):
        return self.__dict__.copy()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
