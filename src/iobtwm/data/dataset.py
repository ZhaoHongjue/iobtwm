"""Map a ``WindowKey`` to one training example."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.utils.data import Dataset

from iobtwm.data.config import Modality
from iobtwm.data.manifest import ClipTable
from iobtwm.data.sampling import WindowKey
from iobtwm.data.segments import SegmentIndex
from iobtwm.data.spectral import SpectralTransform
from iobtwm.data.store import SignalStore

Example = dict[str, Tensor]
"""One example: a tensor per modality plus ``labels`` (multi-hot over the classes of every
clip the window touches), ``distance`` (nearest vehicle distance in metres over those clips, NaN
if unknown), ``dataset`` (index into ``ClipTable.datasets``), ``segment`` and ``start_seconds``."""


class WindowDataset(Dataset[Example]):
    """Cut windows out of the continuous segment waveforms."""

    def __init__(
        self,
        store: SignalStore,
        table: ClipTable,
        segments: SegmentIndex,
        tick_rate: int,
        window_ticks: int,
        *,
        remove_dc: bool,
        transform: SpectralTransform | None,
    ) -> None:
        """Create a dataset reading from ``store``.

        Args:
            store: Waveforms of all segments.
            table: Clip metadata, for labels and distances.
            segments: The segments ``store`` was built from.
            tick_rate: Rate of the start ticks in ``WindowKey``.
            window_ticks: Example length in ticks.
            remove_dc: Subtract each window's mean per modality.
            transform: Applied to every example in the worker, or None to return waveforms.
        """
        self.store = store
        self.tick_rate = tick_rate
        self.remove_dc = remove_dc
        self.transform = transform
        self.clip_ticks = round(table.clip_seconds * tick_rate)
        self.window_ticks = window_ticks
        self.clip_offsets = segments.offsets
        self.clip_labels = table.labels[segments.clips]
        self.clip_distance = table.distance[segments.clips]
        self.segment_dataset = table.dataset[segments.first_clips()].astype(np.int64)

    def _wave(self, modality: Modality, segment: int, start: int) -> Tensor:
        per_tick = self.store.rates[modality] // self.tick_rate
        first = int(self.store.offsets[modality][segment]) + start * per_tick
        wave = np.array(self.store.signals[modality][first : first + self.window_ticks * per_tick])
        if self.remove_dc:
            wave -= wave.mean()
        return torch.from_numpy(wave)

    def _targets(self, segment: int, start: int) -> tuple[npt.NDArray[np.bool_], float]:
        base = int(self.clip_offsets[segment])
        first = base + start // self.clip_ticks
        last = base + math.ceil((start + self.window_ticks) / self.clip_ticks)
        labels = self.clip_labels[first:last].any(axis=0)
        distances = self.clip_distance[first:last]
        known = distances[~np.isnan(distances)]
        return labels, float(known.min()) if len(known) else math.nan

    def __getitem__(self, key: WindowKey) -> Example:
        segment, start = key
        example: Example = {m: self._wave(m, segment, start) for m in self.store.rates}
        if self.transform is not None:
            with torch.no_grad():
                example = self.transform(example)
        labels, distance = self._targets(segment, start)
        example["labels"] = torch.from_numpy(labels.astype(np.float32))
        example["distance"] = torch.tensor(distance, dtype=torch.float32)
        example["dataset"] = torch.tensor(self.segment_dataset[segment])
        example["segment"] = torch.tensor(segment)
        example["start_seconds"] = torch.tensor(start / self.tick_rate, dtype=torch.float32)
        return example
