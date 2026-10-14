"""Map a ``WindowKey`` to one training example.

This is the last pipeline stage before batching. ``WindowDataset`` is a map-style dataset whose
keys are ``WindowKey`` tuples produced by a ``WindowSampler``, not integer indices: it cuts the
window out of the in-memory segment waveforms, optionally applies the STFT, and attaches the
labels of the clips the window touches.
"""

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
"""One example (or, after collation, one batch with a leading batch dimension).

Keys:
    audio, seismic: Waveform ``[samples]`` or spectrum ``[channels, freq, frames]`` per
        modality in ``DataConfig.modalities``.
    labels: float32 multi-hot ``[classes]`` over every clip the window touches.
    distance: float32 scalar, nearest vehicle distance in metres over those clips, NaN if
        unknown.
    dataset: int64 index into ``ClipTable.datasets``.
    segment: int64 segment index.
    start_seconds: float32 window start within its segment.
"""


class WindowDataset(Dataset[Example]):
    """Cut windows out of the continuous segment waveforms.

    Index it with a ``WindowKey``; it has no length of its own because the set of windows is
    decided per epoch by the sampler.
    """

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
        """Create a dataset reading from a store.

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
        # Per-clip targets in segment order, so a window's clips are one contiguous slice.
        self.clip_offsets = segments.offsets
        self.clip_labels = table.labels[segments.clips]
        self.clip_distance = table.distance[segments.clips]
        self.segment_dataset = table.dataset[segments.first_clips()].astype(np.int64)

    def _wave(self, modality: Modality, segment: int, start: int) -> Tensor:
        """Copy one window of one modality out of the store.

        Args:
            modality: Which modality.
            segment: Segment index.
            start: Window start in ticks within the segment.

        Returns:
            The window as a float32 tensor, with its mean removed if ``remove_dc`` is set.
        """
        per_tick = self.store.rates[modality] // self.tick_rate
        first = int(self.store.offsets[modality][segment]) + start * per_tick
        # np.array copies, so the store (possibly a read-only memory map) is never modified.
        wave = np.array(self.store.signals[modality][first : first + self.window_ticks * per_tick])
        if self.remove_dc:
            wave -= wave.mean()
        return torch.from_numpy(wave)

    def _targets(self, segment: int, start: int) -> tuple[npt.NDArray[np.bool_], float]:
        """Return the labels and distance of the clips a window touches.

        Args:
            segment: Segment index.
            start: Window start in ticks within the segment.

        Returns:
            The union of the clips' labels and the smallest known distance (NaN if none).
        """
        base = int(self.clip_offsets[segment])
        first = base + start // self.clip_ticks
        last = base + math.ceil((start + self.window_ticks) / self.clip_ticks)
        labels = self.clip_labels[first:last].any(axis=0)
        distances = self.clip_distance[first:last]
        known = distances[~np.isnan(distances)]
        return labels, float(known.min()) if len(known) else math.nan

    def __getitem__(self, key: WindowKey) -> Example:
        """Return the example of one window.

        Args:
            key: (segment index, start tick).

        Returns:
            The example, see ``Example``.
        """
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
