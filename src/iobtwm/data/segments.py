"""Group clips into time-continuous segments.

A segment is a maximal run of clips from one recording and one node that are consecutive in
time and belong to the same split. Clips are consecutive when their clip numbers differ by one
and, where clip start times are known (M3N-VC), the starts differ by exactly one clip duration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from iobtwm.data.manifest import ClipTable
from iobtwm.data.splits import DROPPED

_START_TOLERANCE_S: Final[float] = 1e-3


@dataclass(frozen=True, slots=True)
class SegmentIndex:
    """Segments as flat arrays.

    Attributes:
        clips: Indices into the ``ClipTable``, segment after segment, in time order.
        offsets: Segment ``i`` owns ``clips[offsets[i]:offsets[i + 1]]``.
        split: Split code of each segment.
    """

    clips: npt.NDArray[np.int64]
    offsets: npt.NDArray[np.int64]
    split: npt.NDArray[np.int8]

    def __len__(self) -> int:
        return len(self.split)

    def num_clips(self) -> npt.NDArray[np.int64]:
        """Return the number of clips of every segment."""
        counts: npt.NDArray[np.int64] = np.diff(self.offsets)
        return counts

    def first_clips(self) -> npt.NDArray[np.int64]:
        """Return the ``ClipTable`` index of the first clip of every segment."""
        firsts: npt.NDArray[np.int64] = self.clips[self.offsets[:-1]]
        return firsts


def build_segments(table: ClipTable, codes: npt.NDArray[np.int8]) -> SegmentIndex:
    """Split the clips that have a split code into time-continuous segments."""
    kept = np.flatnonzero(codes != DROPPED)
    if len(kept) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return SegmentIndex(empty, np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int8))
    order = kept[np.lexsort((table.clip[kept], table.node[kept], table.record[kept]))]
    prev, cur = order[:-1], order[1:]
    start_step = table.clip_start[cur] - table.clip_start[prev]
    start_ok = np.isnan(start_step) | (np.abs(start_step - table.clip_seconds) < _START_TOLERANCE_S)
    continuous = (
        (table.subset[cur] == table.subset[prev])
        & (table.record[cur] == table.record[prev])
        & (table.node[cur] == table.node[prev])
        & (codes[cur] == codes[prev])
        & (table.clip[cur] - table.clip[prev] == 1)
        & start_ok
    )
    starts = np.flatnonzero(np.concatenate([[True], ~continuous]))
    offsets = np.append(starts, len(order)).astype(np.int64)
    return SegmentIndex(
        clips=order.astype(np.int64),
        offsets=offsets,
        split=codes[order[starts]].astype(np.int8),
    )
