"""Decide which windows of which segments form the examples of an epoch.

This is the fifth pipeline stage. ``window_grid`` turns a ``SamplingConfig`` into a
``WindowGrid`` (window length, stride, random offset or not); ``plan_windows`` lays the grid over
every segment; ``WindowSampler`` feeds the result to a DataLoader as ``WindowKey`` tuples, one
epoch at a time.

Positions are measured in ticks: the largest rate that divides every modality's sample rate
(100 Hz for 1600 Hz audio and 100 Hz seismic), so a window start is a whole sample in every
modality.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from torch.utils.data import Sampler

from iobtwm.data.config import SamplingConfig

WindowKey = tuple[int, int]
"""(segment index, start tick) of one example."""


def to_ticks(seconds: float, tick_rate: int, what: str) -> int:
    """Convert a duration to whole ticks.

    Args:
        seconds: The duration.
        tick_rate: Ticks per second.
        what: Name of the setting, for the error message.

    Returns:
        The duration in ticks.

    Raises:
        ValueError: If ``seconds`` is not a positive whole number of ticks.
    """
    ticks = seconds * tick_rate
    if not math.isclose(ticks, round(ticks)) or round(ticks) < 1:
        raise ValueError(f"{what} of {seconds} s is not a whole number of 1/{tick_rate} s ticks")
    return round(ticks)


@dataclass(frozen=True, slots=True)
class WindowGrid:
    """Window length and start spacing, in ticks.

    Attributes:
        window: Example length.
        stride: Distance between consecutive starts within a segment.
        random_offset: Shift every segment's grid by a random offset in ``[0, stride)`` each
            epoch.
    """

    window: int
    stride: int
    random_offset: bool


def window_grid(
    config: SamplingConfig, clip_ticks: int, tick_rate: int, *, train: bool
) -> WindowGrid:
    """Return the window grid of a sampling config for a train or an evaluation split.

    Args:
        config: Sampling settings.
        clip_ticks: Length of one clip in ticks.
        tick_rate: Ticks per second.
        train: Whether the grid is for the train split. Only ``random_crop`` differs between
            train (random offset, stride = window) and evaluation (fixed grid).

    Returns:
        The grid.

    Raises:
        ValueError: If a ``random_crop`` duration is not a whole number of ticks.
    """
    match config.mode:
        case "clip":
            return WindowGrid(window=clip_ticks, stride=clip_ticks, random_offset=False)
        case "multi_clip":
            stride = config.clip_stride if config.clip_stride is not None else config.num_clips
            return WindowGrid(config.num_clips * clip_ticks, stride * clip_ticks, False)
        case "random_crop":
            window = to_ticks(config.window_seconds, tick_rate, "window_seconds")
            if train:
                return WindowGrid(window=window, stride=window, random_offset=True)
            eval_seconds = config.eval_stride_seconds or config.window_seconds
            stride = to_ticks(eval_seconds, tick_rate, "eval_stride_seconds")
            return WindowGrid(window=window, stride=stride, random_offset=False)


def plan_windows(
    lengths: npt.NDArray[np.int64],
    grid: WindowGrid,
    rng: np.random.Generator | None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Lay a window grid over segments and return every window that fits.

    Segments shorter than one window contribute nothing. With ``grid.random_offset`` each
    segment's grid starts at a random tick in ``[0, stride)``, limited so at least one window
    still fits.

    Args:
        lengths: Length of every segment in ticks.
        grid: The grid to lay.
        rng: Source of the random offsets; ignored without ``grid.random_offset``.

    Returns:
        Two arrays of equal length: the position of each window's segment in ``lengths`` and
        the window's start tick within that segment. Windows are grouped by segment, in order.
    """
    offsets = np.zeros(len(lengths), dtype=np.int64)
    fits = lengths >= grid.window
    if grid.random_offset and rng is not None:
        high = np.minimum(grid.stride, lengths - grid.window + 1)
        offsets[fits] = rng.integers(0, high[fits])
    counts = np.where(fits, (lengths - grid.window - offsets) // grid.stride + 1, 0)
    # Vectorised "for each segment, for k in range(count)": repeat the segment index count
    # times, and number the windows of each segment 0, 1, ... by subtracting its first index.
    segment = np.repeat(np.arange(len(lengths), dtype=np.int64), counts)
    first = np.repeat(np.cumsum(counts) - counts, counts)
    step = np.arange(len(segment), dtype=np.int64) - first
    return segment, offsets[segment] + step * grid.stride


class WindowSampler(Sampler[WindowKey]):
    """Yield the ``WindowKey`` of every example of one split, epoch by epoch.

    The sampler runs in the main process, so per-epoch changes reach persistent workers too.
    Call ``set_epoch`` before each epoch: training splits then get a fresh random grid offset
    per segment and a fresh shuffle, evaluation splits stay fixed. The layout is a pure function
    of (seed, epoch), so runs are reproducible.
    """

    def __init__(
        self,
        segments: npt.NDArray[np.int64],
        lengths: npt.NDArray[np.int64],
        grid: WindowGrid,
        *,
        shuffle: bool,
        seed: int,
    ) -> None:
        """Create a sampler and lay out epoch 0.

        Args:
            segments: Store indices of the segments of this split.
            lengths: Length of each of those segments in ticks.
            grid: Window grid of this split.
            shuffle: Shuffle the windows of each epoch (train) or keep them in order (eval).
            seed: Base seed; each epoch uses ``(seed, epoch)``.
        """
        self.segments = segments
        self.lengths = lengths
        self.grid = grid
        self.shuffle = shuffle
        self.seed = seed
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        """Lay out the windows of an epoch.

        Args:
            epoch: Epoch number; the same number always gives the same layout.
        """
        rng = np.random.default_rng([self.seed, epoch])
        position, starts = plan_windows(self.lengths, self.grid, rng)
        if self.shuffle:
            order = rng.permutation(len(position))
            position, starts = position[order], starts[order]
        self._keys = np.stack([self.segments[position], starts], axis=1)

    def __len__(self) -> int:
        """Return the number of windows in the current epoch."""
        return len(self._keys)

    def __iter__(self) -> Iterator[WindowKey]:
        """Yield the windows of the current epoch as (segment index, start tick)."""
        segments: list[int] = self._keys[:, 0].tolist()
        starts: list[int] = self._keys[:, 1].tolist()
        return zip(segments, starts, strict=True)
