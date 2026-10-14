"""Continuous waveforms of every segment, resampled to common rates and kept in memory.

This is the fourth pipeline stage. Each segment is read from ``signals.h5`` as one block of
consecutive clips, joined into one waveform and resampled as a whole, so clip borders carry no
resampling artefacts. All segments of a modality live in one flat float32 array; ``offsets``
locate each segment. ``normalize_store`` then rescales the amplitudes with statistics of the
train split, and ``save_store`` / ``open_store`` cache the result as memory-mapped ``.npy`` files.

Because every segment is a whole number of clips and every clip is a whole number of samples at
the target rates, the flat array of a modality can be reshaped to ``[clips, clip_samples]`` with
rows in ``SegmentIndex.clips`` order. Normalisation relies on that.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
from scipy.signal import resample_poly

from iobtwm.data.config import Modality, Normalization
from iobtwm.data.manifest import ClipTable
from iobtwm.data.segments import SegmentIndex

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SignalStore:
    """Flat per-modality waveforms of all segments.

    Attributes:
        rates: Sample rate of each modality in Hz.
        signals: Concatenated waveforms of all segments, in segment order. May be a read-only
            memory map when opened from a cache.
        offsets: ``len(segments) + 1`` boundaries per modality; segment ``i`` of modality ``m``
            is ``signals[m][offsets[m][i]:offsets[m][i + 1]]``.
    """

    rates: dict[Modality, int]
    signals: dict[Modality, npt.NDArray[np.float32]]
    offsets: dict[Modality, npt.NDArray[np.int64]]

    def segment(self, modality: Modality, index: int) -> npt.NDArray[np.float32]:
        """Return the waveform of one segment.

        Args:
            modality: Which modality.
            index: Segment index.

        Returns:
            A view into ``signals``; copy it before modifying.
        """
        bounds = self.offsets[modality]
        return self.signals[modality][bounds[index] : bounds[index + 1]]


def resample(wave: npt.NDArray[np.float64], source: float, target: int) -> npt.NDArray[np.float32]:
    """Resample a waveform with a polyphase anti-aliasing filter.

    Args:
        wave: One-dimensional waveform.
        source: Its sample rate in Hz; must be a whole number.
        target: The wanted sample rate in Hz.

    Returns:
        The waveform at ``target`` Hz with ``ceil(len(wave) * target / source)`` samples.
        Unchanged (only cast) when the rates are equal.

    Raises:
        ValueError: If ``source`` is not an integer rate.
    """
    if not float(source).is_integer():
        raise ValueError(f"only integer sample rates are supported, got {source}")
    ratio = Fraction(target, int(source))
    if ratio == 1:
        return wave.astype(np.float32)
    out = resample_poly(wave, ratio.numerator, ratio.denominator)
    return np.asarray(out, dtype=np.float32)


def _read_rows(dataset: h5py.Dataset, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """Read rows of a 2-D HDF5 dataset and join them into one waveform.

    One contiguous slice covering all rows is read and the rows are picked from it, which is
    much faster than HDF5 fancy indexing. Rows of one segment are close together (Parkland
    interleaves ``_dwt`` rows, so the slice is about twice as large there).

    Args:
        dataset: An HDF5 dataset of shape ``[rows, samples]``.
        rows: Rows to read, in the order they should be joined.

    Returns:
        The rows concatenated, as float64.
    """
    low, high = int(rows.min()), int(rows.max())
    block = np.asarray(dataset[low : high + 1])
    return block[rows - low].reshape(-1).astype(np.float64)


def _load_modality(
    root: Path,
    table: ClipTable,
    segments: SegmentIndex,
    modality: Modality,
    rate: int,
) -> list[npt.NDArray[np.float32]]:
    """Read and resample one modality of every segment.

    Each HDF5 file is opened once and all its segments are read in turn.

    Args:
        root: The unified root directory.
        table: The clips, for subsets, HDF5 rows and native rates.
        segments: The segments to read.
        modality: Which modality.
        rate: Target sample rate in Hz.

    Returns:
        One waveform per segment, in segment order.
    """
    pieces: list[npt.NDArray[np.float32]] = [np.zeros(0, dtype=np.float32)] * len(segments)
    segment_subset = table.subset[segments.first_clips()]
    for subset_index in np.unique(segment_subset):
        info = table.subsets[int(subset_index)]
        logger.info("loading %s %s", info.name, modality)
        with h5py.File(root / info.name / "signals.h5", "r") as handle:
            dataset = handle[f"signals/{modality}"]
            for s in np.flatnonzero(segment_subset == subset_index):
                clips = segments.clips[segments.offsets[s] : segments.offsets[s + 1]]
                wave = _read_rows(dataset, table.row[clips])
                pieces[s] = resample(wave, info.rates[modality], rate)
    return pieces


def load_store(
    root: Path,
    table: ClipTable,
    segments: SegmentIndex,
    rates: Mapping[Modality, int],
) -> SignalStore:
    """Read, join and resample the waveforms of all segments.

    Args:
        root: The unified root directory.
        table: The clips.
        segments: The segments to load.
        rates: Target sample rate per modality; also selects the modalities to load.

    Returns:
        The store, not yet normalised.

    Raises:
        ValueError: If a clip does not span a whole number of samples at a target rate, or a
            resampled segment does not have exactly ``clips * clip_samples`` samples.
    """
    signals: dict[Modality, npt.NDArray[np.float32]] = {}
    offsets: dict[Modality, npt.NDArray[np.int64]] = {}
    for modality, rate in rates.items():
        clip_samples = table.clip_seconds * rate
        if not math.isclose(clip_samples, round(clip_samples)):
            raise ValueError(f"a {table.clip_seconds} s clip is not whole samples at {rate} Hz")
        pieces = _load_modality(root, table, segments, modality, rate)
        lengths = np.asarray([len(p) for p in pieces], dtype=np.int64)
        expected = segments.num_clips() * round(clip_samples)
        if not np.array_equal(lengths, expected):
            raise ValueError(f"{modality}: resampled segment lengths do not match the clip counts")
        signals[modality] = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        offsets[modality] = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return SignalStore(rates=dict(rates), signals=signals, offsets=offsets)


def _scale_keys(
    table: ClipTable, clips: npt.NDArray[np.int64], normalization: Normalization
) -> npt.NDArray[np.int64]:
    """Return the normalisation group of each clip.

    Args:
        table: The clips.
        clips: ``ClipTable`` rows to look up.
        normalization: ``dataset`` groups by dataset; anything else by (dataset, node).

    Returns:
        One integer group id per entry of ``clips``.
    """
    if normalization == "dataset":
        return table.dataset[clips].astype(np.int64)
    return table.dataset[clips].astype(np.int64) * len(table.nodes) + table.node[clips]


def clip_scales(
    clip_rms: npt.NDArray[np.float64],
    keys: npt.NDArray[np.int64],
    is_train: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Return a scale for every clip: the median train RMS of its group.

    The median is robust to the few loud clips where a vehicle passes close by. Silent clips are
    ignored. Groups without usable train clips get the median of the other groups' scales, or 1
    if there are none.

    Args:
        clip_rms: RMS of every clip (after removing its mean).
        keys: Normalisation group of every clip.
        is_train: Whether every clip is in the train split.

    Returns:
        The scale of every clip.
    """
    usable = is_train & (clip_rms > 0)
    groups, inverse = np.unique(keys, return_inverse=True)
    group_scale = np.full(len(groups), np.nan)
    for g in range(len(groups)):
        selected = usable & (inverse == g)
        if selected.any():
            group_scale[g] = np.median(clip_rms[selected])
    known = group_scale[~np.isnan(group_scale)]
    group_scale[np.isnan(group_scale)] = float(np.median(known)) if len(known) else 1.0
    scales: npt.NDArray[np.float64] = group_scale[inverse]
    return scales


def normalize_store(
    store: SignalStore,
    table: ClipTable,
    segments: SegmentIndex,
    normalization: Normalization,
) -> None:
    """Divide every clip in place by the train-split scale of its group.

    Args:
        store: The store to rescale; its arrays must be writable.
        table: The clips.
        segments: The segments ``store`` was built from.
        normalization: ``node`` for one scale per (dataset, node), ``dataset`` for one per
            dataset, ``none`` to leave the store untouched.
    """
    if normalization == "none":
        return
    keys = _scale_keys(table, segments.clips, normalization)
    is_train = np.repeat(segments.split == 0, segments.num_clips())
    for modality, rate in store.rates.items():
        # A view with one row per clip, in segments.clips order (see the module docstring).
        clips = store.signals[modality].reshape(-1, round(table.clip_seconds * rate))
        rms = clips.std(axis=1, dtype=np.float64)
        scales = clip_scales(rms, keys, is_train)
        clips /= scales[:, None].astype(np.float32)
        logger.info("%s scales: min %.3g, max %.3g", modality, scales.min(), scales.max())


def save_store(store: SignalStore, directory: Path) -> None:
    """Write a store to a directory as ``.npy`` files.

    The files are written to a temporary sibling first and renamed, so a crash never leaves a
    half-written cache behind.

    Args:
        store: The store to save.
        directory: Target directory; must not exist yet.
    """
    tmp = directory.with_name(f"{directory.name}.tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    for modality in store.rates:
        np.save(tmp / f"{modality}.npy", store.signals[modality])
        np.save(tmp / f"{modality}_offsets.npy", store.offsets[modality])
    (tmp / "rates.json").write_text(json.dumps(store.rates))
    tmp.rename(directory)


def open_store(directory: Path) -> SignalStore:
    """Open a store written by ``save_store``.

    The signals are memory-mapped read-only, so opening is instant and forked DataLoader workers
    share the pages instead of copying them.

    Args:
        directory: A directory written by ``save_store``.

    Returns:
        The store.
    """
    rates: dict[Modality, int] = json.loads((directory / "rates.json").read_text())
    signals: dict[Modality, npt.NDArray[np.float32]] = {}
    offsets: dict[Modality, npt.NDArray[np.int64]] = {}
    for modality in rates:
        signals[modality] = np.load(directory / f"{modality}.npy", mmap_mode="r")
        offsets[modality] = np.load(directory / f"{modality}_offsets.npy")
    return SignalStore(rates=rates, signals=signals, offsets=offsets)
