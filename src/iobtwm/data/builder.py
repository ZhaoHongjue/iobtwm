"""Build the train/val/test data loaders from a ``DataConfig``.

``build_data`` runs the whole pipeline (manifest, split, segments, store, samplers, loaders) and
returns an ``IoBTData`` that holds the loaders and everything needed to interpret their batches.

Typical use::

    data = build_data(DataConfig(cache_dir=Path("cache/data")))
    for epoch in range(epochs):
        for batch in data.iterate("train", epoch=epoch, device="cuda"):
            ...  # batch["audio"], batch["seismic"], batch["labels"], ...
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch.utils.data import DataLoader

from iobtwm.data.config import SPLIT_NAMES, DataConfig, Modality, SplitName
from iobtwm.data.dataset import Example, WindowDataset
from iobtwm.data.manifest import ClipTable, load_clip_table
from iobtwm.data.sampling import WindowSampler, window_grid
from iobtwm.data.segments import SegmentIndex, build_segments
from iobtwm.data.spectral import SpectralTransform
from iobtwm.data.splits import assign_splits
from iobtwm.data.store import SignalStore, load_store, normalize_store, open_store, save_store

logger = logging.getLogger(__name__)

# Bump when the cached store layout or its contents change for the same config.
_CACHE_VERSION: Final[int] = 2


@dataclass(frozen=True, slots=True)
class SplitSummary:
    """Size of one split.

    Attributes:
        segments: Number of continuous segments.
        hours: Total duration of the segments (summed over nodes).
        examples: Examples per epoch with the current sampling settings.
    """

    segments: int
    hours: float
    examples: int


@dataclass(frozen=True)
class IoBTData:
    """Loaders and everything needed to interpret their batches.

    Attributes:
        config: The settings the data was built with.
        table: Clip metadata, including the class and dataset vocabularies.
        segments: Continuous segments and their splits.
        store: Waveforms of all segments.
        transform: The STFT. Already applied in the loaders when ``spectral.placement`` is
            ``loader``; applied by ``prepare`` when it is ``device``.
        samplers: Per-split window samplers.
        loaders: Per-split data loaders. Splits without examples are absent.
    """

    config: DataConfig
    table: ClipTable
    segments: SegmentIndex
    store: SignalStore
    transform: SpectralTransform
    samplers: dict[SplitName, WindowSampler]
    loaders: dict[SplitName, DataLoader[Example]]

    @property
    def classes(self) -> tuple[str, ...]:
        """Names of the multi-hot ``labels`` entries."""
        return self.table.classes

    def set_epoch(self, epoch: int) -> None:
        """Lay out the windows of an epoch in every split.

        Args:
            epoch: Epoch number.
        """
        for sampler in self.samplers.values():
            sampler.set_epoch(epoch)

    def prepare(self, batch: Example, device: torch.device | str) -> Example:
        """Make a loader batch ready for the model.

        Args:
            batch: A batch from one of ``loaders``.
            device: Where the model runs.

        Returns:
            The batch on ``device``, with the STFT applied there if ``spectral.placement`` is
            ``device``.
        """
        moved = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if self.config.spectral.placement == "device":
            with torch.no_grad():
                self.transform.to(device)
                return self.transform.forward(moved)
        return moved

    def iterate(
        self, split: SplitName, *, epoch: int = 0, device: torch.device | str = "cpu"
    ) -> Iterator[Example]:
        """Iterate over one epoch of a split.

        Args:
            split: Which split; must be in ``loaders``.
            epoch: Epoch number; sets the window layout of the split first.
            device: Where the batches should end up.

        Yields:
            Batches prepared with ``prepare``.

        Raises:
            KeyError: If ``split`` has no loader (it has no examples).
        """
        self.samplers[split].set_epoch(epoch)
        for batch in self.loaders[split]:
            yield self.prepare(batch, device)

    def summary(self) -> dict[SplitName, SplitSummary]:
        """Return the size of every split, including empty ones."""
        clips = self.segments.num_clips()
        out: dict[SplitName, SplitSummary] = {}
        for code, name in enumerate(SPLIT_NAMES):
            mask = self.segments.split == code
            hours = float(clips[mask].sum()) * self.table.clip_seconds / 3600
            examples = len(self.samplers[name]) if name in self.samplers else 0
            out[name] = SplitSummary(int(mask.sum()), hours, examples)
        return out


def tick_rate(config: DataConfig) -> int:
    """Return the finest time step shared by all modalities.

    Args:
        config: Supplies the modalities and their rates.

    Returns:
        The greatest common divisor of the modality rates, in Hz.
    """
    return math.gcd(*(config.rate(m) for m in config.modalities))


def cache_key(config: DataConfig) -> str:
    """Hash the settings that determine the stored signals.

    Sampling and spectral settings are left out: they only affect how windows are cut from the
    store, so changing them reuses the cache.

    Args:
        config: The data settings.

    Returns:
        A 16-character hex digest.
    """
    fields = {
        "version": _CACHE_VERSION,
        "root": str(config.root.resolve()),
        "subsets": config.subsets,
        "rates": {m: config.rate(m) for m in config.modalities},
        "normalization": config.normalization,
        "split": dataclasses.asdict(config.split),
    }
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]


def _store(config: DataConfig, table: ClipTable, segments: SegmentIndex) -> SignalStore:
    """Load the normalised signals, from the cache when it already holds them.

    Args:
        config: The data settings.
        table: The clips.
        segments: The segments to load.

    Returns:
        The store; memory-mapped whenever ``config.cache_dir`` is set.
    """
    rates: dict[Modality, int] = {m: config.rate(m) for m in config.modalities}
    cached = None if config.cache_dir is None else config.cache_dir / cache_key(config)
    if cached is not None and cached.exists():
        logger.info("opening cached signals %s", cached)
        return open_store(cached)
    store = load_store(config.root, table, segments, rates)
    normalize_store(store, table, segments, config.normalization)
    if cached is not None:
        save_store(store, cached)
        # Reopen as a memory map so this run behaves exactly like later cached runs.
        return open_store(cached)
    return store


def _loader(
    config: DataConfig, dataset: WindowDataset, sampler: WindowSampler
) -> DataLoader[Example]:
    """Wrap a dataset and its sampler in a DataLoader.

    Args:
        config: Supplies batch size and worker count.
        dataset: The split's dataset.
        sampler: The split's sampler.

    Returns:
        The DataLoader.
    """
    # Fork so workers share the in-memory signals instead of receiving a pickled copy.
    context = "fork" if config.num_workers > 0 else None
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        multiprocessing_context=context,
    )


def _split_loader(
    config: DataConfig,
    store: SignalStore,
    table: ClipTable,
    segments: SegmentIndex,
    transform: SpectralTransform | None,
    name: SplitName,
) -> tuple[WindowSampler, DataLoader[Example]] | None:
    """Build the sampler and loader of one split.

    Args:
        config: The data settings.
        store: Waveforms of all segments.
        table: The clips.
        segments: All segments; only those of split ``name`` are used.
        transform: STFT to run in the workers, or None.
        name: Which split.

    Returns:
        The sampler and loader, or None if the split has no examples. A warning is logged when
        the split has segments but none is long enough for one window.
    """
    ticks = tick_rate(config)
    clip_ticks = round(table.clip_seconds * ticks)
    grid = window_grid(config.sampling, clip_ticks, ticks, train=name == "train")
    members = np.flatnonzero(segments.split == SPLIT_NAMES.index(name)).astype(np.int64)
    lengths = segments.num_clips()[members] * clip_ticks
    sampler = WindowSampler(members, lengths, grid, shuffle=name == "train", seed=config.split.seed)
    if len(sampler) == 0:
        if len(members):
            logger.warning("%s: no segment is long enough for one window", name)
        return None
    dataset = WindowDataset(
        store, table, segments, ticks, grid.window, remove_dc=config.remove_dc, transform=transform
    )
    return sampler, _loader(config, dataset, sampler)


def build_data(config: DataConfig) -> IoBTData:
    """Read, split, resample and normalise the data, and build one loader per split.

    Args:
        config: The data settings.

    Returns:
        The loaders with their samplers and metadata.

    Raises:
        ValueError: If a setting is incompatible with the data (e.g. an STFT frame that is not
            whole samples, or subsets with different clip durations).
    """
    # The transform validates the STFT settings, so build it before the slow loading.
    transform = SpectralTransform(config.spectral, {m: config.rate(m) for m in config.modalities})
    table = load_clip_table(Path(config.root), config.subsets)
    segments = build_segments(table, assign_splits(table, config.split))
    store = _store(config, table, segments)
    in_loader = transform if config.spectral.placement == "loader" else None
    samplers: dict[SplitName, WindowSampler] = {}
    loaders: dict[SplitName, DataLoader[Example]] = {}
    for name in SPLIT_NAMES:
        built = _split_loader(config, store, table, segments, in_loader, name)
        if built is not None:
            samplers[name], loaders[name] = built
    return IoBTData(config, table, segments, store, transform, samplers, loaders)
