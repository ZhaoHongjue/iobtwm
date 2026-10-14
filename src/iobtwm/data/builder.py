"""Build the train/val/test data loaders from a ``DataConfig``.

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
        """Lay out the windows of ``epoch`` in every split."""
        for sampler in self.samplers.values():
            sampler.set_epoch(epoch)

    def prepare(self, batch: Example, device: torch.device | str) -> Example:
        """Move ``batch`` to ``device`` and apply the STFT there if it runs on the device."""
        moved = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if self.config.spectral.placement == "device":
            with torch.no_grad():
                self.transform.to(device)
                return self.transform.forward(moved)
        return moved

    def iterate(
        self, split: SplitName, *, epoch: int = 0, device: torch.device | str = "cpu"
    ) -> Iterator[Example]:
        """Yield the prepared batches of ``split`` for ``epoch``."""
        self.samplers[split].set_epoch(epoch)
        for batch in self.loaders[split]:
            yield self.prepare(batch, device)

    def summary(self) -> dict[SplitName, SplitSummary]:
        """Return the size of every split."""
        clips = self.segments.num_clips()
        out: dict[SplitName, SplitSummary] = {}
        for code, name in enumerate(SPLIT_NAMES):
            mask = self.segments.split == code
            hours = float(clips[mask].sum()) * self.table.clip_seconds / 3600
            examples = len(self.samplers[name]) if name in self.samplers else 0
            out[name] = SplitSummary(int(mask.sum()), hours, examples)
        return out


def tick_rate(config: DataConfig) -> int:
    """Return the finest time step shared by all modalities, in Hz."""
    return math.gcd(*(config.rate(m) for m in config.modalities))


def cache_key(config: DataConfig) -> str:
    """Hash the settings that determine the stored signals."""
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
    """Load the signals, from the cache when it already holds them."""
    rates: dict[Modality, int] = {m: config.rate(m) for m in config.modalities}
    cached = None if config.cache_dir is None else config.cache_dir / cache_key(config)
    if cached is not None and cached.exists():
        logger.info("opening cached signals %s", cached)
        return open_store(cached)
    store = load_store(config.root, table, segments, rates)
    normalize_store(store, table, segments, config.normalization)
    if cached is not None:
        save_store(store, cached)
        return open_store(cached)
    return store


def _loader(
    config: DataConfig, dataset: WindowDataset, sampler: WindowSampler
) -> DataLoader[Example]:
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
    """Return the sampler and loader of split ``name``, or None if it has no examples."""
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
    """Read, split, resample and normalise the data, and build one loader per split."""
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
