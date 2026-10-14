"""Data pipeline: split the unified datasets and serve training windows.

``build_data(DataConfig(...))`` returns an ``IoBTData`` with one loader per split. Every option
(split strategy, window sampling, spectral representation, where the STFT runs) is a field of
``DataConfig`` and its nested configs.

Modules, in pipeline order:
    config: All settings.
    manifest: Read the unified manifests into a ``ClipTable``.
    splits: Assign clips to train/val/test.
    segments: Group clips into time-continuous segments.
    store: Load, resample, normalise and cache the segment waveforms.
    sampling: Decide which windows form each epoch.
    dataset: Cut a window into an example.
    spectral: Run-time STFT.
    builder: Tie the stages together.

See ``README.md`` next to this file for a guide and ``docs/data-package.md`` for the reference.
"""

from __future__ import annotations

from iobtwm.data.builder import IoBTData, SplitSummary, build_data
from iobtwm.data.config import (
    DEFAULT_SUBSETS,
    DataConfig,
    SamplingConfig,
    SpectralConfig,
    SplitConfig,
)
from iobtwm.data.dataset import Example
from iobtwm.data.spectral import SpectralTransform

__all__ = [
    "DEFAULT_SUBSETS",
    "DataConfig",
    "Example",
    "IoBTData",
    "SamplingConfig",
    "SpectralConfig",
    "SpectralTransform",
    "SplitConfig",
    "SplitSummary",
    "build_data",
]
