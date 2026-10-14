"""Data pipeline: split the unified datasets and serve training windows.

``build_data(DataConfig(...))`` returns an ``IoBTData`` with one loader per split. Every option
(split strategy, window sampling, spectral representation, where the STFT runs) is a field of
``DataConfig`` and its nested configs.
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
