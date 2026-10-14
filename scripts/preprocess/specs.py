"""Registry of source folders and how to read each family of sample files.

Every folder under ``datasets/`` that contains ``.pt`` samples is either listed in
:data:`SUBSETS` (converted) or :data:`EXCLUDED` (skipped, with the reason). Conversion and
verification both fail on folders that are in neither, so new data cannot slip through.

Normalized columns (labels, distances, node, ...) are conveniences derived from the original
sample and its filename. They never replace the original metadata, which is stored verbatim.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import torch

type Representation = Literal["waveform", "fft"]

# Label ids used by the original preprocessing code; the ids are stored in the samples.
# cs537aiot/src/data_preprocess/Parkland/extract_samples.py (LABELS)
PARKLAND_LABELS: Final[Mapping[int, str]] = {
    0: "polaris",
    1: "warhog",
    2: "silverado",
    3: "motor",
    4: "tesla",
    5: "mustang",
    6: "walk",
    7: "bicycle",
    8: "forester",
    9: "pickup",
    10: "scooter",
}
# Inferred from GracesQuarters day_*_sensor_all filenames vs. integer labels.
GQ_LABELS: Final[Mapping[int, str]] = {0: "polaris", 1: "warthog", 2: "silverado", 3: "husky"}
# Inferred from ict_pt_files filenames vs. integer label pairs.
ICT_MIXED_LABELS: Final[Mapping[int, str]] = {0: "miata", 1: "gle350", 2: "mustang", 3: "cx30"}


@dataclass(frozen=True, slots=True)
class Normalized:
    """Per-sample metadata in one common shape across datasets.

    Attributes:
        labels: Class names, lowercase, in the order the source stores them.
        distances_m: Distance in metres per label (aligned with ``labels``); ``None`` if
            unknown or a sentinel (-1, NaN, inf) in the source.
        name_labels: Class names implied by the filename, for cross-checking (``None`` when
            the filename carries none).
        node: Sensor node id (``rs3``, ``gq-1``, ...).
        run: Recording run number.
        clip: Clip index within the run/recording (consecutive clips are consecutive in time).
        recording: Recording/session folder name (Parkland only).
        scene: Scene id (M3N-VC only).
        offset_ms: Clip-grid offset in ms (M3N-VC ``o1000`` clips overlap plain clips by half).
        node_xy: Sensor position, local metres (M3N-VC only).
        vehicle_xy: Vehicle position, local metres (M3N-VC only).
        clip_start: Clip start time, unix seconds (M3N-VC only).
    """

    labels: tuple[str, ...]
    distances_m: tuple[float | None, ...]
    name_labels: tuple[str, ...] | None = None
    node: str | None = None
    run: int | None = None
    clip: int | None = None
    recording: str | None = None
    scene: str | None = None
    offset_ms: int | None = None
    node_xy: tuple[float, ...] | None = None
    vehicle_xy: tuple[float, ...] | None = None
    clip_start: float | None = None


type Sample = Mapping[str, object]
type Normalizer = Callable[[Sample, re.Match[str]], Normalized]


@dataclass(frozen=True, slots=True)
class Family:
    """How to parse filenames and normalize metadata for one family of sample files."""

    name: str
    pattern: re.Pattern[str]
    normalize: Normalizer


@dataclass(frozen=True, slots=True)
class SubsetSpec:
    """One source folder that gets converted into one unified subset.

    Attributes:
        folder: Path relative to the datasets root; also the output path.
        dataset: Dataset name shared by related folders.
        family: Filename/metadata family.
        representation: ``waveform`` (time samples, stored flattened) or ``fft``
            (real/imag spectra per interval, stored in their original shape).
        audio_sr: Audio sample rate in Hz.
        seismic_sr: Seismic sample rate in Hz.
        note: Caveats worth surfacing (e.g. inferred sample rates).
    """

    folder: str
    dataset: str
    family: Family
    representation: Representation = "waveform"
    audio_sr: float = 8000.0
    seismic_sr: float = 100.0
    note: str | None = None


# ---------------------------------------------------------------------------- helpers


def _label_names(raw: object, id_map: Mapping[int, str] | None) -> tuple[str, ...]:
    """Turn any of the label encodings found in the sources into lowercase names."""
    if isinstance(raw, str):
        return (raw.lower(),)
    if isinstance(raw, torch.Tensor):
        if id_map is None:
            raise ValueError("integer label without an id map")
        return tuple(id_map[int(v)] for v in raw.reshape(-1).tolist())
    if isinstance(raw, np.ndarray):
        return tuple(n for v in raw.reshape(-1).tolist() for n in _label_names(v, id_map))
    if isinstance(raw, list | tuple):
        return tuple(n for v in raw for n in _label_names(v, id_map))
    if isinstance(raw, set | frozenset):
        return tuple(sorted(n for v in raw for n in _label_names(v, id_map)))
    raise TypeError(f"unsupported label {raw!r}")


def _clean_distance(raw: object) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, int | float | np.floating | np.integer):
        raise TypeError(f"unsupported distance {raw!r}")
    value = float(raw)
    return value if math.isfinite(value) and value >= 0 else None


def _distances(raw: object, labels: tuple[str, ...]) -> tuple[float | None, ...]:
    """Align a scalar or per-label distance with ``labels``."""
    if raw is None:
        return (None,) * len(labels)
    if isinstance(raw, dict):
        by_name = {str(k).lower(): _clean_distance(v) for k, v in raw.items()}
        return tuple(by_name.get(label) for label in labels)
    value = _clean_distance(raw)
    return (value,) if len(labels) == 1 else (None,) * len(labels)


def _xy(raw: object) -> tuple[float, ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, list | tuple):
        raise TypeError(f"unsupported coordinates {raw!r}")
    return tuple(float(v) for v in raw)


def _int(m: re.Match[str], group: str) -> int | None:
    value = m.group(group)
    return None if value is None else int(value)


# ---------------------------------------------------------------------------- families


def _parkland(sample: Sample, m: re.Match[str]) -> Normalized:
    recording = m.group("recording")
    name_label = next(
        (n for n in PARKLAND_LABELS.values() if recording.lower().startswith(n)), None
    )
    # The recording folder names the vehicle. Stored ids agree with it in every time-domain
    # folder, but 16,907 legacy samples in individual_samples use an incompatible id scheme
    # (e.g. forester and mustang both stored as 3), so the name wins when it is recognizable.
    if name_label is not None:
        labels: tuple[str, ...] = (name_label,)
    else:
        labels = _label_names(sample["label"], PARKLAND_LABELS)
    return Normalized(
        labels=labels,
        distances_m=(None,) * len(labels),
        name_labels=None if name_label is None else (name_label,),
        node=m.group("node"),
        clip=_int(m, "clip"),
        recording=recording,
    )


def _gq_sensor_all(sample: Sample, m: re.Match[str]) -> Normalized:
    labels = _label_names(sample["label"], GQ_LABELS)
    return Normalized(
        labels=labels,
        distances_m=(None,) * len(labels),
        name_labels=(m.group("prefix"),),
        node=f"gq-{m.group('node')}",
        run=_int(m, "run"),
        clip=_int(m, "clip"),
    )


def _gq_2024(sample: Sample, m: re.Match[str]) -> Normalized:
    labels = _label_names(sample["label"], None)
    prefix = m.groupdict().get("prefix")
    return Normalized(
        labels=labels,
        distances_m=_distances(sample.get("distance"), labels),
        name_labels=None if prefix is None else (prefix,),
        node=m.group("node"),
        run=_int(m, "run"),
        clip=_int(m, "clip"),
    )


def _mixed_from_name(sample: Sample, m: re.Match[str]) -> Normalized:
    # The stored ids are unusable here: 5 of the 6 GQ pairs store (0, 1) whatever the vehicles,
    # so labels come from the filename written by the mixing script.
    labels = (m.group("a"), m.group("b"))
    return Normalized(
        labels=labels,
        distances_m=(None,) * len(labels),
        name_labels=labels,
        clip=_int(m, "clip"),
    )


def _mixed(id_map: Mapping[int, str]) -> Normalizer:
    def normalize(sample: Sample, m: re.Match[str]) -> Normalized:
        labels = _label_names(sample["label"], id_map)
        return Normalized(
            labels=labels,
            distances_m=(None,) * len(labels),
            name_labels=(m.group("a"), m.group("b")),
            clip=_int(m, "clip"),
        )

    return normalize


def _ict(sample: Sample, m: re.Match[str]) -> Normalized:
    labels = _label_names(sample["label"], None)
    return Normalized(
        labels=labels,
        distances_m=_distances(sample.get("distance"), labels),
        name_labels=(m.group("prefix"),),
        node=m.group("node"),
        run=_int(m, "run"),
        clip=_int(m, "clip"),
    )


def _westpoint(sample: Sample, m: re.Match[str]) -> Normalized:
    labels = _label_names(sample["label"], None)
    return Normalized(
        labels=labels,
        distances_m=_distances(sample.get("distance"), labels),
        node=m.group("node"),
        run=_int(m, "run"),
        clip=_int(m, "clip"),
    )


def _m3n(sample: Sample, m: re.Match[str]) -> Normalized:
    labels = _label_names(sample["label"], None)
    data = sample["data"]
    assert isinstance(data, dict)
    run, clip, scene, clip_start = (
        sample["run"],
        sample["clip"],
        sample["scene"],
        sample["clip_start"],
    )
    assert isinstance(run, int) and isinstance(clip, int) and isinstance(scene, str)
    assert isinstance(clip_start, float)
    return Normalized(
        labels=labels,
        distances_m=_distances(sample.get("distance"), labels),
        name_labels=tuple(m.group("labels").split("-")),
        node=next(iter(data)),
        run=run,
        clip=clip,
        scene=scene,
        offset_ms=int(m.group("offset") or 0),
        node_xy=_xy(sample.get("node_xy")),
        vehicle_xy=_xy(sample.get("vehicle_xy")),
        clip_start=clip_start,
    )


PARKLAND: Final = Family(
    "parkland",
    re.compile(r"^(?P<recording>.+?)_(?P<node>rs\d+)_(?P<clip>\d+)(?P<dwt>_dwt)?\.pt$"),
    _parkland,
)
GQ_SENSOR_ALL: Final = Family(
    "gq_sensor_all",
    re.compile(r"^(?P<prefix>[a-z]+)_run(?P<run>\d+)_gq_(?P<node>\d+)_(?P<clip>\d+)\.pt$"),
    _gq_sensor_all,
)
GQ_2024_NAMED: Final = Family(
    "gq_2024_named",
    re.compile(
        r"^(?P<prefix>[a-z]+)_run(?P<run>\d+)_(?P<node>gq-\d+)_(?P<clip>\d+)_(?P<dist>-?\d+)\.pt$"
    ),
    _gq_2024,
)
GQ_2024_RUNS: Final = Family(
    "gq_2024_runs",
    re.compile(r"^run(?P<run>\d+)_(?P<node>gq-\d+)_(?P<clip>\d+)\.pt$"),
    _gq_2024,
)
GQ_MIXED: Final = Family(
    "gq_mixed",
    re.compile(r"^(?P<a>[a-z0-9]+)_(?P<b>[a-z0-9]+)_mixed_(?P<clip>\d+)\.pt$"),
    _mixed_from_name,
)
ICT_MIXED: Final = Family("ict_mixed", GQ_MIXED.pattern, _mixed(ICT_MIXED_LABELS))
ICT: Final = Family(
    "ict",
    re.compile(
        r"^(?P<prefix>[a-z0-9]+)_run(?P<run>\d+)_(?P<node>rs\d+)_(?P<clip>\d+)_(?P<dist>\d+)\.pt$"
    ),
    _ict,
)
WESTPOINT: Final = Family(
    "westpoint",
    re.compile(r"^run(?P<run>\d+)_(?P<node>rs\d+)_(?P<clip>\d+)__(?P<dist>-?\d+)\.pt$"),
    _westpoint,
)
M3N: Final = Family(
    "m3n",
    re.compile(
        r"^(?P<scene>[a-z]\d+)_(?P<labels>[a-z0-9-]+)_run(?P<run>\d+)(?:_o(?P<offset>\d+))?"
        r"_(?P<node>rs\d+)_(?P<clip>\d+)(?:_(?P<dist>\d+))?\.pt$"
    ),
    _m3n,
)


# ---------------------------------------------------------------------------- registry

_GQ: Final = "GracesQuarters"
_GQ_SENSOR_ALL_FOLDERS: Final = (
    "day_1_sensor_all",
    "day_2_sensor_all",
    "GracesQuartersFiltered",
    "GracesQuartersFilteredNone",
    "GracesQuartersDay1Filtered",
    "GracesQuartersDay1FilteredNone",
    "GracesQuartersDay1Filtered_random",
    "GracesQuartersDay1FilteredNone_random",
    "GracesQuartersDay2Filtered",
    "GracesQuartersDay2FilteredNone",
    "GracesQuartersDay2Filtered_random",
    "GracesQuartersDay2FilteredNone_random",
)
_M3N_SCENES: Final = ("a06", "h08", "h24", "i22", "i29", "s31")
_FFT_NOTE: Final = (
    "Frequency domain: per 0.2 s interval FFT, channel 0 = real, channel 1 = imag. "
    "Some of these samples exist in no time-domain folder, so they are kept as-is."
)

SUBSETS: Final[tuple[SubsetSpec, ...]] = (
    SubsetSpec("Parkland/individual_time_samples", "Parkland", PARKLAND),
    SubsetSpec("Parkland/individual_time_samples_with_wind", "Parkland", PARKLAND),
    SubsetSpec("Parkland/extra_time_samples", "Parkland", PARKLAND),
    SubsetSpec("Parkland/individual_samples", "Parkland", PARKLAND, "fft", note=_FFT_NOTE),
    *(SubsetSpec(f"{_GQ}/{f}", _GQ, GQ_SENSOR_ALL) for f in _GQ_SENSOR_ALL_FOLDERS),
    SubsetSpec(f"{_GQ}/2024-07-19-GQ-Background/individual_time_samples", _GQ, GQ_2024_NAMED),
    SubsetSpec(f"{_GQ}/2024-08-05-GQ/individual_time_samples", _GQ, GQ_2024_NAMED),
    SubsetSpec(f"{_GQ}/2024-08-06-GQ/individual_time_samples", _GQ, GQ_2024_RUNS),
    SubsetSpec(f"{_GQ}/2024-08-07-GQ/individual_time_samples", _GQ, GQ_2024_RUNS),
    SubsetSpec(f"{_GQ}/2024-08-07-GQ_past/individual_time_samples", _GQ, GQ_2024_RUNS),
    SubsetSpec(
        f"{_GQ}/GracesQuartersMixed",
        _GQ,
        GQ_MIXED,
        note="Stored integer labels disagree with filenames (5 of 6 vehicle pairs store (0, 1)); "
        "the labels column is taken from the filename, raw ids are kept in original_meta.",
    ),
    SubsetSpec("ICT_train_dis/2024-03-29-ICT/individual_time_samples", "ICT", ICT),
    SubsetSpec("ICT_val_dis/2024-03-29-ICT/individual_time_samples", "ICT", ICT),
    SubsetSpec(
        "ict_pt_files",
        "ICT",
        ICT_MIXED,
        note="Sample rates inferred (32000 audio / 400 seismic samples, same 80:1 ratio as "
        "other ICT data, i.e. 4 s at 8000/100 Hz); not recorded in the source.",
    ),
    SubsetSpec("2026-04-14-West-Point/individual_time_samples", "WestPoint", WESTPOINT),
    *(
        SubsetSpec(
            f"M3N-VC/{s}/individual_time_samples", "M3N-VC", M3N, audio_sr=1600.0, seismic_sr=200.0
        )
        for s in _M3N_SCENES
    ),
)

_DERIVED: Final = "FFT of {} (same filenames); regenerable, checked by check_derived.py"
EXCLUDED: Final[Mapping[str, str]] = {
    "Parkland/individual_freq_samples": _DERIVED.format("Parkland/individual_time_samples"),
    "Parkland/individual_freq_samples_with_wind": _DERIVED.format(
        "Parkland/individual_time_samples_with_wind"
    ),
    "Parkland/extra_freq_samples": _DERIVED.format("Parkland/extra_time_samples"),
}
# Excluded frequency folder -> converted time-domain folder it is derived from.
DERIVED_FROM: Final[Mapping[str, str]] = {
    "Parkland/individual_freq_samples": "Parkland/individual_time_samples",
    "Parkland/individual_freq_samples_with_wind": "Parkland/individual_time_samples_with_wind",
    "Parkland/extra_freq_samples": "Parkland/extra_time_samples",
}


def spec_by_folder(specs: tuple[SubsetSpec, ...] = SUBSETS) -> dict[str, SubsetSpec]:
    """Index specs by folder, rejecting duplicates."""
    out: dict[str, SubsetSpec] = {}
    for spec in specs:
        if spec.folder in out:
            raise ValueError(f"duplicate spec for {spec.folder}")
        out[spec.folder] = spec
    return out


def normalize(spec: SubsetSpec, filename: str, sample: object) -> Normalized:
    """Parse the filename and normalize the sample's metadata.

    Raises:
        ValueError: If the filename does not match the family's pattern.
        TypeError: If the sample is not a dict.
    """
    m = spec.family.pattern.match(filename)
    if m is None:
        raise ValueError(f"{spec.folder}/{filename} does not match family {spec.family.name}")
    if not isinstance(sample, dict):
        raise TypeError(f"{spec.folder}/{filename}: sample is {type(sample).__name__}, not dict")
    return spec.family.normalize(sample, m)
