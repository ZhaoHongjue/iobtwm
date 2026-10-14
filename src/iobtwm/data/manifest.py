"""Read unified manifests into one flat table of clips."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from iobtwm.data.config import Modality

logger = logging.getLogger(__name__)

# Spelling variants in the source file names.
LABEL_ALIASES: Final[dict[str, str]] = {"polairs": "polaris", "warhog": "warthog"}

_COLUMNS: Final[list[str]] = [
    "sample_index",
    "sample_id",
    "dataset",
    "representation",
    "labels",
    "distances_m",
    "node",
    "run",
    "clip",
    "recording",
    "scene",
    "offset_ms",
    "clip_start",
]


@dataclass(frozen=True, slots=True)
class SubsetInfo:
    """Per-subset facts from ``info.json``.

    Attributes:
        name: Subset path relative to the unified root.
        rates: Native sample rate of each modality in Hz.
        clip_seconds: Duration of one clip.
    """

    name: str
    rates: dict[Modality, float]
    clip_seconds: float


@dataclass(frozen=True, slots=True)
class ClipTable:
    """All usable clips of the selected subsets, one row per (clip, node).

    Integer columns index into the vocabulary tuples. ``run`` is the run number (-1 for named
    recordings). ``labels`` is a multi-hot matrix over ``classes`` and ``distance`` is the distance
    to the nearest labelled vehicle (NaN if unknown).
    """

    subsets: tuple[SubsetInfo, ...]
    datasets: tuple[str, ...]
    records: tuple[str, ...]
    nodes: tuple[str, ...]
    scenes: tuple[str, ...]
    classes: tuple[str, ...]
    clip_seconds: float
    subset: npt.NDArray[np.int32]
    row: npt.NDArray[np.int64]
    dataset: npt.NDArray[np.int32]
    record: npt.NDArray[np.int32]
    node: npt.NDArray[np.int32]
    scene: npt.NDArray[np.int32]
    run: npt.NDArray[np.int64]
    clip: npt.NDArray[np.int64]
    clip_start: npt.NDArray[np.float64]
    labels: npt.NDArray[np.bool_]
    distance: npt.NDArray[np.float32]

    def __len__(self) -> int:
        return len(self.row)


@dataclass(frozen=True, slots=True)
class _RawRows:
    """Manifest rows of one subset after filtering, before vocabularies are built."""

    row: list[int]
    dataset: list[str]
    record: list[str]
    node: list[str]
    scene: list[str]
    run: list[int]
    clip: list[int]
    clip_start: list[float]
    labels: list[tuple[str, ...]]
    distance: list[float]


def read_subset_info(root: Path, subset: str) -> SubsetInfo:
    """Read sample rates and clip duration of ``subset`` from its ``info.json``.

    Raises:
        ValueError: If the subset is not a waveform subset or its modalities disagree on the
            clip duration.
    """
    info = json.loads((root / subset / "info.json").read_text())
    if info["representation"] != "waveform":
        kind = info["representation"]
        raise ValueError(f"{subset} stores {kind!r}, only waveforms are supported")
    rates: dict[Modality, float] = {"audio": info["audio_sr"], "seismic": info["seismic_sr"]}
    durations = {
        m: info["columns"][f"signals/{m}"]["row_shape"][-1] / rate for m, rate in rates.items()
    }
    if not math.isclose(durations["audio"], durations["seismic"]):
        raise ValueError(f"{subset}: modalities disagree on clip duration: {durations}")
    return SubsetInfo(name=subset, rates=rates, clip_seconds=durations["audio"])


def normalize_label(label: str) -> str:
    """Lower-case ``label`` and fix known spelling variants."""
    lowered = label.lower()
    return LABEL_ALIASES.get(lowered, lowered)


def record_key(
    subset: str, dataset: str, recording: str | None, run: int | None, scene: str | None = None
) -> str:
    """Return the key that identifies one continuous recording session.

    Parkland names recordings explicitly and spreads the nodes of one recording over
    ``individual_time_samples`` and ``extra_time_samples``, so its key ignores the subset. Other
    datasets number runs within a subset and, where scenes exist, within a scene.

    Raises:
        ValueError: If neither a recording name nor a run number is available.
    """
    if recording is not None:
        return f"{dataset}/{recording}"
    if run is not None:
        return f"{subset}/{scene}/run{run}" if scene else f"{subset}/run{run}"
    raise ValueError(f"{subset}: row has neither a recording name nor a run number")


def _nearest_distance(distances: list[float | None]) -> float:
    known = [d for d in distances if d is not None]
    return min(known) if known else math.nan


def _read_rows(root: Path, subset: str) -> _RawRows:
    """Read the manifest of ``subset``, dropping wavelet duplicates and offset clips."""
    table = pq.read_table(root / subset / "manifest.parquet", columns=_COLUMNS).to_pydict()
    out = _RawRows([], [], [], [], [], [], [], [], [], [])
    for i, sample_id in enumerate(table["sample_id"]):
        offset = table["offset_ms"][i]
        if sample_id.endswith("_dwt") or (offset is not None and offset != 0):
            continue
        dataset = table["dataset"][i]
        start = table["clip_start"][i]
        out.row.append(table["sample_index"][i])
        out.dataset.append(dataset)
        recording, run, scene = table["recording"][i], table["run"][i], table["scene"][i]
        out.record.append(record_key(subset, dataset, recording, run, scene))
        out.node.append(table["node"][i] or "")
        out.scene.append(table["scene"][i] or "")
        out.run.append(-1 if table["run"][i] is None else table["run"][i])
        out.clip.append(table["clip"][i])
        out.clip_start.append(math.nan if start is None else start)
        out.labels.append(tuple(sorted({normalize_label(x) for x in table["labels"][i]})))
        out.distance.append(_nearest_distance(table["distances_m"][i]))
    return out


def _encode(values: list[str]) -> tuple[tuple[str, ...], npt.NDArray[np.int32]]:
    vocab, inverse = np.unique(np.asarray(values, dtype=object), return_inverse=True)
    return tuple(str(v) for v in vocab), inverse.astype(np.int32)


def _multi_hot(
    labels: list[tuple[str, ...]],
) -> tuple[tuple[str, ...], npt.NDArray[np.bool_]]:
    classes = tuple(sorted({c for row in labels for c in row}))
    index = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((len(labels), len(classes)), dtype=np.bool_)
    for i, row in enumerate(labels):
        matrix[i, [index[c] for c in row]] = True
    return classes, matrix


def load_clip_table(root: Path, subsets: tuple[str, ...]) -> ClipTable:
    """Read and merge the manifests of ``subsets``.

    Raises:
        ValueError: If the subsets use different clip durations or a (record, node, clip) appears
            twice.
    """
    infos = tuple(read_subset_info(root, s) for s in subsets)
    clip_seconds = {round(i.clip_seconds, 6) for i in infos}
    if len(clip_seconds) != 1:
        raise ValueError(f"subsets use different clip durations: {sorted(clip_seconds)}")
    raw = [_read_rows(root, s) for s in subsets]
    subset_idx = np.concatenate([np.full(len(r.row), i, dtype=np.int32) for i, r in enumerate(raw)])
    datasets, dataset = _encode([v for r in raw for v in r.dataset])
    records, record = _encode([v for r in raw for v in r.record])
    nodes, node = _encode([v for r in raw for v in r.node])
    scenes, scene = _encode([v for r in raw for v in r.scene])
    classes, labels = _multi_hot([v for r in raw for v in r.labels])
    clip = np.asarray([v for r in raw for v in r.clip], dtype=np.int64)
    keys = np.stack([record.astype(np.int64), node.astype(np.int64), clip], axis=1)
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError("the selected subsets contain the same (record, node, clip) twice")
    logger.info("loaded %d clips from %d subsets", len(clip), len(subsets))
    return ClipTable(
        subsets=infos,
        datasets=datasets,
        records=records,
        nodes=nodes,
        scenes=scenes,
        classes=classes,
        clip_seconds=clip_seconds.pop(),
        subset=subset_idx,
        row=np.asarray([v for r in raw for v in r.row], dtype=np.int64),
        dataset=dataset,
        record=record,
        node=node,
        scene=scene,
        run=np.asarray([v for r in raw for v in r.run], dtype=np.int64),
        clip=clip,
        clip_start=np.asarray([v for r in raw for v in r.clip_start], dtype=np.float64),
        labels=labels,
        distance=np.asarray([v for r in raw for v in r.distance], dtype=np.float32),
    )
