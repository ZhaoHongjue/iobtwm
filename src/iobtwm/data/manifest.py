"""Read unified manifests into one flat table of clips.

This is the first pipeline stage. ``load_clip_table`` reads ``info.json`` and
``manifest.parquet`` of every selected subset, drops rows that duplicate other rows (``_dwt``
wavelet twins in Parkland, ``o1000`` half-offset clips in M3N-VC), and merges the rest into one
``ClipTable`` of numpy columns. Later stages only use this table and the HDF5 row numbers in it;
they never read the manifests again.
"""

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

LABEL_ALIASES: Final[dict[str, str]] = {"polairs": "polaris", "warhog": "warthog"}
"""Spelling variants found in the source file names, mapped to the canonical class name."""

# Manifest columns the pipeline needs; the others (hashes, source paths, ...) are skipped.
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

    The vocabulary tuples hold names; the integer array columns of the same name index into
    them, e.g. ``table.datasets[table.dataset[i]]`` is the dataset of row ``i``.

    Attributes:
        subsets: Info of every loaded subset, in ``DataConfig.subsets`` order.
        datasets: Sorted dataset names (``GracesQuarters``, ``M3N-VC``, ...).
        records: Sorted recording keys, see ``record_key``.
        nodes: Sorted node names (``rs1``, ``gq-4``, ...).
        scenes: Sorted scene names; ``""`` for datasets without scenes.
        classes: Sorted class names; the columns of ``labels``.
        clip_seconds: Duration of one clip, shared by all subsets.
        subset: Index into ``subsets`` per row.
        row: Row of this clip in its subset's ``signals.h5``.
        dataset: Index into ``datasets`` per row.
        record: Index into ``records`` per row.
        node: Index into ``nodes`` per row.
        scene: Index into ``scenes`` per row.
        run: Run number, or -1 for named recordings (Parkland).
        clip: Clip number within its recording and node; consecutive numbers are consecutive in
            time.
        clip_start: Clip start time in seconds since the epoch where known (M3N-VC), else NaN.
        labels: Multi-hot ``[rows, classes]`` matrix of the vehicles present.
        distance: Distance in metres to the nearest labelled vehicle, NaN if unknown.
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
        """Return the number of clips (rows)."""
        return len(self.row)


@dataclass(frozen=True, slots=True)
class _RawRows:
    """Manifest rows of one subset after filtering, before vocabularies are built.

    Every list has one entry per kept row; the fields mirror the ``ClipTable`` columns but hold
    names instead of vocabulary indices.
    """

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
    """Read sample rates and clip duration of a subset from its ``info.json``.

    Args:
        root: The unified root directory.
        subset: Subset path relative to ``root``.

    Returns:
        The subset's native rates and clip duration.

    Raises:
        ValueError: If the subset is not a waveform subset or its modalities disagree on the
            clip duration.
    """
    info = json.loads((root / subset / "info.json").read_text())
    if info["representation"] != "waveform":
        kind = info["representation"]
        raise ValueError(f"{subset} stores {kind!r}, only waveforms are supported")
    rates: dict[Modality, float] = {"audio": info["audio_sr"], "seismic": info["seismic_sr"]}
    # Rows are flattened clips, so the last row dimension divided by the rate is the duration.
    durations = {
        m: info["columns"][f"signals/{m}"]["row_shape"][-1] / rate for m, rate in rates.items()
    }
    if not math.isclose(durations["audio"], durations["seismic"]):
        raise ValueError(f"{subset}: modalities disagree on clip duration: {durations}")
    return SubsetInfo(name=subset, rates=rates, clip_seconds=durations["audio"])


def normalize_label(label: str) -> str:
    """Return the canonical class name of a label.

    Args:
        label: A label as stored in the manifest.

    Returns:
        The label in lower case, with known spelling variants (``LABEL_ALIASES``) fixed.
    """
    lowered = label.lower()
    return LABEL_ALIASES.get(lowered, lowered)


def record_key(
    subset: str, dataset: str, recording: str | None, run: int | None, scene: str | None = None
) -> str:
    """Return the key that identifies one continuous recording session.

    Parkland names recordings explicitly and spreads the nodes of one recording over
    ``individual_time_samples`` and ``extra_time_samples``, so its key ignores the subset. Other
    datasets number runs within a subset and, where scenes exist, within a scene.

    Args:
        subset: Subset path of the row.
        dataset: Dataset name of the row.
        recording: Recording name, if the dataset names its recordings.
        run: Run number, if the dataset numbers its runs.
        scene: Scene name, if the dataset has scenes.

    Returns:
        ``<dataset>/<recording>``, ``<subset>/<scene>/run<run>`` or ``<subset>/run<run>``.

    Raises:
        ValueError: If neither a recording name nor a run number is available.
    """
    if recording is not None:
        return f"{dataset}/{recording}"
    if run is not None:
        return f"{subset}/{scene}/run{run}" if scene else f"{subset}/run{run}"
    raise ValueError(f"{subset}: row has neither a recording name nor a run number")


def _nearest_distance(distances: list[float | None]) -> float:
    """Return the smallest known distance of a row, or NaN if no distance is known.

    Args:
        distances: One distance per labelled vehicle; None where it is unknown.
    """
    known = [d for d in distances if d is not None]
    return min(known) if known else math.nan


def _read_rows(root: Path, subset: str) -> _RawRows:
    """Read the manifest of a subset, dropping wavelet duplicates and offset clips.

    Args:
        root: The unified root directory.
        subset: Subset path relative to ``root``.

    Returns:
        The kept rows, with record keys built and labels normalised.
    """
    table = pq.read_table(root / subset / "manifest.parquet", columns=_COLUMNS).to_pydict()
    out = _RawRows([], [], [], [], [], [], [], [], [], [])
    for i, sample_id in enumerate(table["sample_id"]):
        offset = table["offset_ms"][i]
        # `_dwt` rows repeat the waveform of their twin; offset clips half-overlap normal clips.
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
    """Turn names into a sorted vocabulary and one vocabulary index per value.

    Args:
        values: One name per row.

    Returns:
        The sorted distinct names and the index of each value in them.
    """
    vocab, inverse = np.unique(np.asarray(values, dtype=object), return_inverse=True)
    return tuple(str(v) for v in vocab), inverse.astype(np.int32)


def _multi_hot(
    labels: list[tuple[str, ...]],
) -> tuple[tuple[str, ...], npt.NDArray[np.bool_]]:
    """Turn per-row label sets into a class vocabulary and a multi-hot matrix.

    Args:
        labels: The class names present in each row.

    Returns:
        The sorted class names and a ``[rows, classes]`` boolean matrix.
    """
    classes = tuple(sorted({c for row in labels for c in row}))
    index = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((len(labels), len(classes)), dtype=np.bool_)
    for i, row in enumerate(labels):
        matrix[i, [index[c] for c in row]] = True
    return classes, matrix


def load_clip_table(root: Path, subsets: tuple[str, ...]) -> ClipTable:
    """Read and merge the manifests of the given subsets.

    Args:
        root: The unified root directory.
        subsets: Subset paths relative to ``root``.

    Returns:
        One table of all kept clips of all subsets.

    Raises:
        ValueError: If a subset is not a waveform subset, the subsets use different clip
            durations, or the same (record, node, clip) appears twice.
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
    # A duplicate would mean two subsets hold the same recording, i.e. a bad subset selection.
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
