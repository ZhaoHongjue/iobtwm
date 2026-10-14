"""Assign clips to train/val/test.

Each strategy returns one code per clip of a ``ClipTable``: the index of the split in
``SPLIT_NAMES`` (0 = train, 1 = val, 2 = test) or ``DROPPED`` for clips that belong to no split.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Final

import numpy as np
import numpy.typing as npt

from iobtwm.data.config import SplitConfig, StratifyKey
from iobtwm.data.manifest import ClipTable

logger = logging.getLogger(__name__)

DROPPED: Final[int] = -1

_WEST_POINT_TRAIN_RUNS: Final[frozenset[int]] = frozenset({0, 1, 2, 6, 7, 8})


def assign_splits(table: ClipTable, config: SplitConfig) -> npt.NDArray[np.int8]:
    """Return the split code of every clip in ``table`` according to ``config.strategy``."""
    match config.strategy:
        case "random_clip":
            return split_random_clip(len(table), config.ratios, config.seed)
        case "time":
            return split_time(table, config.ratios, config.gap_seconds)
        case "record":
            return split_record(table, config)
        case "official":
            return split_official(table, config)
        case "holdout":
            return split_holdout(table, config)


def _counts(n: int, ratios: tuple[float, float, float]) -> tuple[int, int]:
    """Return how many of ``n`` items go to train and val; the rest go to test."""
    n_train = math.floor(n * ratios[0])
    n_val = math.floor(n * ratios[1])
    return n_train, n_val


def split_random_clip(
    n: int, ratios: tuple[float, float, float], seed: int
) -> npt.NDArray[np.int8]:
    """Assign each clip independently at random, like the early group papers did on MOD.

    Neighbouring clips of one recording land in different splits, so test scores are optimistic.
    It is kept to reproduce published numbers.
    """
    order = np.random.default_rng(seed).permutation(n)
    n_train, n_val = _counts(n, ratios)
    codes = np.full(n, 2, dtype=np.int8)
    codes[order[:n_train]] = 0
    codes[order[n_train : n_train + n_val]] = 1
    return codes


def split_time(
    table: ClipTable, ratios: tuple[float, float, float], gap_seconds: float
) -> npt.NDArray[np.int8]:
    """Cut every recording chronologically: the first part trains, then val, then test.

    The cut points are placed on the clip index of each recording, shared by all its nodes.
    Clips within ``gap_seconds`` after a cut are dropped, and with a zero test ratio the clips
    left over by rounding are dropped too.
    """
    gap = math.ceil(gap_seconds / table.clip_seconds - 1e-9)
    codes = np.full(len(table), DROPPED, dtype=np.int8)
    for record in np.unique(table.record):
        mask = table.record == record
        clips = table.clip[mask]
        first = int(clips.min())
        n_train, n_val = _counts(int(clips.max()) - first + 1, ratios)
        position = clips - first
        record_codes = np.full(len(clips), DROPPED, dtype=np.int8)
        record_codes[position < n_train] = 0
        record_codes[(position >= n_train + gap) & (position < n_train + n_val)] = 1
        if ratios[2] > 0:
            record_codes[position >= n_train + n_val + gap] = 2
        codes[mask] = record_codes
    return codes


def allocate_records(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Return how many of ``n`` recordings go to (train, val, test).

    Test gets at least one recording when its ratio is non-zero, so every group is evaluated.
    Val only gets its rounded share, because many groups hold two or three recordings and a
    forced val recording in each would inflate val far beyond its ratio. Train always keeps at
    least one recording, so a single recording goes to train.
    """
    n_test = max(1, round(n * ratios[2])) if ratios[2] > 0 else 0
    n_val = round(n * ratios[1])
    while n - n_test - n_val < 1:
        if n_val > 0:
            n_val -= 1
        else:
            n_test -= 1
    return n - n_val - n_test, n_val, n_test


def _first_rows(table: ClipTable) -> npt.NDArray[np.int64]:
    """Return one ``ClipTable`` row of every recording, indexed by recording."""
    _, first = np.unique(table.record, return_index=True)
    rows: npt.NDArray[np.int64] = first.astype(np.int64)
    return rows


def _record_groups(
    table: ClipTable, records: npt.NDArray[np.int64], stratify: tuple[StratifyKey, ...]
) -> dict[tuple[str, ...], list[int]]:
    """Group ``records`` by the stratification keys in ``stratify``."""
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for record in records.tolist():
        mask = table.record == record
        key: list[str] = []
        for name in stratify:
            if name == "dataset":
                key.append(table.datasets[int(table.dataset[mask][0])])
            elif name == "scene":
                key.append(table.scenes[int(table.scene[mask][0])])
            else:
                present = table.labels[mask].any(axis=0)
                key.append("+".join(c for c, p in zip(table.classes, present, strict=True) if p))
        groups[tuple(key)].append(record)
    return groups


def _assign_records(
    table: ClipTable,
    records: npt.NDArray[np.int64],
    config: SplitConfig,
    ratios: tuple[float, float, float],
    record_code: npt.NDArray[np.int8],
) -> None:
    """Write the split of every recording in ``records`` into ``record_code``.

    Recordings are shuffled within each stratification group and divided by
    ``allocate_records``. Groups with one recording go to train and are logged.
    """
    rng = np.random.default_rng(config.seed)
    for key, members in sorted(_record_groups(table, records, config.stratify).items()):
        if len(members) == 1:
            logger.info("record split: only one recording for %s, it goes to train", key)
        order = [members[i] for i in rng.permutation(len(members))]
        n_train, n_val, _ = allocate_records(len(order), ratios)
        boundaries = (n_train, n_train + n_val)
        for position, record in enumerate(order):
            record_code[record] = np.searchsorted(boundaries, position, side="right")


def split_record(table: ClipTable, config: SplitConfig) -> npt.NDArray[np.int8]:
    """Keep every recording whole and assign whole recordings to splits."""
    record_code = np.zeros(len(table.records), dtype=np.int8)
    all_records = np.arange(len(table.records), dtype=np.int64)
    _assign_records(table, all_records, config, config.ratios, record_code)
    codes: npt.NDArray[np.int8] = record_code[table.record]
    return codes


def official_code(dataset: str, run: int) -> int | None:
    """Return the published split of a run (0 train, 2 test), or None if there is none.

    M3N-VC and ICT train on even runs and hold out odd runs. West Point trains on runs 0-2 and
    6-8 and holds out the rest (``splits/2026-04-14-West-Point/indices``). ICT and West Point
    call their held-out runs "val"; here they are test, like M3N-VC.
    """
    match dataset:
        case "M3N-VC" | "ICT":
            return 0 if run % 2 == 0 else 2
        case "WestPoint":
            return 0 if run in _WEST_POINT_TRAIN_RUNS else 2
        case _:
            return None


def split_official(table: ClipTable, config: SplitConfig) -> npt.NDArray[np.int8]:
    """Use the published run-level splits where they exist, ``record`` elsewhere."""
    record_code = np.zeros(len(table.records), dtype=np.int8)
    rest: list[int] = []
    for record, row in enumerate(_first_rows(table).tolist()):
        code = official_code(table.datasets[table.dataset[row]], int(table.run[row]))
        if code is None:
            rest.append(record)
        else:
            record_code[record] = code
    _assign_records(table, np.asarray(rest, dtype=np.int64), config, config.ratios, record_code)
    codes: npt.NDArray[np.int8] = record_code[table.record]
    return codes


def record_names(table: ClipTable, row: int) -> set[str]:
    """Return the names a ``holdout`` entry can use to select the recording of ``row``."""
    dataset = table.datasets[table.dataset[row]]
    names = {dataset, table.subsets[table.subset[row]].name}
    scene = table.scenes[table.scene[row]]
    if scene:
        names.add(f"{dataset}/{scene}")
    return names


def split_holdout(table: ClipTable, config: SplitConfig) -> npt.NDArray[np.int8]:
    """Send held-out recordings to test and split the rest into train and val by recording.

    Raises:
        ValueError: If a ``holdout`` name matches no recording, or every recording is held out.
    """
    wanted = set(config.holdout)
    record_code = np.full(len(table.records), 2, dtype=np.int8)
    rest: list[int] = []
    matched: set[str] = set()
    for record, row in enumerate(_first_rows(table).tolist()):
        hits = record_names(table, row) & wanted
        matched |= hits
        if not hits:
            rest.append(record)
    if unknown := wanted - matched:
        raise ValueError(f"holdout names match no recording: {sorted(unknown)}")
    if not rest:
        raise ValueError("holdout leaves no recording for training")
    train, val, _ = config.ratios
    ratios = (train / (train + val), val / (train + val), 0.0)
    _assign_records(table, np.asarray(rest, dtype=np.int64), config, ratios, record_code)
    codes: npt.NDArray[np.int8] = record_code[table.record]
    return codes
