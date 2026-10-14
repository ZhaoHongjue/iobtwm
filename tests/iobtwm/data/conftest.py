"""A tiny unified dataset written to ``tmp_path``.

Toy/plain (audio 20 Hz, seismic 10 Hz, runs):
    run0: label a, nodes n1 and n2, clips 0-9.
    run1: label a, node n1, clips 0-9.
    run2: label b, node n1, clips 0-2 and 4-6 (a gap at 3).
    run3: label b, node n1, clips 0-5.
Toy/named (audio 40 Hz, seismic 10 Hz, named recordings, clip start times):
    rec_x: label "Warhog" (alias of warthog), node n1, clips 0-3, each with a ``_dwt`` twin and
        an ``offset_ms=1000`` twin that must be ignored.

Clip ``c`` of manifest row ``r`` holds ``r * 1000 + sample`` in audio and ``-(r * 1000 + sample)``
in seismic, so tests can check which rows were read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.preprocess.layout import MANIFEST_SCHEMA

CLIP_SECONDS = 2.0


@dataclass
class Row:
    """One manifest row of the toy dataset."""

    sample_id: str
    labels: list[str]
    clip: int
    node: str = "n1"
    run: int | None = None
    recording: str | None = None
    scene: str | None = None
    distances: list[float | None] = field(default_factory=lambda: [None])
    offset_ms: int | None = None
    clip_start: float | None = None


def write_subset(
    root: Path, name: str, dataset: str, rows: list[Row], audio_sr: int, seismic_sr: int
) -> None:
    """Write ``signals.h5``, ``manifest.parquet`` and ``info.json`` of one toy subset."""
    directory = root / name
    directory.mkdir(parents=True)
    lengths = {"audio": round(audio_sr * CLIP_SECONDS), "seismic": round(seismic_sr * CLIP_SECONDS)}
    with h5py.File(directory / "signals.h5", "w") as handle:
        for modality, sign in (("audio", 1.0), ("seismic", -1.0)):
            values = np.arange(len(rows))[:, None] * 1000 + np.arange(lengths[modality])
            handle.create_dataset(f"signals/{modality}", data=(sign * values).astype(np.float32))
    n = len(rows)
    columns = {
        "sample_index": list(range(n)),
        "sample_id": [r.sample_id for r in rows],
        "dataset": [dataset] * n,
        "subset": [name] * n,
        "representation": ["waveform"] * n,
        "labels": [r.labels for r in rows],
        "distances_m": [r.distances for r in rows],
        "name_labels": [None] * n,
        "node": [r.node for r in rows],
        "run": [r.run for r in rows],
        "clip": [r.clip for r in rows],
        "recording": [r.recording for r in rows],
        "scene": [r.scene for r in rows],
        "offset_ms": [r.offset_ms for r in rows],
        "node_xy": [None] * n,
        "vehicle_xy": [None] * n,
        "clip_start": [r.clip_start for r in rows],
        "audio_sr": [float(audio_sr)] * n,
        "seismic_sr": [float(seismic_sr)] * n,
        "source_file": [f"{name}/{r.sample_id}.pt" for r in rows],
        "source_sha256": ["0" * 64] * n,
        "source_bytes": [0] * n,
        "original_meta": ["{}"] * n,
    }
    pq.write_table(pa.table(columns, schema=MANIFEST_SCHEMA), directory / "manifest.parquet")
    info = {
        "representation": "waveform",
        "audio_sr": float(audio_sr),
        "seismic_sr": float(seismic_sr),
        "columns": {f"signals/{m}": {"row_shape": [n]} for m, n in lengths.items()},
    }
    (directory / "info.json").write_text(json.dumps(info))


def plain_rows() -> list[Row]:
    """Rows of Toy/plain, see the module docstring."""
    rows: list[Row] = []
    layout = [
        (0, "a", "n1", range(10)),
        (0, "a", "n2", range(10)),
        (1, "a", "n1", range(10)),
        (2, "b", "n1", [0, 1, 2, 4, 5, 6]),
        (3, "b", "n1", range(6)),
    ]
    for run, label, node, clips in layout:
        for clip in clips:
            sample_id = f"run{run}_{node}_{clip}"
            distance = float(10 * run + clip)
            rows.append(Row(sample_id, [label], clip, node, run, distances=[distance]))
    return rows


def named_rows() -> list[Row]:
    """Rows of Toy/named, see the module docstring."""
    rows: list[Row] = []
    for clip in range(4):
        start = 100.0 + CLIP_SECONDS * clip
        rows.append(Row(f"rec_x_n1_{clip}", ["Warhog"], clip, recording="rec_x", clip_start=start))
        rows.append(Row(f"rec_x_n1_{clip}_dwt", ["Warhog"], clip, recording="rec_x"))
        rows.append(
            Row(f"rec_x_o1000_n1_{clip}", ["Warhog"], clip, recording="rec_x", offset_ms=1000)
        )
    return rows


@pytest.fixture
def toy_root(tmp_path: Path) -> Path:
    """A unified root with the subsets Toy/plain and Toy/named."""
    root = tmp_path / "unified"
    write_subset(root, "Toy/plain", "Toy", plain_rows(), audio_sr=20, seismic_sr=10)
    write_subset(root, "Toy/named", "Named", named_rows(), audio_sr=40, seismic_sr=10)
    return root
