"""End-to-end: convert a small fake source tree, verify it, and check tampering is caught."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from scripts.preprocess import convert, verify
from scripts.preprocess.layout import MANIFEST_FILE, SIGNALS_FILE, SPLITS_DIR
from scripts.preprocess.specs import SubsetSpec, spec_by_folder

PARKLAND = "Parkland/individual_time_samples"
WESTPOINT = "2026-04-14-West-Point/individual_time_samples"
M3N = "M3N-VC/a06/individual_time_samples"
GQ = "GracesQuarters/2024-08-06-GQ/individual_time_samples"
FOLDERS = (PARKLAND, WESTPOINT, M3N, GQ)


def _parkland(i: int, dwt: bool) -> dict[str, object]:
    g = torch.Generator().manual_seed(i)
    audio, seismic = torch.randn(1, 10, 1600, generator=g), torch.randn(1, 10, 20, generator=g)
    sample: dict[str, object] = {
        "label": torch.tensor(4),
        "flag": {"shake": {"audio": True, "seismic": True}},
        "data": {"shake": {"audio": audio, "seismic": seismic}},
    }
    if dwt:
        sample["data"] = {"shake": {"audio": audio.reshape(1, 1, 16000), "seismic": seismic}}
        sample["wavelet"] = {"db4": {"shake": {"audio": torch.randn(1, 1, 14, 16000)}}}
    return sample


def _westpoint(i: int) -> dict[str, object]:
    return {
        "flag": {"shake": {"audio": True, "seismic": True}},
        "label": {"kona"},
        "distance": {"kona": np.float64(10.0 + i) if i else float("nan")},
        "coordinates": {},
        "data": {
            "shake": {
                "seismic": torch.full((1, 10, 20), float(i)),
                "audio": torch.zeros(1, 10, 1600),
            }
        },
    }


def _m3n(i: int) -> dict[str, object]:
    return {
        "data": {
            "rs1": {
                "audio": torch.arange(3200, dtype=torch.int16) + i,
                "seismic": torch.arange(400, dtype=torch.int32),
            }
        },
        "label": "cx30",
        "distance": 9.5,
        "node_xy": [1.0, 2.0],
        "vehicle_xy": [3.0, 4.0],
        "clip_start": 1698125156.0 + 2 * i,
        "scene": "a06",
        "run": 6,
        "clip": i,
    }


def _gq(i: int) -> dict[str, object]:
    return {
        "flag": {"shake": {"audio": True, "seismic": True}},
        "label": np.array(["warthog", "polaris"], dtype=object),
        "distance": {"warthog": 0.5 * i},
        "data": {"shake": {"seismic": torch.ones(1, 10, 20), "audio": torch.ones(1, 10, 1600) * i}},
    }


def _write_sources(src: Path) -> None:
    files: dict[str, dict[str, object]] = {}
    for i in range(3):
        files[f"{PARKLAND}/tesla_rs3_{i}.pt"] = _parkland(i, dwt=False)
        files[f"{PARKLAND}/tesla_rs3_{i}_dwt.pt"] = _parkland(i, dwt=True)
        files[f"{WESTPOINT}/run1_rs1_{i}__{10 + i}.pt"] = _westpoint(i)
        files[f"{M3N}/a06_cx30_run6_o1000_rs1_{i:05d}_9.pt"] = _m3n(i)
        files[f"{GQ}/run0_gq-2_{i}.pt"] = _gq(i)
    for rel, sample in files.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        torch.save(sample, src / rel)
    index = src / "2026-04-14-West-Point/indices/train_index.txt"
    index.parent.mkdir(parents=True)
    index.write_text(
        "/home/x/2026-04-14-West-Point/individual_time_samples/run1_rs1_2__12.pt\n"
        "/home/x/2026-04-14-West-Point/individual_time_samples/run9_rs1_0__1.pt\n"
    )


SPECS: tuple[SubsetSpec, ...] = tuple(spec_by_folder()[f] for f in FOLDERS)


def _convert(src: Path, out: Path) -> None:
    for spec in SPECS:
        convert.convert_subset(src, out, spec, workers=0, overwrite=False)
    convert.convert_splits(src, out, FOLDERS)


@pytest.fixture
def converted(tmp_path: Path) -> tuple[Path, Path]:
    src, out = tmp_path / "src", tmp_path / "out"
    _write_sources(src)
    _convert(src, out)
    return src, out


def _verify(src: Path, out: Path) -> list[str]:
    return verify.run(src, out, workers=0, specs=SPECS).failures


def test_clean_conversion_verifies(converted: tuple[Path, Path]) -> None:
    src, out = converted
    assert _verify(src, out) == []


def test_layout_and_split_resolution(converted: tuple[Path, Path]) -> None:
    _, out = converted
    with h5py.File(out / PARKLAND / SIGNALS_FILE, "r") as h5:
        assert h5["signals/audio"].shape == (6, 16000)
        assert h5["extras/wavelet/db4/shake/audio"].shape == (3, 1, 1, 14, 16000)
    with h5py.File(out / M3N / SIGNALS_FILE, "r") as h5:
        assert h5["signals/audio"].dtype == np.int16
    split = pq.read_table(out / SPLITS_DIR / "2026-04-14-West-Point/indices/train_index.parquet")
    assert split["sample_id"].to_pylist() == [f"{WESTPOINT}/run1_rs1_2__12", None]


def _flip_signal_bit(src: Path, out: Path) -> None:
    with h5py.File(out / GQ / SIGNALS_FILE, "r+") as h5:
        row = h5["signals/audio"][1]
        row.view(np.uint32)[5] ^= 1
        h5["signals/audio"][1] = row


def _change_extra(src: Path, out: Path) -> None:
    with h5py.File(out / PARKLAND / SIGNALS_FILE, "r+") as h5:
        h5["extras/wavelet/db4/shake/audio"][0, 0, 0, 0, 0] += 1


def _edit_meta(src: Path, out: Path) -> None:
    path = out / WESTPOINT / MANIFEST_FILE
    rows = pq.read_table(path).to_pylist()
    meta = json.loads(rows[1]["original_meta"])
    meta["distance"]["kona"]["v"] = 99.0
    rows[1]["original_meta"] = json.dumps(meta)
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(path)), path)


def _edit_normalized(src: Path, out: Path) -> None:
    path = out / M3N / MANIFEST_FILE
    rows = pq.read_table(path).to_pylist()
    rows[0]["labels"] = ["miata"]
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(path)), path)


def _change_source(src: Path, out: Path) -> None:
    torch.save(_m3n(7), src / M3N / "a06_cx30_run6_o1000_rs1_00001_9.pt")


def _delete_source(src: Path, out: Path) -> None:
    (src / GQ / "run0_gq-2_2.pt").unlink()


def _add_unregistered_folder(src: Path, out: Path) -> None:
    (src / "NewDataset").mkdir()
    torch.save(_gq(0), src / "NewDataset/x.pt")


def _edit_split(src: Path, out: Path) -> None:
    path = out / SPLITS_DIR / "2026-04-14-West-Point/indices/train_index.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["sample_id"] = f"{WESTPOINT}/run1_rs1_1__11"
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(path)), path)


@pytest.mark.parametrize(
    "tamper",
    [
        _flip_signal_bit,
        _change_extra,
        _edit_meta,
        _edit_normalized,
        _change_source,
        _delete_source,
        _add_unregistered_folder,
        _edit_split,
    ],
)
def test_tampering_is_detected(
    converted: tuple[Path, Path], tamper: Callable[[Path, Path], None]
) -> None:
    src, out = converted
    tamper(src, out)
    assert _verify(src, out) != []


def test_rerun_skips_completed_subsets(converted: tuple[Path, Path]) -> None:
    src, out = converted
    assert not convert.convert_subset(src, out, SPECS[0], workers=0, overwrite=False)


def test_check_registry_flags_unknown_and_missing(tmp_path: Path) -> None:
    (tmp_path / "Unknown").mkdir()
    torch.save({}, tmp_path / "Unknown/a.pt")
    problems = convert.check_registry(tmp_path, SPECS[:1])
    assert any("unregistered" in p for p in problems)
    assert any("missing" in p for p in problems)
