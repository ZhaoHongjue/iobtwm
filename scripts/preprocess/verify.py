"""Verify that the unified dataset holds exactly the source data, changed only in format.

Every check runs against the source files as they are on disk now:

1. Registry: every source folder with ``.pt`` files is converted or explicitly excluded.
2. Coverage: each manifest lists exactly the source files on disk, each once, in order.
3. Integrity: each source file's SHA-256 still equals the hash recorded at conversion.
4. Content: each freshly loaded source sample equals the sample rebuilt from ``signals.h5`` +
   ``original_meta`` under a strict comparison (Python types, dtypes, shapes, raw bytes).
5. Signals, checked directly: the raw bytes of every ``data[node][modality]`` array equal the
   stored row (independent of the metadata decoder used in check 4).
6. Normalized columns equal a fresh normalization of the source sample.
7. HDF5 holds nothing unreferenced: extra arrays are referenced exactly once each.
8. Splits: every non-empty line of every index file is present in order, and each resolved
   sample id points to a file with the same folder and name.

Usage:
    uv run python -m scripts.preprocess.verify --src datasets --out datasets_unified --workers 32

Exit code 0 only if every check passes; the full report goes to ``<out>/verify_report.json``.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Final

import h5py
import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
import torch

from scripts.preprocess import codec
from scripts.preprocess.convert import (
    check_registry,
    map_ordered,
    read_source,
    resolve_split_line,
    split_files,
    split_lookup,
)
from scripts.preprocess.layout import (
    EXTRA_PREFIX,
    INFO_FILE,
    MANIFEST_FILE,
    NORMALIZED_COLUMNS,
    SIGNAL_PREFIX,
    SIGNALS_FILE,
    SPLITS_DIR,
    list_samples,
    normalized_record,
    sample_id,
    write_json,
)
from scripts.preprocess.specs import SUBSETS, SubsetSpec, normalize, spec_by_folder

log = logging.getLogger(__name__)

ROWS_PER_TASK: Final = 256
MAX_FAILURES: Final = 50

type Row = dict[str, object]


@dataclass
class Report:
    """Collected results; ``failures`` must be empty for the data to count as verified."""

    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    subsets: dict[str, dict[str, object]] = field(default_factory=dict)
    splits: list[dict[str, object]] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        """Record a failure (the first ``MAX_FAILURES`` are logged)."""
        if len(self.failures) < MAX_FAILURES:
            log.error(msg)
        self.failures.append(msg)


# ---------------------------------------------------------------------------- per-row checks


@dataclass(frozen=True, slots=True)
class RowTask:
    """A contiguous block of manifest rows to check in one worker call."""

    src: str
    out: str
    folder: str
    rows: tuple[Row, ...]


@dataclass(frozen=True, slots=True)
class RowResult:
    """Outcome of one :class:`RowTask`."""

    checked: int
    failures: tuple[str, ...]
    extra_refs: dict[str, list[int]]


def _check_signals_direct(
    sample: dict[str, object], index: int, h5: h5py.File, where: str
) -> list[str]:
    """Compare raw bytes of every data[node][modality] with the stored row, without codec."""
    data = sample["data"]
    assert isinstance(data, dict)
    failures: list[str] = []
    for node, mods in data.items():
        assert isinstance(mods, dict)
        for mod, value in mods.items():
            src: npt.NDArray[np.generic] = (
                value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            )
            stored: npt.NDArray[np.generic] = h5[SIGNAL_PREFIX + mod][index]
            same = src.dtype == stored.dtype and src.size == stored.size
            if not same or np.ascontiguousarray(src).tobytes() != stored.tobytes():
                failures.append(f"{where}: data.{node}.{mod} bytes differ from stored row")
    return failures


def _check_row(
    spec: SubsetSpec, src: Path, h5: h5py.File, row: Row, refs: dict[str, list[int]]
) -> list[str]:
    source_file, index, meta_text = row["source_file"], row["sample_index"], row["original_meta"]
    assert isinstance(source_file, str) and isinstance(index, int) and isinstance(meta_text, str)
    where = source_file
    sample, digest, size = read_source(src / source_file)
    failures: list[str] = []
    if digest != row["source_sha256"] or size != row["source_bytes"]:
        failures.append(f"{where}: source file changed since conversion (sha256/size)")
    meta: codec.Json = json.loads(meta_text)

    def fetch(key: str, r: int) -> npt.NDArray[np.generic]:
        arr: npt.NDArray[np.generic] = h5[key][r]
        return arr

    rebuilt = codec.decode(meta, fetch)
    failures += [f"{where}: {d}" for d in codec.diff(sample, rebuilt)]
    assert isinstance(sample, dict)
    failures += _check_signals_direct(sample, index, h5, where)
    expected = normalized_record(normalize(spec, PurePosixPath(source_file).name, sample))
    actual = {k: row[k] for k in NORMALIZED_COLUMNS}
    if expected != actual:
        failures.append(f"{where}: normalized columns {actual} != recomputed {expected}")
    for ref in codec.iter_refs(meta):
        key, r = ref["key"], ref["row"]
        assert isinstance(key, str) and isinstance(r, int)
        if key.startswith(SIGNAL_PREFIX) and r != index:
            failures.append(f"{where}: {key} ref row {r} != sample index {index}")
        if key.startswith(EXTRA_PREFIX):
            refs.setdefault(key, []).append(r)
    return failures


def check_rows(task: RowTask) -> RowResult:
    """Run the per-row checks for one block (runs in worker processes)."""
    spec = spec_by_folder()[task.folder]
    refs: dict[str, list[int]] = {}
    failures: list[str] = []
    with h5py.File(Path(task.out) / task.folder / SIGNALS_FILE, "r") as h5:
        for row in task.rows:
            try:
                failures += _check_row(spec, Path(task.src), h5, row, refs)
            except (OSError, ValueError, TypeError, KeyError, AssertionError) as e:
                failures.append(f"{row.get('source_file')}: {type(e).__name__}: {e}")
    return RowResult(len(task.rows), tuple(failures), refs)


# ---------------------------------------------------------------------------- per-subset checks


def _check_manifest_structure(spec: SubsetSpec, src: Path, rows: list[Row], report: Report) -> None:
    on_disk = [f"{spec.folder}/{p.name}" for p in list_samples(src / spec.folder)]
    listed = [r["source_file"] for r in rows]
    if listed != on_disk:
        missing = sorted(set(on_disk) - {str(x) for x in listed})
        extra = sorted({str(x) for x in listed} - set(on_disk))
        report.fail(
            f"{spec.folder}: manifest != files on disk ({len(listed)} vs {len(on_disk)}; "
            f"missing {missing[:5]}, extra {extra[:5]}, or order differs)"
        )
    for i, r in enumerate(rows):
        expected = {
            "sample_index": i,
            "sample_id": sample_id(spec.folder, PurePosixPath(str(r["source_file"])).name),
            "dataset": spec.dataset,
            "subset": spec.folder,
            "representation": spec.representation,
            "audio_sr": spec.audio_sr,
            "seismic_sr": spec.seismic_sr,
        }
        actual = {k: r[k] for k in expected}
        if actual != expected:
            report.fail(f"{spec.folder} row {i}: {actual} != {expected}")
            break


def _check_extras(
    spec: SubsetSpec, h5: h5py.File, refs: dict[str, list[int]], report: Report
) -> None:
    stored: dict[str, int] = {}

    def visit(name: str, obj: object) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        if name.startswith(EXTRA_PREFIX):
            stored[name] = obj.shape[0]
        elif not name.startswith(SIGNAL_PREFIX):
            report.fail(f"{spec.folder}: unexpected dataset {name} in {SIGNALS_FILE}")

    h5.visititems(visit)
    for key in sorted(set(stored) | set(refs)):
        rows = sorted(refs.get(key, []))
        if rows != list(range(stored.get(key, 0))):
            report.fail(f"{spec.folder}: {key} has {stored.get(key, 0)} rows, refs {len(rows)}")


def verify_subset(src: Path, out: Path, spec: SubsetSpec, workers: int, report: Report) -> None:
    """Run checks 2-7 for one subset."""
    folder = out / spec.folder
    if not (folder / INFO_FILE).exists():
        report.fail(f"{spec.folder}: not converted (no {INFO_FILE})")
        return
    rows: list[Row] = pq.read_table(folder / MANIFEST_FILE).to_pylist()
    _check_manifest_structure(spec, src, rows, report)
    with h5py.File(folder / SIGNALS_FILE, "r") as h5:
        if "signals" not in h5:
            report.fail(f"{spec.folder}: no signals in {SIGNALS_FILE}")
            return
        for key in h5["signals"]:
            if h5[f"signals/{key}"].shape[0] != len(rows):
                report.fail(f"{spec.folder}: signals/{key} rows != {len(rows)} samples")
    tasks = [
        RowTask(str(src), str(out), spec.folder, tuple(rows[i : i + ROWS_PER_TASK]))
        for i in range(0, len(rows), ROWS_PER_TASK)
    ]
    refs: dict[str, list[int]] = collections.defaultdict(list)
    checked = 0
    for res in map_ordered(check_rows, tasks, workers):
        checked += res.checked
        for msg in res.failures:
            report.fail(msg)
        for key, r in res.extra_refs.items():
            refs[key] += r
    with h5py.File(folder / SIGNALS_FILE, "r") as h5:
        _check_extras(spec, h5, refs, report)
    report.subsets[spec.folder] = {"samples_checked": checked, "manifest_rows": len(rows)}
    log.info("verified %s: %d samples", spec.folder, checked)


# ---------------------------------------------------------------------------- splits


def verify_splits(src: Path, out: Path, folders: Sequence[str], report: Report) -> None:
    """Check 8: index files vs. their converted Parquet tables."""
    lookup = split_lookup(src, folders)
    expected_files = set()
    for txt in split_files(src):
        rel = txt.relative_to(src)
        dest = out / SPLITS_DIR / rel.with_suffix(".parquet")
        expected_files.add(dest)
        if not dest.exists():
            report.fail(f"split {rel}: missing {dest}")
            continue
        lines = [(i, s.strip()) for i, s in enumerate(txt.read_text().splitlines(), 1) if s.strip()]
        table = pq.read_table(dest).to_pylist()
        if [(r["line"], r["original_path"]) for r in table] != lines:
            report.fail(f"split {rel}: lines differ from the original index file")
            continue
        unresolved = 0
        for r in table:
            sid, line = r["sample_id"], str(r["original_path"])
            if sid != resolve_split_line(rel.parts[0], line, lookup):
                report.fail(f"split {rel} line {r['line']}: {sid} is not the unique match")
            if sid is None:
                unresolved += 1
                continue
            sp, lp = PurePosixPath(str(sid)), PurePosixPath(line)
            if sp.name != lp.stem or sp.parent.name != lp.parent.name:
                report.fail(f"split {rel} line {r['line']}: {sid} does not match {line}")
        report.splits.append(
            {"index_file": rel.as_posix(), "lines": len(lines), "unresolved": unresolved}
        )
    for stray in set((out / SPLITS_DIR).rglob("*.parquet")) - expected_files:
        report.fail(f"split table without a source index file: {stray}")


# ---------------------------------------------------------------------------- main


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src", type=Path, default=Path("datasets"), help="source datasets root")
    p.add_argument("--out", type=Path, default=Path("datasets_unified"), help="unified root")
    p.add_argument("--workers", type=int, default=32, help="checker processes (0 = in-process)")
    p.add_argument("--only", default="*", help="glob on subset folder (skips split checks)")
    return p.parse_args(argv)


def run(
    src: Path, out: Path, workers: int, only: str = "*", specs: Sequence[SubsetSpec] = SUBSETS
) -> Report:
    """Run all checks and return the report.

    Args:
        src: Source datasets root.
        out: Unified dataset root.
        workers: Checker processes (0 = in-process).
        only: Glob on subset folders; anything but ``*`` skips the split checks.
        specs: Registry to check against (tests pass a subset of :data:`SUBSETS`).
    """
    report = Report()
    for msg in check_registry(src, specs):
        report.fail(msg)
    for spec in specs:
        if fnmatch(spec.folder, only):
            verify_subset(src, out, spec, workers, report)
    if only == "*":
        verify_splits(src, out, [s.folder for s in specs], report)
    else:
        report.notes.append(f"--only {only!r}: split files not checked")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(sys.argv[1:] if argv is None else argv)
    report = run(args.src, args.out, args.workers, args.only)
    total = sum(int(str(s["samples_checked"])) for s in report.subsets.values())
    write_json(
        {
            "ok": not report.failures,
            "samples_checked": total,
            "num_failures": len(report.failures),
            "failures": report.failures[:1000],
            "notes": report.notes,
            "subsets": report.subsets,
            "splits": report.splits,
        },
        args.out / "verify_report.json",
    )
    if report.failures:
        log.error("FAILED: %d problems (see verify_report.json)", len(report.failures))
        return 1
    log.info("OK: %d samples in %d subsets identical to source", total, len(report.subsets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
