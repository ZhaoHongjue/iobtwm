"""Convert every registered source folder into the unified HDF5 + Parquet format.

Sources are only read, never written. Each subset is written to ``<folder>.partial`` and
renamed into place when complete, so an interrupted run never leaves a half subset that looks
finished; rerunning skips completed subsets (``--overwrite`` redoes them).

Usage:
    uv run python -m scripts.preprocess.convert --src datasets --out datasets_unified --workers 32
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import logging
import multiprocessing
import shutil
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Final

import h5py
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from scripts.preprocess import codec
from scripts.preprocess.layout import (
    CATALOG_FILE,
    EXTRA_PREFIX,
    FORMAT_VERSION,
    INFO_FILE,
    MANIFEST_FILE,
    SIGNAL_PREFIX,
    SIGNALS_FILE,
    SPLIT_SCHEMA,
    SPLITS_DIR,
    ManifestRow,
    discover_sample_folders,
    dump_meta,
    list_samples,
    sample_id,
    write_json,
    write_manifest,
)
from scripts.preprocess.specs import (
    EXCLUDED,
    SUBSETS,
    Normalized,
    SubsetSpec,
    normalize,
    spec_by_folder,
)

log = logging.getLogger(__name__)

CHUNK_BYTES: Final = 256 * 1024
FLUSH_ROWS: Final = 512
MAX_EXAMPLES: Final = 10


# ---------------------------------------------------------------------------- per-file work


@dataclass(frozen=True, slots=True)
class FileResult:
    """Everything the writer needs from one source file."""

    filename: str
    sha256: str
    size: int
    normalized: Normalized
    meta: codec.Json
    arrays: tuple[tuple[str, npt.NDArray[np.generic]], ...]


def read_source(path: Path) -> tuple[object, str, int]:
    """Load a source sample, hashing exactly the bytes that are deserialized.

    Returns:
        The loaded object, the SHA-256 of the file, and its size in bytes.
    """
    raw = path.read_bytes()
    # weights_only=False: samples contain sets, numpy object arrays and numpy scalars, which
    # the safe unpickler rejects. These are the project's own data files.
    obj = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    return obj, hashlib.sha256(raw).hexdigest(), len(raw)


def storage_key(path: codec.KeyPath, _arr: npt.NDArray[np.generic]) -> str | None:
    """Decide where each numeric array of a sample is stored.

    ``data/<node>/<modality>`` -> ``signals/<modality>`` (node goes to the manifest);
    ``wavelet/...`` -> ``extras/wavelet/...``; anything else (labels, ...) stays inline.
    """
    if path and path[0] == "data":
        if len(path) != 3:
            raise ValueError(f"unexpected array location {'/'.join(path)}")
        return SIGNAL_PREFIX + path[2]
    if path and path[0] == "wavelet":
        return EXTRA_PREFIX + "/".join(path)
    return None


def to_storage(spec: SubsetSpec, key: str, arr: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    """Storage layout of an array: waveform signals are flattened, everything else kept as-is.

    Flattening ``[1, 10, 1600]`` -> ``[16000]`` is a pure reshape in C order (the 10
    intervals are consecutive in time), and the original shape is recorded in the metadata.
    """
    if key.startswith(SIGNAL_PREFIX) and spec.representation == "waveform":
        return np.ascontiguousarray(arr).reshape(-1)
    return np.ascontiguousarray(arr)


def encode_file(src: str, folder: str, filename: str) -> FileResult:
    """Read, normalize and encode one source file (runs in worker processes)."""
    spec = spec_by_folder()[folder]
    obj, digest, size = read_source(Path(src) / folder / filename)
    normalized = normalize(spec, filename, obj)
    assert isinstance(obj, dict)  # normalize() raised otherwise
    data = obj.get("data")
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError(f"{folder}/{filename}: expected exactly one node under 'data'")
    enc = codec.encode(obj, storage_key)
    arrays = tuple((a.key, to_storage(spec, a.key, a.array)) for a in enc.arrays)
    return FileResult(filename, digest, size, normalized, enc.meta, arrays)


def _encode_file_args(args: tuple[str, str, str]) -> FileResult:
    return encode_file(*args)


def _init_worker() -> None:
    torch.set_num_threads(1)


def map_ordered[T, R](fn: Callable[[T], R], items: Sequence[T], workers: int) -> Iterator[R]:
    """Map in order, in-process when ``workers == 0``, else over a bounded process pool."""
    if workers == 0:
        yield from map(fn, items)
        return
    ctx = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker) as pool:
        yield from pool.map(fn, items, chunksize=8, buffersize=workers * 4)


# ---------------------------------------------------------------------------- HDF5 writer


@dataclass
class _Column:
    dataset: h5py.Dataset
    shape: tuple[int, ...]
    dtype: np.dtype[np.generic]
    rows: int = 0
    flushed: int = 0
    buffer: list[npt.NDArray[np.generic]] = field(default_factory=list)


class SignalWriter:
    """Writes signal rows (exactly one per sample) and extra rows into one HDF5 file."""

    def __init__(self, path: Path, n: int, attrs: dict[str, object]) -> None:
        self._file = h5py.File(path, "w")
        self._n = n
        self._cols: dict[str, _Column] = {}
        for k, v in attrs.items():
            self._file.attrs[k] = v

    def _column(self, key: str, arr: npt.NDArray[np.generic]) -> _Column:
        col = self._cols.get(key)
        if col is None:
            chunk_rows = max(1, CHUNK_BYTES // max(1, arr.nbytes))
            signal = key.startswith(SIGNAL_PREFIX)
            ds = self._file.create_dataset(
                key,
                shape=(self._n if signal else 0, *arr.shape),
                maxshape=(self._n if signal else None, *arr.shape),
                dtype=arr.dtype,
                chunks=(max(1, min(chunk_rows, self._n)), *arr.shape),
            )
            col = self._cols[key] = _Column(ds, arr.shape, arr.dtype)
        if arr.shape != col.shape or arr.dtype != col.dtype:
            raise ValueError(f"{key}: row {arr.dtype}{arr.shape} != column {col.dtype}{col.shape}")
        return col

    def append(self, key: str, arr: npt.NDArray[np.generic]) -> int:
        """Append one row to ``key`` and return its row index."""
        col = self._column(key, arr)
        col.buffer.append(arr)
        col.rows += 1
        if len(col.buffer) >= FLUSH_ROWS:
            self._flush(col)
        return col.rows - 1

    def _flush(self, col: _Column) -> None:
        if not col.buffer:
            return
        ds = col.dataset
        end = col.flushed + len(col.buffer)
        if ds.shape[0] < end:
            ds.resize(end, axis=0)
        ds[col.flushed : end] = np.stack(col.buffer)
        col.flushed = end
        col.buffer.clear()

    def close(self) -> dict[str, dict[str, object]]:
        """Flush everything, check every signal has one row per sample, describe columns."""
        described: dict[str, dict[str, object]] = {}
        for key, col in self._cols.items():
            self._flush(col)
            if key.startswith(SIGNAL_PREFIX) and col.rows != self._n:
                raise ValueError(f"{key}: {col.rows} rows for {self._n} samples")
            described[key] = {
                "rows": col.rows,
                "row_shape": list(col.shape),
                "dtype": col.dtype.str,
            }
        self._file.close()
        return described


# ---------------------------------------------------------------------------- subsets


@dataclass
class _Tally:
    labels: collections.Counter[str] = field(default_factory=collections.Counter)
    nodes: collections.Counter[str] = field(default_factory=collections.Counter)
    name_label_mismatches: int = 0
    mismatch_examples: list[str] = field(default_factory=list)

    def add(self, filename: str, n: Normalized) -> None:
        self.labels["+".join(n.labels)] += 1
        self.nodes[n.node or "?"] += 1
        if n.name_labels is not None and sorted(n.name_labels) != sorted(n.labels):
            self.name_label_mismatches += 1
            if len(self.mismatch_examples) < MAX_EXAMPLES:
                self.mismatch_examples.append(f"{filename}: name {n.name_labels} vs {n.labels}")


def _write_rows(
    spec: SubsetSpec, results: Iterable[FileResult], writer: SignalWriter, tally: _Tally
) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    signal_keys: set[str] | None = None
    for i, res in enumerate(results):
        keys = {k for k, _ in res.arrays if k.startswith(SIGNAL_PREFIX)}
        if signal_keys is None:
            signal_keys = keys
        elif keys != signal_keys:
            raise ValueError(f"{spec.folder}/{res.filename}: signals {keys} != {signal_keys}")
        placed = {key: writer.append(key, arr) for key, arr in res.arrays}
        if any(placed[k] != i for k in keys):
            raise AssertionError("signal rows out of step with sample index")
        meta = codec.assign_rows(res.meta, placed)
        tally.add(res.filename, res.normalized)
        rows.append(
            ManifestRow(
                sample_index=i,
                sample_id=sample_id(spec.folder, res.filename),
                dataset=spec.dataset,
                subset=spec.folder,
                representation=spec.representation,
                normalized=res.normalized,
                audio_sr=spec.audio_sr,
                seismic_sr=spec.seismic_sr,
                source_file=f"{spec.folder}/{res.filename}",
                source_sha256=res.sha256,
                source_bytes=res.size,
                original_meta=dump_meta(meta),
            )
        )
        if (i + 1) % 10000 == 0:
            log.info("  %s: %d samples", spec.folder, i + 1)
    return rows


def convert_subset(src: Path, out: Path, spec: SubsetSpec, workers: int, overwrite: bool) -> bool:
    """Convert one source folder. Returns ``False`` if it was already done and skipped."""
    final = out / spec.folder
    if (final / INFO_FILE).exists() and not overwrite:
        log.info("skip %s (already converted)", spec.folder)
        return False
    partial = final.with_name(final.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)

    files = list_samples(src / spec.folder)
    log.info("convert %s: %d files", spec.folder, len(files))
    attrs: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "subset": spec.folder,
        "representation": spec.representation,
        "audio_sr": spec.audio_sr,
        "seismic_sr": spec.seismic_sr,
    }
    writer = SignalWriter(partial / SIGNALS_FILE, len(files), attrs)
    tally = _Tally()
    tasks = [(str(src), spec.folder, f.name) for f in files]
    rows = _write_rows(spec, map_ordered(_encode_file_args, tasks, workers), writer, tally)
    columns = writer.close()
    write_manifest(rows, partial / MANIFEST_FILE)
    info = {
        "format_version": FORMAT_VERSION,
        "subset": spec.folder,
        "dataset": spec.dataset,
        "family": spec.family.name,
        "representation": spec.representation,
        "num_samples": len(rows),
        "audio_sr": spec.audio_sr,
        "seismic_sr": spec.seismic_sr,
        "note": spec.note,
        "columns": columns,
        "source_bytes": sum(r.source_bytes for r in rows),
        "label_counts": dict(tally.labels.most_common()),
        "node_counts": dict(tally.nodes.most_common()),
        "name_label_mismatches": tally.name_label_mismatches,
        "name_label_mismatch_examples": tally.mismatch_examples,
    }
    write_json(info, partial / INFO_FILE)
    if final.exists():
        shutil.rmtree(final)
    partial.rename(final)
    if tally.name_label_mismatches:
        log.warning(
            "%s: %d samples whose filename disagrees with the stored label (see info.json)",
            spec.folder,
            tally.name_label_mismatches,
        )
    return True


# ---------------------------------------------------------------------------- splits


def split_lookup(src: Path, folders: Iterable[str]) -> dict[tuple[str, str, str], list[str]]:
    """(top-level dir, parent dir name, filename) -> sample ids, over converted folders."""
    lookup: dict[tuple[str, str, str], list[str]] = collections.defaultdict(list)
    for folder in folders:
        top, parent = PurePosixPath(folder).parts[0], PurePosixPath(folder).name
        for f in list_samples(src / folder):
            lookup[(top, parent, f.name)].append(sample_id(folder, f.name))
    return lookup


def split_files(src: Path) -> list[Path]:
    """Index files (``*.txt``) under ``src``."""
    return sorted(src.rglob("*.txt"))


def resolve_split_line(
    top: str, line: str, lookup: dict[tuple[str, str, str], list[str]]
) -> str | None:
    """Map one index line (a path on some other machine) to a sample id, if unambiguous."""
    p = PurePosixPath(line)
    ids = lookup.get((top, p.parent.name, p.name), [])
    return ids[0] if len(ids) == 1 else None


def convert_splits(src: Path, out: Path, folders: Iterable[str]) -> list[dict[str, object]]:
    """Rewrite every index file as a Parquet table of (line, original_path, sample_id)."""
    lookup = split_lookup(src, folders)
    summary: list[dict[str, object]] = []
    for txt in split_files(src):
        rel = txt.relative_to(src)
        lines = [(i, s.strip()) for i, s in enumerate(txt.read_text().splitlines(), 1) if s.strip()]
        ids = [resolve_split_line(rel.parts[0], s, lookup) for _, s in lines]
        table = pa.Table.from_pylist(
            [
                {"line": i, "original_path": s, "sample_id": sid}
                for (i, s), sid in zip(lines, ids, strict=True)
            ],
            schema=SPLIT_SCHEMA,
        )
        dest = out / SPLITS_DIR / rel.with_suffix(".parquet")
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dest)
        resolved = sum(sid is not None for sid in ids)
        summary.append({"index_file": rel.as_posix(), "lines": len(lines), "resolved": resolved})
        if resolved < len(lines):
            log.warning("%s: %d/%d lines not found locally", rel, len(lines) - resolved, len(lines))
    return summary


# ---------------------------------------------------------------------------- catalog / main


def check_registry(src: Path, specs: Sequence[SubsetSpec]) -> list[str]:
    """Problems with the registry vs. what is on disk (unknown or missing folders)."""
    discovered = set(discover_sample_folders(src))
    known = {s.folder for s in specs}
    problems = [
        f"unregistered folder with .pt files: {f}"
        for f in sorted(discovered - known - set(EXCLUDED))
    ]
    problems += [f"registered folder missing on disk: {f}" for f in sorted(known - discovered)]
    return problems


def other_files(src: Path) -> dict[str, int]:
    """Counts of files that are neither samples nor index files, by top dir and suffix."""
    counts: collections.Counter[str] = collections.Counter()
    for p in src.rglob("*"):
        if p.is_file() and p.suffix not in (".pt", ".txt"):
            counts[f"{p.relative_to(src).parts[0]}/*{p.suffix}"] += 1
    return dict(sorted(counts.items()))


def write_catalog(src: Path, out: Path, splits: list[dict[str, object]]) -> None:
    """Write ``catalog.json`` describing every converted subset found under ``out``."""
    subsets = []
    for spec in SUBSETS:
        info = out / spec.folder / INFO_FILE
        if info.exists():
            subsets.append(spec.folder)
    write_json(
        {
            "format_version": FORMAT_VERSION,
            "source_root": str(src.resolve()),
            "subsets": subsets,
            "excluded": dict(EXCLUDED),
            "not_converted_files": other_files(src),
            "splits": splits,
        },
        out / CATALOG_FILE,
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src", type=Path, default=Path("datasets"), help="source datasets root")
    p.add_argument("--out", type=Path, default=Path("datasets_unified"), help="output root")
    p.add_argument("--workers", type=int, default=32, help="loader processes (0 = in-process)")
    p.add_argument("--only", default="*", help="glob on subset folder, e.g. 'M3N-VC/*'")
    p.add_argument("--overwrite", action="store_true", help="redo already converted subsets")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(sys.argv[1:] if argv is None else argv)
    src, out = args.src, args.out
    problems = check_registry(src, SUBSETS)
    if problems:
        for msg in problems:
            log.error(msg)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    for spec in SUBSETS:
        if fnmatch(spec.folder, args.only):
            convert_subset(src, out, spec, args.workers, args.overwrite)
    converted = [s.folder for s in SUBSETS if (out / s.folder / INFO_FILE).exists()]
    write_catalog(src, out, convert_splits(src, out, converted))
    log.info("done: %d subsets in %s", len(converted), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
