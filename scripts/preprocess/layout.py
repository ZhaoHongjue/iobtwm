"""On-disk layout of the unified dataset, shared by conversion and verification.

```
<out>/
  catalog.json                      # every subset, every excluded folder, split files
  <source folder>/                  # mirrors datasets/<source folder>
    signals.h5                      # signals/<modality>: [N, ...]; extras/<path>: [M, ...]
    manifest.parquet                # one row per sample, row i <-> signals[i]
    info.json                       # subset-level facts (rates, shapes, counts, ...)
  splits/<index file path>.parquet  # original index files resolved to sample ids
```
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.preprocess.codec import Json
from scripts.preprocess.specs import Normalized

FORMAT_VERSION: Final = 1
SIGNALS_FILE: Final = "signals.h5"
MANIFEST_FILE: Final = "manifest.parquet"
INFO_FILE: Final = "info.json"
CATALOG_FILE: Final = "catalog.json"
SPLITS_DIR: Final = "splits"
SIGNAL_PREFIX: Final = "signals/"
EXTRA_PREFIX: Final = "extras/"

# Fields have heterogeneous DataType parameters, hence Field[Any].
_MANIFEST_FIELDS: Final[list[pa.Field[Any]]] = [
    pa.field("sample_index", pa.int64(), nullable=False),
    pa.field("sample_id", pa.string(), nullable=False),
    pa.field("dataset", pa.string(), nullable=False),
    pa.field("subset", pa.string(), nullable=False),
    pa.field("representation", pa.string(), nullable=False),
    pa.field("labels", pa.list_(pa.string()), nullable=False),
    pa.field("distances_m", pa.list_(pa.float64()), nullable=False),
    pa.field("name_labels", pa.list_(pa.string())),
    pa.field("node", pa.string()),
    pa.field("run", pa.int64()),
    pa.field("clip", pa.int64()),
    pa.field("recording", pa.string()),
    pa.field("scene", pa.string()),
    pa.field("offset_ms", pa.int64()),
    pa.field("node_xy", pa.list_(pa.float64())),
    pa.field("vehicle_xy", pa.list_(pa.float64())),
    pa.field("clip_start", pa.float64()),
    pa.field("audio_sr", pa.float64(), nullable=False),
    pa.field("seismic_sr", pa.float64(), nullable=False),
    pa.field("source_file", pa.string(), nullable=False),
    pa.field("source_sha256", pa.string(), nullable=False),
    pa.field("source_bytes", pa.int64(), nullable=False),
    pa.field("original_meta", pa.string(), nullable=False),
]
MANIFEST_SCHEMA: Final = pa.schema(_MANIFEST_FIELDS)
NORMALIZED_COLUMNS: Final = tuple(Normalized.__dataclass_fields__)

_SPLIT_FIELDS: Final[list[pa.Field[Any]]] = [
    pa.field("line", pa.int64(), nullable=False),
    pa.field("original_path", pa.string(), nullable=False),
    pa.field("sample_id", pa.string()),
]
SPLIT_SCHEMA: Final = pa.schema(_SPLIT_FIELDS)


@dataclass(frozen=True, slots=True)
class ManifestRow:
    """One manifest row before it is written to Parquet."""

    sample_index: int
    sample_id: str
    dataset: str
    subset: str
    representation: str
    normalized: Normalized
    audio_sr: float
    seismic_sr: float
    source_file: str
    source_sha256: str
    source_bytes: int
    original_meta: str


def normalized_record(n: Normalized) -> dict[str, object]:
    """Normalized fields as Parquet-ready values (tuples become lists)."""
    return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(n).items()}


def write_manifest(rows: Sequence[ManifestRow], path: Path) -> None:
    """Write manifest rows to ``path`` with the fixed schema."""
    records = []
    for r in rows:
        rec = {k: getattr(r, k) for k in MANIFEST_SCHEMA.names if k not in NORMALIZED_COLUMNS}
        records.append(rec | normalized_record(r.normalized))
    table = pa.Table.from_pylist(records, schema=MANIFEST_SCHEMA)
    pq.write_table(table, path, compression="zstd")


def dump_meta(meta: Json) -> str:
    """Serialize tagged metadata (NaN/inf are kept; they round-trip through ``json``)."""
    return json.dumps(meta, separators=(",", ":"), allow_nan=True)


def write_json(obj: object, path: Path) -> None:
    """Write pretty JSON."""
    path.write_text(json.dumps(obj, indent=2, allow_nan=True) + "\n")


def natural_key(name: str) -> tuple[tuple[int, int | str], ...]:
    """Sort key that orders embedded integers numerically (clip 2 before clip 10)."""
    parts = re.split(r"(-?\d+)", name)
    return tuple((0, int(p)) if re.fullmatch(r"-?\d+", p) else (1, p) for p in parts if p)


def list_samples(folder: Path) -> list[Path]:
    """All ``.pt`` files directly in ``folder``, in natural order."""
    return sorted(
        (p for p in folder.iterdir() if p.suffix == ".pt"), key=lambda p: natural_key(p.name)
    )


def discover_sample_folders(src: Path) -> list[str]:
    """Folders under ``src`` (relative, POSIX) that directly contain ``.pt`` files."""
    folders = {p.parent.relative_to(src).as_posix() for p in src.rglob("*.pt")}
    return sorted(folders)


def sample_id(folder: str, filename: str) -> str:
    """Stable id of a sample: its source path without the ``.pt`` suffix."""
    return f"{folder}/{filename.removesuffix('.pt')}"
