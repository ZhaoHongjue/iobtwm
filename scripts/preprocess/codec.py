"""Lossless codec between original sample objects and (JSON metadata, external arrays).

An original sample is whatever ``torch.save`` wrote: a nest of dicts, lists, sets, tensors,
numpy arrays and scalars. :func:`encode` replaces selected large arrays with references and
turns everything else into tagged JSON. :func:`decode` inverts it exactly. :func:`diff` compares
two objects strictly (Python types, dtypes, shapes and raw bytes), and is what verification
relies on to prove that nothing but the storage format changed.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import torch

TAG: Final = "__t"
MAX_DIFFS: Final = 20

type Json = bool | int | float | str | list[Json] | dict[str, Json] | None
type KeyPath = tuple[str, ...]
type Extractor = Callable[[KeyPath, npt.NDArray[np.generic]], str | None]
type Fetcher = Callable[[str, int], npt.NDArray[np.generic]]


@dataclass(frozen=True, slots=True)
class ExternalArray:
    """An array pulled out of a sample to be stored outside the JSON metadata.

    Attributes:
        key: Storage key (e.g. ``signals/audio``), unique within one sample.
        array: The array exactly as found in the sample (original dtype and shape).
    """

    key: str
    array: npt.NDArray[np.generic]


@dataclass(frozen=True, slots=True)
class Encoded:
    """Result of :func:`encode`.

    Attributes:
        meta: Tagged JSON tree; external arrays appear as ``ref`` nodes with ``row = -1``.
        arrays: Arrays referenced from ``meta``, in encounter order.
    """

    meta: Json
    arrays: tuple[ExternalArray, ...]


def _torch_dtype_name(t: torch.Tensor) -> str:
    return str(t.dtype).removeprefix("torch.")


def _torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown torch dtype {name!r}")
    return dtype


def _ref(key: str, kind: str, dtype: str, shape: tuple[int, ...]) -> dict[str, Json]:
    return {TAG: "ref", "key": key, "row": -1, "kind": kind, "dtype": dtype, "shape": list(shape)}


def _encode_numeric_ndarray(
    x: npt.NDArray[np.generic], path: KeyPath, extract: Extractor, out: list[ExternalArray]
) -> Json:
    key = extract(path, x)
    if key is not None:
        if any(a.key == key for a in out):
            raise ValueError(f"storage key {key!r} used twice in one sample (at {path})")
        out.append(ExternalArray(key, x))
        return _ref(key, "ndarray", x.dtype.str, x.shape)
    data: Json = x.reshape(-1).tolist()
    return {TAG: "ndarray", "dtype": x.dtype.str, "shape": list(x.shape), "data": data}


def _encode_tensor(
    x: torch.Tensor, path: KeyPath, extract: Extractor, out: list[ExternalArray]
) -> Json:
    if x.requires_grad or x.device.type != "cpu":
        raise ValueError(f"unexpected tensor state at {path}: requires_grad/device")
    key = extract(path, x.numpy())
    if key is not None:
        if any(a.key == key for a in out):
            raise ValueError(f"storage key {key!r} used twice in one sample (at {path})")
        out.append(ExternalArray(key, x.numpy()))
        return _ref(key, "tensor", _torch_dtype_name(x), tuple(x.shape))
    data: Json = x.reshape(-1).tolist()
    return {TAG: "tensor", "dtype": _torch_dtype_name(x), "shape": list(x.shape), "data": data}


def _encode(x: object, path: KeyPath, extract: Extractor, out: list[ExternalArray]) -> Json:
    # Exact type checks: np.float64 subclasses float and np.str_ subclasses str, and both must
    # come back as numpy scalars, not as plain Python values.
    if x is None or type(x) in (bool, int, float, str):
        assert x is None or isinstance(x, bool | int | float | str)
        return x
    if isinstance(x, np.generic):
        value: Json = x.item()
        return {TAG: "np.scalar", "dtype": x.dtype.str, "v": value}
    if isinstance(x, torch.Tensor):
        return _encode_tensor(x, path, extract, out)
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            items = [_encode(v, (*path, str(i)), extract, out) for i, v in enumerate(x.flat)]
            return {TAG: "ndarray_object", "shape": list(x.shape), "items": items}
        return _encode_numeric_ndarray(x, path, extract, out)
    if type(x) is dict:
        result: dict[str, Json] = {}
        for k, v in x.items():
            if type(k) is not str or k == TAG:
                raise TypeError(f"unsupported dict key {k!r} at {path}")
            result[k] = _encode(v, (*path, k), extract, out)
        return result
    if type(x) is list:
        return [_encode(v, (*path, str(i)), extract, out) for i, v in enumerate(x)]
    if type(x) is tuple:
        return {
            TAG: "tuple",
            "items": [_encode(v, (*path, str(i)), extract, out) for i, v in enumerate(x)],
        }
    if type(x) in (set, frozenset):
        assert isinstance(x, set | frozenset)
        items = sorted((_encode(v, (*path, "{}"), extract, out) for v in x), key=json.dumps)
        return {TAG: type(x).__name__, "items": items}
    raise TypeError(f"unsupported type {type(x).__name__} at {path}")


def encode(sample: object, extract: Extractor) -> Encoded:
    """Encode a sample into tagged JSON plus external arrays.

    Args:
        sample: The object returned by ``torch.load``.
        extract: Called for every numeric array with its path; returns the storage key if the
            array should be stored externally, or ``None`` to inline it in the JSON.

    Returns:
        The encoded sample. External array refs have ``row = -1`` until :func:`assign_rows`.

    Raises:
        TypeError: If the sample contains a type the codec cannot represent exactly.
    """
    out: list[ExternalArray] = []
    meta = _encode(sample, (), extract, out)
    return Encoded(meta, tuple(out))


def assign_rows(meta: Json, rows: Mapping[str, int]) -> Json:
    """Return a copy of ``meta`` with each ref's ``row`` taken from ``rows`` (by storage key)."""
    if isinstance(meta, list):
        return [assign_rows(v, rows) for v in meta]
    if isinstance(meta, dict):
        if meta.get(TAG) == "ref":
            key = meta["key"]
            assert isinstance(key, str)
            return {**meta, "row": rows[key]}
        return {k: assign_rows(v, rows) for k, v in meta.items()}
    return meta


def iter_refs(meta: Json) -> list[dict[str, Json]]:
    """Return every ref node in ``meta``."""
    if isinstance(meta, list):
        return [r for v in meta for r in iter_refs(v)]
    if isinstance(meta, dict):
        if meta.get(TAG) == "ref":
            return [meta]
        return [r for v in meta.values() for r in iter_refs(v)]
    return []


def _as_int_list(v: Json) -> list[int]:
    assert isinstance(v, list) and all(type(i) is int for i in v)
    return [i for i in v if isinstance(i, int)]


def _decode_ref(node: Mapping[str, Json], fetch: Fetcher) -> object:
    key, row, kind, dtype = node["key"], node["row"], node["kind"], node["dtype"]
    assert isinstance(key, str) and isinstance(row, int) and isinstance(dtype, str)
    shape = tuple(_as_int_list(node["shape"]))
    stored = fetch(key, row)
    if kind == "tensor":
        expected = torch.empty(0, dtype=_torch_dtype(dtype)).numpy().dtype
    else:
        expected = np.dtype(dtype)
    if stored.dtype != expected:
        raise ValueError(f"stored dtype {stored.dtype} != recorded {expected} for {key}[{row}]")
    arr = np.array(stored, copy=True).reshape(shape)
    return torch.from_numpy(arr) if kind == "tensor" else arr


def _decode_tagged(node: Mapping[str, Json], fetch: Fetcher) -> object:
    tag = node[TAG]
    if tag == "ref":
        return _decode_ref(node, fetch)
    if tag == "np.scalar":
        dtype = node["dtype"]
        assert isinstance(dtype, str)
        return np.dtype(dtype).type(node["v"])
    if tag == "tensor":
        dtype = node["dtype"]
        assert isinstance(dtype, str)
        flat = torch.tensor(node["data"], dtype=_torch_dtype(dtype))
        return flat.reshape(_as_int_list(node["shape"]))
    if tag == "ndarray":
        dtype = node["dtype"]
        assert isinstance(dtype, str)
        return np.array(node["data"], dtype=np.dtype(dtype)).reshape(_as_int_list(node["shape"]))
    items = node["items"]
    assert isinstance(items, list)
    decoded = [_decode(v, fetch) for v in items]
    if tag == "ndarray_object":
        arr = np.empty(len(decoded), dtype=object)
        arr[:] = decoded
        return arr.reshape(_as_int_list(node["shape"]))
    if tag == "tuple":
        return tuple(decoded)
    if tag == "set":
        return set(decoded)
    if tag == "frozenset":
        return frozenset(decoded)
    raise ValueError(f"unknown tag {tag!r}")


def _decode(node: Json, fetch: Fetcher) -> object:
    if isinstance(node, list):
        return [_decode(v, fetch) for v in node]
    if isinstance(node, dict):
        if TAG in node:
            return _decode_tagged(node, fetch)
        return {k: _decode(v, fetch) for k, v in node.items()}
    return node


def decode(meta: Json, fetch: Fetcher) -> object:
    """Rebuild the original sample from its metadata and stored arrays.

    Args:
        meta: Tagged JSON produced by :func:`encode` (with rows assigned).
        fetch: Returns the stored array for ``(key, row)``; it may be flattened, it is
            reshaped back to the recorded shape. Its dtype must match the recorded dtype.

    Returns:
        An object that :func:`diff` reports as identical to the original.
    """
    return _decode(meta, fetch)


def _float_bits(x: float) -> bytes:
    return struct.pack("<d", x)


def _diff_ndarray(a: npt.NDArray[np.generic], b: npt.NDArray[np.generic], path: str) -> list[str]:
    if a.dtype.str != b.dtype.str or a.shape != b.shape:
        return [f"{path}: array {a.dtype.str}{a.shape} != {b.dtype.str}{b.shape}"]
    if a.dtype == object:
        out: list[str] = []
        for i, (u, v) in enumerate(zip(a.flat, b.flat, strict=True)):
            out += diff(u, v, f"{path}[{i}]")
        return out
    if np.ascontiguousarray(a).tobytes() != np.ascontiguousarray(b).tobytes():
        n = int(
            np.sum(np.ascontiguousarray(a).view(np.uint8) != np.ascontiguousarray(b).view(np.uint8))
        )
        return [f"{path}: array bytes differ ({n} of {a.nbytes} bytes)"]
    return []


def diff(a: object, b: object, path: str = "$") -> list[str]:
    """Strictly compare two objects and describe every difference found.

    Exact Python types must match (``float`` vs ``np.float64`` differ, ``list`` vs ``tuple``
    differ), dict key order must match, arrays must match in dtype, shape and raw bytes (so
    NaN payloads and ``-0.0`` are compared bit for bit).

    Args:
        a: First object.
        b: Second object.
        path: Location prefix used in messages.

    Returns:
        Human-readable differences, at most ``MAX_DIFFS``; empty means identical.
    """
    if type(a) is not type(b):
        return [f"{path}: type {type(a).__name__} != {type(b).__name__}"]
    out: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            return [f"{path}: keys {list(a)} != {list(b)}"]
        for k in a:
            out += diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list | tuple) and isinstance(b, list | tuple):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        for i, (u, v) in enumerate(zip(a, b, strict=True)):
            out += diff(u, v, f"{path}[{i}]")
    elif isinstance(a, set | frozenset):
        if a != b:
            out.append(f"{path}: {a!r} != {b!r}")
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.dtype != b.dtype or a.shape != b.shape or a.requires_grad != b.requires_grad:
            return [f"{path}: tensor {a.dtype}{tuple(a.shape)} != {b.dtype}{tuple(b.shape)}"]
        out += _diff_ndarray(a.numpy(), b.numpy(), path)
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        out += _diff_ndarray(a, b, path)
    elif isinstance(a, np.generic) and isinstance(b, np.generic):
        if a.dtype.str != b.dtype.str or a.tobytes() != b.tobytes():
            out.append(f"{path}: {a!r} != {b!r}")
    elif isinstance(a, float) and isinstance(b, float):
        if _float_bits(a) != _float_bits(b):
            out.append(f"{path}: {a!r} != {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out[:MAX_DIFFS]
