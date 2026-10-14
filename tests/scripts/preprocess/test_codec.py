"""Tests for the lossless sample codec."""

from __future__ import annotations

import json

import numpy as np
import numpy.typing as npt
import pytest
import torch

from scripts.preprocess import codec


def _extract(path: codec.KeyPath, _arr: npt.NDArray[np.generic]) -> str | None:
    return f"signals/{path[2]}" if path[:1] == ("data",) and len(path) == 3 else None


def _roundtrip(obj: object) -> object:
    enc = codec.encode(obj, _extract)
    store = {a.key: a.array.reshape(-1) for a in enc.arrays}
    meta = codec.assign_rows(enc.meta, dict.fromkeys(store, 0))
    meta = json.loads(json.dumps(meta, allow_nan=True))  # through text, as in the manifest
    return codec.decode(meta, lambda key, _row: store[key])


SAMPLES = {
    "parkland": {
        "label": torch.tensor(4),
        "flag": {"shake": {"audio": True, "seismic": True}},
        "data": {
            "shake": {
                "audio": torch.randn(1, 10, 1600),
                "seismic": torch.randn(1, 10, 20),
            }
        },
    },
    "gq_sensor_all": {
        "flag": {"shake": {"seismic": True, "audio": True}},
        "label": torch.tensor(1),
        "targets": np.array(["warthog", "pedestrian"], dtype=object),
        "type": "single_car",
        "data": {
            "shake": {
                "seismic": np.arange(200, dtype=np.int64),
                "audio": np.arange(16000, dtype=np.int64),
            }
        },
    },
    "westpoint": {
        "label": {"kona"},
        "distance": {"kona": np.float64("nan"), "x": float("inf")},
        "coordinates": {},
        "data": {"shake": {"audio": torch.zeros(1, 10, 1600)}},
    },
    "m3n": {
        "data": {"rs1": {"audio": torch.arange(3200, dtype=torch.int16)}},
        "label": ["cx30", "miata"],
        "distance": float("inf"),
        "node_xy": [1.5, -2.25],
        "vehicle_xy": None,
        "clip_start": 1698125156.0,
        "run": 6,
        "clip": 0,
    },
    "mixed": {
        "label": [torch.tensor(0), torch.tensor(2)],
        "data": {"shake": {"audio": np.zeros((16000, 1))}},
    },
    "odd_scalars": {
        "neg_zero": -0.0,
        "f32": np.float32(1.1),
        "i8": np.int8(-3),
        "s": np.str_("x"),
        "t": (1, "a", None),
        "fs": frozenset({1, 2}),
        "empty": torch.zeros(0),
        "bool_t": torch.tensor([True, False]),
        "big_endian": np.array([1.0, 2.0], dtype=">f8"),
        "data": {"n": {"m": torch.ones(2)}},
    },
}


@pytest.mark.parametrize("name", SAMPLES)
def test_roundtrip_is_identical(name: str) -> None:
    obj = SAMPLES[name]
    assert codec.diff(obj, _roundtrip(obj)) == []


def test_arrays_are_stored_externally_with_original_dtype() -> None:
    enc = codec.encode(SAMPLES["gq_sensor_all"], _extract)
    assert {a.key: a.array.dtype for a in enc.arrays} == {
        "signals/seismic": np.dtype(np.int64),
        "signals/audio": np.dtype(np.int64),
    }


def _flip_bit(t: torch.Tensor) -> torch.Tensor:
    out = t.clone()
    out.view(torch.int32).reshape(-1)[7] ^= 1
    return out


X = torch.arange(10, dtype=torch.float32)
BASE = {"x": X, "f": 1.0, "s": {"a"}, "l": [1, 2]}
CHANGES = {
    "bit_flip": {**BASE, "x": _flip_bit(X)},
    "dtype": {**BASE, "x": X.double()},
    "shape": {**BASE, "x": X.reshape(2, 5)},
    "tensor_vs_ndarray": {**BASE, "x": np.arange(10, dtype=np.float32)},
    "float_vs_np_float": {**BASE, "f": np.float64(1.0)},
    "neg_zero": {**BASE, "f": -0.0},
    "set_vs_list": {**BASE, "s": ["a"]},
    "list_vs_tuple": {**BASE, "l": (1, 2)},
    "key_order": {"f": 1.0, "x": X, "s": {"a"}, "l": [1, 2]},
    "missing_key": {"x": X, "f": 1.0, "s": {"a"}},
    "int_vs_bool": {**BASE, "l": [True, 2]},
}


@pytest.mark.parametrize("name", CHANGES)
def test_diff_detects_change(name: str) -> None:
    assert codec.diff(BASE, CHANGES[name]) != []


def test_diff_treats_identical_nan_as_equal() -> None:
    a = {"v": float("nan"), "t": torch.tensor([float("nan")])}
    b = {"v": float("nan"), "t": torch.tensor([float("nan")])}
    assert codec.diff(a, b) == []


@pytest.mark.parametrize(
    "obj",
    [{1: "int key"}, {codec.TAG: "reserved"}, {"o": object()}, {"c": 1 + 2j}],
)
def test_encode_rejects_unrepresentable(obj: object) -> None:
    with pytest.raises(TypeError):
        codec.encode(obj, _extract)


def test_decode_rejects_dtype_mismatch() -> None:
    enc = codec.encode({"data": {"n": {"m": torch.ones(3)}}}, _extract)
    meta = codec.assign_rows(enc.meta, {"signals/m": 0})
    with pytest.raises(ValueError, match="dtype"):
        codec.decode(meta, lambda _k, _r: np.ones(3, dtype=np.float64))
