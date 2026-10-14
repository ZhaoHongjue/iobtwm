"""Tests for the folder registry, filename parsing and metadata normalization."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.preprocess import specs
from scripts.preprocess.specs import Normalized


def test_registry_is_consistent() -> None:
    folders = specs.spec_by_folder()  # raises on duplicates
    assert not set(folders) & set(specs.EXCLUDED)
    assert set(specs.DERIVED_FROM) == set(specs.EXCLUDED)
    assert set(specs.DERIVED_FROM.values()) <= set(folders)


CASES = [
    (
        "Parkland/individual_time_samples",
        "Polaris0150pm_rs1_12_dwt.pt",
        {"label": torch.tensor(0), "data": {}},
        Normalized(("polaris",), (None,), ("polaris",), "rs1", clip=12, recording="Polaris0150pm"),
    ),
    (
        "Parkland/individual_samples",
        "forester_rs1_0.pt",
        {"label": torch.tensor(3), "data": {}},  # legacy id scheme: 3 would map to "motor"
        Normalized(("forester",), (None,), ("forester",), "rs1", clip=0, recording="forester"),
    ),
    (
        "GracesQuarters/day_1_sensor_all",
        "silverado_run2_gq_1_1426.pt",
        {"label": torch.tensor(2), "targets": np.array(["sedan"], dtype=object), "data": {}},
        Normalized(("silverado",), (None,), ("silverado",), "gq-1", run=2, clip=1426),
    ),
    (
        "GracesQuarters/GracesQuartersDay1FilteredNone",
        "husky_run3_gq_1_0.pt",
        {"label": "husky", "data": {}},
        Normalized(("husky",), (None,), ("husky",), "gq-1", run=3, clip=0),
    ),
    (
        "GracesQuarters/2024-07-19-GQ-Background/individual_time_samples",
        "background_run3_gq-2_12_-1.pt",
        {"label": "background", "distance": -1.0, "data": {}},
        Normalized(("background",), (None,), ("background",), "gq-2", run=3, clip=12),
    ),
    (
        "GracesQuarters/2024-08-06-GQ/individual_time_samples",
        "run0_gq-2_0.pt",
        {
            "label": np.array(["warthog", "polaris"], dtype=object),
            "distance": {"warthog": 0.84},
            "data": {},
        },
        Normalized(("warthog", "polaris"), (0.84, None), None, "gq-2", run=0, clip=0),
    ),
    (
        "GracesQuarters/GracesQuartersMixed",
        "silverado_polaris_mixed_3.pt",
        {"label": [torch.tensor(0), torch.tensor(1)], "data": {}},
        Normalized(("silverado", "polaris"), (None, None), ("silverado", "polaris"), clip=3),
    ),
    (
        "ict_pt_files",
        "cx30_gle350_mixed_0.pt",
        {"label": torch.tensor([3, 1]), "data": {}},
        Normalized(("cx30", "gle350"), (None, None), ("cx30", "gle350"), clip=0),
    ),
    (
        "ICT_train_dis/2024-03-29-ICT/individual_time_samples",
        "cx30_run6_rs10_138_36.pt",
        {"label": "cx30", "distance": np.float64(36.55), "data": {}},
        Normalized(("cx30",), (36.55,), ("cx30",), "rs10", run=6, clip=138),
    ),
    (
        "2026-04-14-West-Point/individual_time_samples",
        "run1_rs1_410__-1.pt",
        {"label": {"kona"}, "distance": {"kona": float("nan")}, "data": {}},
        Normalized(("kona",), (None,), None, "rs1", run=1, clip=410),
    ),
    (
        "M3N-VC/i22/individual_time_samples",
        "i22_cx30-miata_run4_o1000_rs10_00007.pt",
        {
            "data": {"rs10": {}},
            "label": ["cx30", "miata"],
            "distance": float("inf"),
            "node_xy": [1.0, 2.0],
            "vehicle_xy": None,
            "clip_start": 1.5,
            "scene": "i22",
            "run": 4,
            "clip": 7,
        },
        Normalized(
            ("cx30", "miata"),
            (None, None),
            ("cx30", "miata"),
            "rs10",
            run=4,
            clip=7,
            scene="i22",
            offset_ms=1000,
            node_xy=(1.0, 2.0),
            clip_start=1.5,
        ),
    ),
]


@pytest.mark.parametrize(("folder", "filename", "sample", "expected"), CASES)
def test_normalize(folder: str, filename: str, sample: object, expected: Normalized) -> None:
    spec = specs.spec_by_folder()[folder]
    assert specs.normalize(spec, filename, sample) == expected


def test_normalize_rejects_unknown_filename() -> None:
    spec = specs.spec_by_folder()["ict_pt_files"]
    with pytest.raises(ValueError, match="does not match"):
        specs.normalize(spec, "something_else.pt", {"label": "x"})
