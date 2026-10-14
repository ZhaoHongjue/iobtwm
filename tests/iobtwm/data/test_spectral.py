from __future__ import annotations

import pytest
import torch

from iobtwm.data.config import Modality, Representation, SpectralConfig
from iobtwm.data.spectral import SpectralTransform

RATES: dict[Modality, int] = {"audio": 1600, "seismic": 100}


@pytest.mark.parametrize(
    ("representation", "channels"), [("magnitude", 1), ("log_magnitude", 1), ("complex", 2)]
)
def test_shapes(representation: Representation, channels: int) -> None:
    transform = SpectralTransform(SpectralConfig(representation=representation), RATES)
    batch = {"audio": torch.randn(3, 3200), "seismic": torch.randn(3, 200), "labels": torch.ones(3)}

    out = transform(batch)

    assert out["audio"].shape == (3, channels, 161, 19)
    assert out["seismic"].shape == (3, channels, 11, 19)
    assert out["audio"].shape[1:] == transform.output_shape("audio", 3200)
    assert out["labels"] is batch["labels"]


def test_unbatched_input() -> None:
    transform = SpectralTransform(SpectralConfig(), RATES)

    assert transform({"seismic": torch.randn(200)})["seismic"].shape == (1, 11, 19)


def test_waveform_passes_through() -> None:
    transform = SpectralTransform(SpectralConfig(representation="waveform"), RATES)
    wave = torch.randn(2, 200)

    assert transform({"seismic": wave})["seismic"] is wave
    assert transform.output_shape("seismic", 200) == (200,)


def test_rect_complex_matches_per_interval_fft() -> None:
    config = SpectralConfig(
        representation="complex", window="rect", window_seconds=0.2, hop_seconds=0.2
    )
    wave = torch.randn(2, 3200)

    out = SpectralTransform(config, RATES)({"audio": wave})["audio"]

    expected = torch.fft.rfft(wave.reshape(2, 10, 320), dim=-1).transpose(1, 2)
    torch.testing.assert_close(out[:, 0], expected.real, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out[:, 1], expected.imag, atol=1e-4, rtol=1e-4)


def test_log_magnitude_is_log_of_magnitude() -> None:
    wave = {"audio": torch.randn(2, 3200)}
    magnitude = SpectralTransform(SpectralConfig(representation="magnitude"), RATES)(wave)
    log = SpectralTransform(SpectralConfig(representation="log_magnitude", log_eps=1e-3), RATES)(
        wave
    )

    torch.testing.assert_close(log["audio"], torch.log(magnitude["audio"] + 1e-3))


def test_rejects_fractional_hop() -> None:
    with pytest.raises(ValueError, match="hop"):
        SpectralTransform(SpectralConfig(hop_seconds=0.125), RATES)
