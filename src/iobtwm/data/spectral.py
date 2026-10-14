"""Turn waveforms into model inputs with a run-time STFT."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from iobtwm.data.config import Modality, SpectralConfig


def _frame_length(seconds: float, rate: int, what: str) -> int:
    samples = seconds * rate
    if not math.isclose(samples, round(samples)) or round(samples) < 1:
        raise ValueError(f"STFT {what} of {seconds} s is not whole samples at {rate} Hz")
    return round(samples)


class SpectralTransform(nn.Module):
    """STFT of every modality, returned as magnitude, log magnitude or real/imag channels.

    Waveforms of shape ``[..., time]`` become ``[..., channels, freq, frames]`` with one channel
    (magnitude, log magnitude) or two (complex: real, imag). With representation ``waveform`` the
    inputs pass through unchanged. Frames do not overlap the window edges (``center=False``), so
    every frame only sees samples inside the example.
    """

    def __init__(self, config: SpectralConfig, rates: Mapping[Modality, int]) -> None:
        """Prepare the STFT window of every modality.

        Raises:
            ValueError: If the frame or hop length is not a whole number of samples.
        """
        super().__init__()
        self.config = config
        self.frames: dict[Modality, tuple[int, int]] = {}
        for modality, rate in rates.items():
            n_fft = _frame_length(config.window_seconds, rate, "window")
            hop = _frame_length(config.hop_seconds, rate, "hop")
            self.frames[modality] = (n_fft, hop)
            if config.window == "hann":
                window = torch.hann_window(n_fft, periodic=True)
            else:
                window = torch.ones(n_fft)
            self.register_buffer(f"window_{modality}", window, persistent=False)

    def output_shape(self, modality: Modality, num_samples: int) -> tuple[int, ...]:
        """Return the per-example output shape for a waveform of ``num_samples`` samples."""
        if self.config.representation == "waveform":
            return (num_samples,)
        n_fft, hop = self.frames[modality]
        channels = 2 if self.config.representation == "complex" else 1
        return (channels, n_fft // 2 + 1, (num_samples - n_fft) // hop + 1)

    def transform(self, modality: Modality, wave: Tensor) -> Tensor:
        """Transform one modality's waveforms of shape ``[..., time]``."""
        if self.config.representation == "waveform":
            return wave
        n_fft, hop = self.frames[modality]
        lead = wave.shape[:-1]
        spec = torch.stft(
            wave.reshape(-1, wave.shape[-1]),
            n_fft=n_fft,
            hop_length=hop,
            window=self.get_buffer(f"window_{modality}"),
            center=False,
            return_complex=True,
        )
        match self.config.representation:
            case "magnitude":
                out = spec.abs().unsqueeze(1)
            case "log_magnitude":
                out = torch.log(spec.abs() + self.config.log_eps).unsqueeze(1)
            case "complex":
                out = torch.stack([spec.real, spec.imag], dim=1)
        return out.reshape(*lead, *out.shape[1:])

    def forward(self, inputs: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Return a copy of ``inputs`` with every modality transformed."""
        out = dict(inputs)
        for modality in self.frames:
            if modality in out:
                out[modality] = self.transform(modality, out[modality])
        return out
