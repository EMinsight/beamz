from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beamz.const import LIGHT_SPEED
from beamz.simulation.boundaries import PML, Boundary


@dataclass(frozen=True)
class GridSpec:
    """Resolution policy for constructing a simulation grid."""

    min_steps_per_wvl: float = 10.0
    wavelength: float | None = None
    resolution: float | None = None
    courant: float = 0.99

    @classmethod
    def auto(
        cls,
        *,
        min_steps_per_wvl: float = 10.0,
        wavelength: float | None = None,
        courant: float = 0.99,
    ) -> "GridSpec":
        return cls(
            min_steps_per_wvl=float(min_steps_per_wvl),
            wavelength=wavelength,
            courant=float(courant),
        )

    @classmethod
    def uniform(cls, resolution: float, *, courant: float = 0.99) -> "GridSpec":
        return cls(resolution=float(resolution), courant=float(courant))

    def resolve_resolution(self, *, max_index: float = 1.0) -> float:
        if self.resolution is not None:
            return float(self.resolution)
        if self.wavelength is None:
            raise ValueError(
                "GridSpec.auto requires wavelength when resolution is absent."
            )
        n_max = max(float(max_index), 1.0)
        return float(self.wavelength) / (n_max * float(self.min_steps_per_wvl))

    def resolve_time_step(self, resolution: float, *, dims: int) -> float:
        dim_count = max(1, int(dims))
        return (
            float(self.courant)
            * float(resolution)
            / (LIGHT_SPEED * np.sqrt(float(dim_count)))
        )


@dataclass(frozen=True)
class GaussianPulse:
    """Gaussian carrier-envelope source-time specification."""

    freq0: float
    fwidth: float
    amplitude: float = 1.0
    offset: float = 4.0
    remove_dc_component: bool = True

    def _time_width(self) -> float:
        return 1.0 / (2.0 * np.pi * max(float(self.fwidth), 1e-30))

    def spectrum(self, freqs, *, normalize: bool = False) -> np.ndarray:
        """Return the analytic positive-frequency source spectrum."""
        freq_arr = np.asarray(freqs, dtype=float)
        fwidth = max(float(self.fwidth), 1e-30)
        width = self._time_width()
        peak = float(self.offset) / fwidth
        df = freq_arr - float(self.freq0)
        spectrum = (
            float(self.amplitude)
            * width
            * np.sqrt(2.0 * np.pi)
            * np.exp(-0.5 * (df / fwidth) ** 2)
            * np.exp(1j * 2.0 * np.pi * df * peak)
        )
        if normalize:
            center = float(self.amplitude) * width * np.sqrt(2.0 * np.pi)
            spectrum = spectrum / max(abs(center), 1e-300)
        return np.asarray(spectrum, dtype=np.complex128)

    def dft_normalization_spectrum(self, freqs) -> np.ndarray:
        """Return the source spectrum in BeamZ's native monitor normalization."""
        return self.spectrum(freqs, normalize=True) / (2.0 * np.pi)

    def sample(self, time) -> tuple[np.ndarray, np.ndarray]:
        t = np.asarray(time, dtype=float)
        width = self._time_width()
        peak = float(self.offset) / max(float(self.fwidth), 1e-30)
        envelope = float(self.amplitude) * np.exp(-((t - peak) ** 2) / (2.0 * width**2))
        phase = 2.0 * np.pi * float(self.freq0) * t
        signal = envelope * np.cos(phase)
        quadrature = envelope * np.sin(phase)
        if self.remove_dc_component and signal.size:
            signal = signal - np.mean(signal)
        return signal.astype(np.float32), quadrature.astype(np.float32)


@dataclass(frozen=True)
class ModeSpec:
    """Mode-solver selection options."""

    num_modes: int = 1
    mode_index: int = 0
    polarization: str | None = None
    target_neff: float | None = None
    num_freqs: int | None = None


@dataclass(frozen=True)
class BoundarySpec:
    """Convenience container for simulation boundary construction."""

    boundaries: tuple[Boundary, ...]

    @classmethod
    def all_sides(cls, boundary: Boundary | None = None) -> "BoundarySpec":
        return cls((boundary if boundary is not None else PML(edges="all"),))


inf = np.inf
