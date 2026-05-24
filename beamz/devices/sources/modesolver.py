from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from beamz.const import LIGHT_SPEED
from beamz.devices.sources.mode import ModeSource
from beamz.devices.sources.solve import solve_modes
from beamz.simulation.specs import GaussianPulse, ModeSpec


def _plane_axis_and_spans(plane):
    center = getattr(plane, "center", None)
    size = getattr(plane, "size", None)
    if center is None or size is None:
        center = getattr(plane, "position", (0.0, 0.0, 0.0))
        size = getattr(plane, "size", (0.0, 1.0, 1.0))
    if len(center) == 2:
        center = (center[0], center[1], 0.0)
    if len(size) == 2:
        size = (size[0], size[1], 0.0)
    normal_index = int(np.argmin(np.abs(np.asarray(size, dtype=float))))
    axis = ("x", "y", "z")[normal_index]
    transverse = [float(size[idx]) for idx in range(3) if idx != normal_index]
    return axis, tuple(float(v) for v in center), tuple(transverse)


@dataclass(frozen=True)
class ModeData:
    frequencies: np.ndarray
    neffs: np.ndarray
    e_fields: np.ndarray
    h_fields: np.ndarray

    def to_dataframe(self):
        rows = []
        index = []
        for f_idx, freq in enumerate(self.frequencies):
            neff_row = np.atleast_1d(self.neffs[f_idx])
            for mode_index, neff in enumerate(neff_row):
                rows.append(
                    {
                        "n eff": float(np.real(neff)),
                        "k eff": float(max(np.imag(neff), 0.0)),
                    }
                )
                index.append((float(freq), int(mode_index)))
        return pd.DataFrame(
            rows,
            index=pd.MultiIndex.from_tuples(index, names=("f", "mode_index")),
        )


class ModeSolver:
    """Convenience wrapper for solving modes on a simulation plane."""

    def __init__(
        self,
        *,
        simulation,
        plane,
        mode_spec: ModeSpec | None = None,
        freqs,
    ):
        self.simulation = simulation
        self.plane = plane
        self.mode_spec = mode_spec if mode_spec is not None else ModeSpec()
        self.freqs = np.asarray(freqs, dtype=float).reshape(-1)
        if self.freqs.size == 0:
            raise ValueError("ModeSolver requires at least one frequency.")
        self._modes = None

    def solve(self):
        grid = self.simulation.design.rasterize(resolution=self.simulation.resolution)
        eps = np.asarray(grid.permittivity)
        axis, center, _spans = _plane_axis_and_spans(self.plane)
        offset = getattr(self.simulation, "coordinate_offset", (0.0, 0.0, 0.0))
        center = tuple(c + o for c, o in zip(center, offset, strict=True))
        axis_index = {"z": 0, "y": 1, "x": 2}[axis]
        grid_index = int(
            np.clip(
                round(center[{"z": 2, "y": 1, "x": 0}[axis]] / self.simulation.resolution),
                0,
                eps.shape[axis_index] - 1,
            )
        )
        eps_profile = np.take(eps, grid_index, axis=axis_index)
        neffs_by_freq = []
        e_by_freq = []
        h_by_freq = []
        for freq in self.freqs:
            neffs, e_fields, h_fields, _ = solve_modes(
                eps=eps_profile,
                omega=2.0 * np.pi * float(freq),
                dL=self.simulation.resolution,
                m=int(self.mode_spec.num_modes),
                direction=f"-{axis}",
                filter_pol=self.mode_spec.polarization,
                target_neff=self.mode_spec.target_neff,
                return_fields=True,
            )
            neffs_by_freq.append(np.asarray(neffs))
            e_by_freq.append(np.asarray(e_fields))
            h_by_freq.append(np.asarray(h_fields))
        self._modes = ModeData(
            frequencies=self.freqs,
            neffs=np.asarray(neffs_by_freq),
            e_fields=np.asarray(e_by_freq),
            h_fields=np.asarray(h_by_freq),
        )
        return self._modes

    def to_source(self, *, mode_index=0, direction="+", source_time=None):
        if source_time is None:
            freq0 = float(self.freqs[len(self.freqs) // 2])
            source_time = GaussianPulse(freq0=freq0, fwidth=freq0 / 10.0)
        signal, signal_quadrature = source_time.sample(self.simulation.time)
        axis, center, spans = _plane_axis_and_spans(self.plane)
        offset = getattr(self.simulation, "coordinate_offset", (0.0, 0.0, 0.0))
        center = tuple(c + o for c, o in zip(center, offset, strict=True))
        sign = str(direction)[0] if str(direction).startswith(("+", "-")) else "+"
        full_direction = f"{sign}{axis}"
        freq0 = float(getattr(source_time, "freq0", self.freqs[len(self.freqs) // 2]))
        profile_freqs = None
        if getattr(self.mode_spec, "num_freqs", None):
            count = int(self.mode_spec.num_freqs)
            profile_freqs = np.linspace(float(np.min(self.freqs)), float(np.max(self.freqs)), count)
        return ModeSource(
            grid=self.simulation.design.rasterize(resolution=self.simulation.resolution),
            center=center,
            width=float(spans[0]),
            height=float(spans[1]) if len(spans) > 1 else None,
            wavelength=LIGHT_SPEED / freq0,
            pol=getattr(self.mode_spec, "polarization", None) or "te",
            signal=signal,
            signal_quadrature=signal_quadrature,
            profile_frequencies=profile_freqs,
            direction=full_direction,
        )

    def sim_with_source(self, *, mode_index=0, direction="+", source_time=None):
        source = self.to_source(
            mode_index=mode_index,
            direction=direction,
            source_time=source_time,
        )
        return self.simulation.copy(update={"sources": [source]})

    def plot_field_components(self, *_, **kwargs):
        from beamz.visual.mpl import plot_mode_fields

        if "field_names" in kwargs:
            kwargs["components"] = tuple(kwargs.pop("field_names"))
        kwargs.pop("mode_indices", None)
        if "f" in kwargs:
            frequency = float(kwargs.pop("f"))
        else:
            frequency = float(self.freqs[0])
        axis, center, _spans = _plane_axis_and_spans(self.plane)
        if axis != "x":
            raise NotImplementedError("ModeSolver plotting currently supports x-normal planes.")
        offset = getattr(self.simulation, "coordinate_offset", (0.0, 0.0, 0.0))
        plane_x = center[0] + offset[0]
        return plot_mode_fields(
            self.simulation.design.rasterize(resolution=self.simulation.resolution),
            plane_x=plane_x,
            wavelength=LIGHT_SPEED / frequency,
            polarization=getattr(self.mode_spec, "polarization", None),
            num_modes=int(self.mode_spec.num_modes),
            show=kwargs.pop("show", False),
            **kwargs,
        )
