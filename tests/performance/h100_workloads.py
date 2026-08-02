"""Canonical accelerator workloads used to compare BeamZ execution backends."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

import beamz as bz
from beamz.design import MaterialGrid


@dataclass(frozen=True, slots=True)
class H100Workload:
    """A shape-stable 3D FDTD workload with an explicit feature envelope."""

    name: str
    shape_zyx: tuple[int, int, int]
    timesteps: int
    resolution: float = 40e-9
    pml_cells: int = 10
    heterogeneous: bool = False
    cpml: bool = False
    source: bool = False
    monitor: bool = False

    def resized(
        self,
        *,
        shape_zyx: tuple[int, int, int] | None = None,
        timesteps: int | None = None,
    ) -> H100Workload:
        """Return the same feature workload with a different problem size."""
        return replace(
            self,
            shape_zyx=self.shape_zyx if shape_zyx is None else shape_zyx,
            timesteps=self.timesteps if timesteps is None else int(timesteps),
        )

    def __post_init__(self) -> None:
        if len(self.shape_zyx) != 3 or any(value <= 2 for value in self.shape_zyx):
            raise ValueError("shape_zyx must contain three sizes greater than two")
        if self.timesteps < 2:
            raise ValueError("timesteps must be at least two")
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if self.cpml and 2 * self.pml_cells >= min(self.shape_zyx):
            raise ValueError("CPML must leave at least one interior cell per axis")

    @property
    def dt(self) -> float:
        # A fixed CFL fraction makes runs comparable across backend implementations.
        return 0.95 * self.resolution / (bz.LIGHT_SPEED * np.sqrt(3.0))

    @property
    def size_xyz(self) -> tuple[float, float, float]:
        nz, ny, nx = self.shape_zyx
        return (
            nx * self.resolution,
            ny * self.resolution,
            nz * self.resolution,
        )

    @property
    def feature_labels(self) -> dict[str, tuple[str, ...]]:
        return {
            "boundaries": ("CPML(all)",) if self.cpml else ("PEC(all)",),
            "sources": ("GaussianSource(Ez)",) if self.source else (),
            "monitors": ("FieldMonitor(plane,DFT)",) if self.monitor else (),
        }

    def build(self) -> bz.Simulation:
        """Build a deterministic simulation without geometry-rasterization noise."""
        nz, ny, nx = self.shape_zyx
        eps = np.ones(self.shape_zyx, dtype=np.float32)
        if self.heterogeneous:
            # A long high-index guide exercises nonuniform material loads throughout
            # the timed region without including CAD rasterization in the measurement.
            z0, z1 = nz * 3 // 8, nz * 5 // 8
            y0, y1 = ny * 3 // 8, ny * 5 // 8
            eps[z0:z1, y0:y1, :] = np.float32(3.45**2)
        material_grid = MaterialGrid(
            permittivity=eps,
            conductivity=np.float32(0.0),
            permeability=np.float32(1.0),
            resolution=self.resolution,
            shape=self.shape_zyx,
        )
        size_x, size_y, size_z = self.size_xyz
        time = np.arange(self.timesteps, dtype=np.float64) * self.dt
        center = (0.5 * size_x, 0.5 * size_y, 0.5 * size_z)

        sources: list[object] = []
        if self.source:
            omega = 2.0 * np.pi * 193.414e12
            envelope = np.exp(-((np.arange(time.size) - 24.0) / 8.0) ** 2)
            sources.append(
                bz.GaussianSource(
                    position=(0.25 * size_x, center[1], center[2]),
                    width=max(2.0 * self.resolution, 0.08 * min(size_y, size_z)),
                    signal=(envelope * np.sin(omega * time)).astype(np.float32),
                )
            )

        monitors: list[object] = []
        if self.monitor:
            clear_y = max(self.resolution, size_y - 2 * self.pml_cells * self.resolution)
            clear_z = max(self.resolution, size_z - 2 * self.pml_cells * self.resolution)
            monitors.append(
                bz.FieldMonitor(
                    center=(0.75 * size_x, center[1], center[2]),
                    size=(0.0, clear_y, clear_z),
                    freqs=np.asarray((190e12, 193.414e12, 196e12)),
                    fields=("Ey", "Ez", "Hy", "Hz"),
                    name="transmission",
                )
            )

        boundaries: list[object] = [bz.PEC(edges="all")]
        if self.cpml:
            boundaries = [
                bz.PML(
                    edges="all",
                    thickness=self.pml_cells * self.resolution,
                    formulation="cpml",
                )
            ]
        return bz.Simulation(
            material_grid=material_grid,
            sources=sources,
            monitors=monitors,
            boundaries=boundaries,
            time=time,
        )


H100_WORKLOADS = {
    "bare_3d": H100Workload(
        name="bare_3d",
        shape_zyx=(128, 256, 384),
        timesteps=500,
    ),
    "realistic_3d": H100Workload(
        name="realistic_3d",
        shape_zyx=(128, 256, 384),
        timesteps=500,
        heterogeneous=True,
        cpml=True,
        source=True,
        monitor=True,
    ),
}
