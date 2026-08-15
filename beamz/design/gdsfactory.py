"""Transparent GDSFactory-to-BeamZ simulation preparation.

This module is deliberately part of :mod:`beamz.design`: it converts a
GDSFactory component into ordinary BeamZ geometry and devices without adding a
solver-integration dependency.  The returned setup exposes the native
:class:`~beamz.design.core.Design`, :class:`~beamz.devices.ports.Port`, and
:class:`~beamz.simulation.api.Simulation` values so applications can inspect or
modify every numerical choice before execution.

Install the optional layout dependency with ``pip install 'beamz[gds]'``.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np

from beamz.const import LIGHT_SPEED, µm
from beamz.design.gds import ImportedComponent, import_component
from beamz.design.grid_spec import GridSpec
from beamz.design.materials import Material
from beamz.devices.boundaries import PML
from beamz.devices.modes import ModeSpec
from beamz.devices.ports import Port
from beamz.simulation import Simulation

Layer = tuple[int, int]


def _positive(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


@dataclass(frozen=True, slots=True)
class Settings:
    """Numerical policy for a GDSFactory component simulation.

    All lengths are in metres and wavelengths are vacuum wavelengths in metres.
    The policy intentionally exposes domain padding, PML, grid, and run duration;
    :func:`prepare` does not silently select these values from a PDK.
    """

    layer: Layer = (1, 0)
    wavelengths: tuple[float, float] = (1.50 * µm, 1.60 * µm)
    wavelength_points: int = 21
    n_core: float = 3.47
    n_clad: float = 1.44
    core_thickness: float = 0.22 * µm
    clad_below: float = 1.0 * µm
    clad_above: float = 1.0 * µm
    xy_padding: float = 3.0 * µm
    z_padding: float = 1.0 * µm
    pml_thickness: float = 1.0 * µm
    port_overlap: float = 0.0
    run_time: float = 2.0e-12
    grid_spec: GridSpec | None = None
    mode_spec: ModeSpec = field(
        default_factory=lambda: ModeSpec(num_modes=1, mode_index=0, polarization="te")
    )

    def __post_init__(self) -> None:
        layer = tuple(int(value) for value in self.layer)
        if len(layer) != 2:
            raise ValueError("layer must be a (layer, datatype) pair.")
        wavelengths = tuple(float(value) for value in self.wavelengths)
        if len(wavelengths) != 2:
            raise ValueError("wavelengths must contain exactly (start, stop).")
        if not all(np.isfinite(value) and value > 0.0 for value in wavelengths):
            raise ValueError("wavelengths must be finite and positive.")
        if wavelengths[0] >= wavelengths[1]:
            raise ValueError("wavelengths must be strictly increasing.")
        if int(self.wavelength_points) < 1:
            raise ValueError("wavelength_points must be at least one.")
        for name in (
            "n_core",
            "n_clad",
            "core_thickness",
            "clad_below",
            "clad_above",
            "xy_padding",
            "z_padding",
            "pml_thickness",
            "run_time",
        ):
            _positive(getattr(self, name), name=name)
        if float(self.port_overlap) < 0.0 or not np.isfinite(float(self.port_overlap)):
            raise ValueError("port_overlap must be finite and non-negative.")
        if self.grid_spec is not None and not isinstance(self.grid_spec, GridSpec):
            raise TypeError("grid_spec must be a GridSpec or None.")
        if not isinstance(self.mode_spec, ModeSpec):
            raise TypeError("mode_spec must be a ModeSpec.")
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "wavelengths", wavelengths)
        object.__setattr__(self, "wavelength_points", int(self.wavelength_points))

    @property
    def frequencies(self) -> np.ndarray:
        """Return ascending frequency samples shared by mode monitors."""
        wl = np.linspace(*self.wavelengths, self.wavelength_points, dtype=float)
        return np.sort(LIGHT_SPEED / wl)

    @property
    def center_frequency(self) -> float:
        """Return the arithmetic center frequency in hertz."""
        return float(np.mean(self.frequencies))

    @property
    def frequency_width(self) -> float:
        """Return a Gaussian bandwidth that covers the requested samples."""
        freqs = self.frequencies
        return max(float(freqs[-1] - freqs[0]), 0.05 * self.center_frequency)

    def resolved_grid_spec(self) -> GridSpec:
        """Return the requested grid or a wavelength-aware default policy."""
        return self.grid_spec or GridSpec.auto(
            wavelength=float(np.mean(self.wavelengths)), min_steps_per_wvl=12.0
        )

    def updated_copy(self, **changes: Any) -> Settings:
        """Return a new validated settings value with selected fields replaced."""
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class PortMetadata:
    """Trace one native BeamZ port back to a GDSFactory port datum.

    ``outward_direction`` is the direction from the component to the external
    circuit; ``port.direction`` is the opposite, inward direction used to launch
    a BeamZ mode.  Sources and monitors share ``datum_center`` in the first
    release, while the appended waveguide segment keeps that plane clear of PML.
    """

    name: str
    datum_center: tuple[float, float, float]
    outward_direction: str
    inward_direction: str
    axis: str
    size: tuple[float, float, float]


def _opposite(direction: str) -> str:
    return ("-" if direction.startswith("+") else "+") + direction[1:]


def _port_metadata(ports: Sequence[Port]) -> tuple[PortMetadata, ...]:
    return tuple(
        PortMetadata(
            name=port.name,
            datum_center=port.center,
            outward_direction=_opposite(port.signed_direction),
            inward_direction=port.signed_direction,
            axis=port.axis,
            size=port.size,
        )
        for port in ports
    )


@dataclass(frozen=True, slots=True)
class PreparedComponent:
    """An inspectable GDSFactory component translated to native BeamZ values.

    The object owns no runtime state.  :meth:`simulation_for` returns a new
    immutable :class:`~beamz.Simulation` for a selected source port and leaves
    this setup unchanged.
    """

    imported: ImportedComponent
    settings: Settings
    component: Any = field(repr=False, compare=False)
    layer_stack: Any | None = field(default=None, repr=False, compare=False)
    material_map: Mapping[str, Material | float] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def design(self):
        """Return the native BeamZ geometry used by every derived simulation."""
        return self.imported.design

    @property
    def ports(self) -> tuple[Port, ...]:
        """Return native modal ports at the named GDSFactory port datums."""
        return self.imported.ports

    @property
    def port_metadata(self) -> tuple[PortMetadata, ...]:
        """Return the documented source/monitor and port-direction convention."""
        return _port_metadata(self.ports)

    @property
    def component_name(self) -> str:
        """Return the resolved GDSFactory component name."""
        return self.imported.component_name

    def port(self, name: str) -> Port:
        """Return a native port by its stable GDSFactory name."""
        return self.imported.port(name)

    def simulation_template(self) -> Simulation:
        """Build a source-free native BeamZ simulation with all modal monitors."""
        monitors = tuple(port.to_monitor(self.settings.frequencies) for port in self.ports)
        return Simulation(
            design=self.design,
            monitors=monitors,
            boundaries=(PML(thickness=self.settings.pml_thickness, formulation="cpml"),),
            grid_spec=self.settings.resolved_grid_spec(),
            run_time=self.settings.run_time,
            normalize_source=0,
        )

    def simulation_for(self, source_port: str) -> Simulation:
        """Return a native BeamZ simulation excited through ``source_port``.

        The source and every modal monitor are ordinary BeamZ device objects.
        Call ``updated_copy`` on the returned simulation to alter any numerical
        control before execution.
        """
        port = self.port(source_port)
        source = port.to_source(
            self.settings.center_frequency,
            self.settings.frequency_width,
            source_time=None,
        )
        return self.simulation_template().updated_copy(sources=(source,))

    def estimate_resources(self, source_port: str | None = None) -> dict[str, Any]:
        """Return a pre-run cell and memory estimate.

        The lightweight estimate is available before native raster/compile support
        is loaded.  ``compiled_estimate`` is included when the optional native
        raster extension is available and offers BeamZ's more detailed allocation
        report.
        """
        port = self.ports[0].name if source_port is None else source_port
        simulation = self.simulation_for(port)
        shape = tuple(int(value) for value in simulation.grid.shape)
        cells = int(np.prod(shape))
        # Six E/H fields plus common material and boundary work arrays.  This is a
        # deliberately conservative planning figure, not a backend allocation claim.
        estimate: dict[str, Any] = {
            "grid_shape": shape,
            "grid_cells": cells,
            "estimated_memory_bytes": cells * 12 * np.dtype(np.float32).itemsize,
            "estimated_memory_gb": cells * 12 * np.dtype(np.float32).itemsize / 1e9,
            "n_simulations": len(self.ports),
            "compiled_estimate": None,
        }
        # A source checkout without the maturin extension can still display a
        # useful setup preview; built wheels provide the detailed estimate.
        with suppress(ImportError):
            estimate["compiled_estimate"] = simulation.memory_estimate()
        return estimate

    def preview(self, source_port: str | None = None, **kwargs: Any):
        """Plot geometry, PML, source, and modal-monitor placement before a run."""
        port = self.ports[0].name if source_port is None else source_port
        return self.simulation_for(port).plot(**kwargs)


def prepare(
    component: Any = "mmi1x2",
    *,
    layer_stack: Any | None = None,
    settings: Settings | None = None,
    material_map: Mapping[str, Material | float] | None = None,
    component_settings: Mapping[str, Any] | None = None,
) -> PreparedComponent:
    """Prepare a GDSFactory component for explicit, local BeamZ simulation.

    The function converts component geometry, adds port extensions to the domain
    boundary, and returns native BeamZ geometry and ports.  It does not compile or
    execute a simulation.  Pass an explicit ``layer_stack`` and ``material_map``
    for PDK-aware vertical geometry; otherwise the settings' simple core/cladding
    model is used.
    """
    resolved = Settings() if settings is None else settings
    if not isinstance(resolved, Settings):
        raise TypeError("settings must be beamz.design.gdsfactory.Settings or None.")
    imported = import_component(
        component,
        layer=resolved.layer,
        n_core=resolved.n_core,
        n_clad=resolved.n_clad,
        core_thickness=resolved.core_thickness,
        clad_below=resolved.clad_below,
        clad_above=resolved.clad_above,
        xy_padding=resolved.xy_padding,
        z_padding=resolved.z_padding,
        extend_ports=True,
        port_overlap=resolved.port_overlap,
        settings=component_settings,
        layer_stack=layer_stack,
        material_map=material_map,
    )
    if not imported.ports:
        raise ValueError("GDSFactory component has no optical ports to simulate.")
    ports = tuple(port.updated_copy(mode_spec=resolved.mode_spec) for port in imported.ports)
    imported = ImportedComponent(
        design=imported.design,
        ports=ports,
        component_name=imported.component_name,
        world_origin=imported.world_origin,
    )
    return PreparedComponent(
        imported=imported,
        settings=resolved,
        component=component,
        layer_stack=layer_stack,
        material_map=material_map,
    )


prepare_component = prepare


__all__ = [
    "PortMetadata",
    "PreparedComponent",
    "Settings",
    "prepare",
    "prepare_component",
]
