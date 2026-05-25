"""Plain xarray conversion helpers for BeamZ objects.

This module is intentionally factory-only. BeamZ keeps solver/runtime objects as
NumPy/JAX arrays and exposes xarray at the result boundary for labeled slicing,
metadata, and plotting.
"""

from __future__ import annotations

from itertools import product
from types import SimpleNamespace

import numpy as np
import xarray as xr

_FIELD_UNITS = {
    "Ex": "V/m",
    "Ey": "V/m",
    "Ez": "V/m",
    "Hx": "A/m",
    "Hy": "A/m",
    "Hz": "A/m",
    "permittivity": "relative",
    "permeability": "relative",
    "conductivity": "S/m",
}


def _plane_axes(plane_2d):
    plane = str(plane_2d or "xy").lower()
    if plane == "xy":
        return ("y", "x")
    if plane == "yz":
        return ("z", "y")
    if plane == "xz":
        return ("z", "x")
    return ("y", "x")


def _spatial_dims(ndim, *, plane_2d="xy"):
    if ndim == 2:
        return _plane_axes(plane_2d)
    if ndim == 3:
        return ("z", "y", "x")
    if ndim == 1:
        return ("s",)
    return tuple(f"dim_{idx}" for idx in range(ndim))


def _axis_coords(dims, shape, *, resolution=None):
    coords = {}
    spacing = None if resolution is None else float(resolution)
    for dim, length in zip(dims, shape, strict=True):
        if dim in {"x", "y", "z", "s"}:
            values = np.arange(length, dtype=float)
            if spacing is not None:
                values = values * spacing
            coords[dim] = (dim, values, {"units": "m"} if spacing is not None else {})
        else:
            coords[dim] = (dim, np.arange(length, dtype=int))
    return coords


def _grid_shape_from_simulation(simulation, spatial_shape):
    fields = getattr(simulation, "fields", None)
    eps = getattr(fields, "permittivity", None)
    if eps is not None:
        return tuple(int(v) for v in np.asarray(eps).shape)

    resolution = getattr(simulation, "resolution", None)
    design = getattr(simulation, "design", None)
    if resolution is None or design is None:
        return tuple(int(v) for v in spatial_shape)

    dx = float(resolution)
    if len(spatial_shape) == 3:
        return (
            max(1, int(round(float(getattr(design, "depth", 0.0) or 0.0) / dx))),
            max(1, int(round(float(getattr(design, "height", 0.0)) / dx))),
            max(1, int(round(float(getattr(design, "width", 0.0)) / dx))),
        )
    if len(spatial_shape) == 2:
        return (
            max(1, int(round(float(getattr(design, "height", 0.0)) / dx))),
            max(1, int(round(float(getattr(design, "width", 0.0)) / dx))),
        )
    return tuple(int(v) for v in spatial_shape)


def _candidate_grid_shapes(name, spatial_shape, *, simulation=None, plane_2d=None):
    spatial_shape = tuple(int(v) for v in spatial_shape)
    if simulation is not None:
        yield _grid_shape_from_simulation(simulation, spatial_shape)

    ranges = [range(max(1, size - 1), size + 2) for size in spatial_shape]
    if len(spatial_shape) == 3:
        from beamz.simulation.yee import component_shape_3d

        for candidate in product(*ranges):
            try:
                if component_shape_3d(str(name), tuple(candidate)) == spatial_shape:
                    yield tuple(int(v) for v in candidate)
            except Exception:
                continue
    elif len(spatial_shape) == 2:
        from beamz.simulation.yee import component_shape_2d

        plane = str(plane_2d or getattr(simulation, "plane_2d", "xy"))
        for candidate in product(*ranges):
            try:
                if (
                    component_shape_2d(str(name), tuple(candidate), plane)
                    == spatial_shape
                ):
                    yield tuple(int(v) for v in candidate)
            except Exception:
                continue


def _yee_axis_coords(
    name, dims, spatial_shape, *, simulation=None, resolution=None, plane_2d=None
):
    if simulation is None or str(name) not in _FIELD_UNITS:
        return None

    dx = getattr(simulation, "resolution", resolution)
    if dx is None:
        return None

    if len(spatial_shape) not in {2, 3}:
        return None

    for grid_shape in _candidate_grid_shapes(
        name,
        spatial_shape,
        simulation=simulation,
        plane_2d=plane_2d,
    ):
        try:
            if len(spatial_shape) == 3:
                from beamz.simulation.yee import component_coordinates_3d_um

                coords_um = component_coordinates_3d_um(
                    str(name),
                    tuple(int(v) for v in grid_shape),
                    float(dx) * 1e6,
                )
            else:
                from beamz.simulation.yee import component_coordinates_2d_um

                coords_um = component_coordinates_2d_um(
                    str(name),
                    tuple(int(v) for v in grid_shape),
                    float(dx) * 1e6,
                    str(plane_2d or getattr(simulation, "plane_2d", "xy")),
                )
        except Exception:
            continue

        coords = {}
        for dim, length in zip(dims, spatial_shape, strict=True):
            values_um = coords_um.get(dim)
            if values_um is None or len(values_um) != int(length):
                break
            coords[dim] = (
                dim,
                np.asarray(values_um, dtype=float) * 1e-6,
                {"units": "m"},
            )
        else:
            return coords
    return None


def _time_dim_and_coords(nframes, *, times=None, steps=None):
    if times is not None and len(times) == nframes:
        coords = {"t": ("t", np.asarray(times, dtype=float), {"units": "s"})}
        if steps is not None and len(steps) == nframes:
            coords["step"] = ("t", np.asarray(steps, dtype=int))
        return "t", coords

    coords = {"frame": ("frame", np.arange(nframes, dtype=int))}
    if steps is not None and len(steps) == nframes:
        coords["step"] = ("frame", np.asarray(steps, dtype=int))
    return "frame", coords


def _as_monitor_like(obj):
    if hasattr(obj, "_plot_proxy"):
        return obj._plot_proxy()
    return obj


def field_data_array(
    values,
    *,
    name,
    simulation=None,
    design=None,
    resolution=None,
    times=None,
    steps=None,
    plane_2d=None,
    attrs=None,
):
    """Create a labeled field ``DataArray`` from stored simulation data."""
    arr = np.asarray(values)
    if arr.ndim < 2:
        raise ValueError(f"Field '{name}' must be at least 2D, got shape {arr.shape}.")

    sim = simulation
    if sim is not None:
        design = sim.design
        resolution = sim.resolution if resolution is None else resolution
        plane_2d = sim.plane_2d if plane_2d is None else plane_2d

    if arr.ndim in {3, 4}:
        time_dim, coords = _time_dim_and_coords(
            arr.shape[0],
            times=times,
            steps=steps,
        )
        spatial = _spatial_dims(arr.ndim - 1, plane_2d=plane_2d)
        dims = (time_dim, *spatial)
        yee_coords = _yee_axis_coords(
            name,
            spatial,
            arr.shape[1:],
            simulation=sim,
            resolution=resolution,
            plane_2d=plane_2d,
        )
        coords.update(
            yee_coords
            if yee_coords is not None
            else _axis_coords(spatial, arr.shape[1:], resolution=resolution)
        )
    else:
        dims = _spatial_dims(arr.ndim, plane_2d=plane_2d)
        coords = _yee_axis_coords(
            name,
            dims,
            arr.shape,
            simulation=sim,
            resolution=resolution,
            plane_2d=plane_2d,
        )
        if coords is None:
            coords = _axis_coords(dims, arr.shape, resolution=resolution)

    da_attrs = {
        "component": str(name),
        "units": _FIELD_UNITS.get(str(name), ""),
    }
    if design is not None:
        da_attrs.update(
            {
                "design_width": float(getattr(design, "width", np.nan)),
                "design_height": float(getattr(design, "height", np.nan)),
                "design_depth": float(getattr(design, "depth", 0.0) or 0.0),
            }
        )
    if attrs:
        da_attrs.update(attrs)

    return xr.DataArray(arr, dims=dims, coords=coords, name=str(name), attrs=da_attrs)


def _center_coords_for_simulation(simulation, dim):
    resolution = getattr(simulation, "resolution", None)
    design = getattr(simulation, "design", None)
    if resolution is None or design is None:
        return None
    span = {
        "x": getattr(design, "width", None),
        "y": getattr(design, "height", None),
        "z": getattr(design, "depth", None),
    }.get(dim)
    if span is None or not np.isfinite(float(span)) or float(span) <= 0.0:
        return None
    count = max(1, int(round(float(span) / float(resolution))))
    return (np.arange(count, dtype=float) + 0.5) * float(resolution)


def _colocate_field_arrays_for_dataset(simulation, data_vars):
    if len(data_vars) <= 1:
        return data_vars

    targets = {}
    for dim in ("x", "y", "z"):
        arrays = [
            da for da in data_vars.values() if dim in da.dims and dim in da.coords
        ]
        if len(arrays) <= 1:
            continue
        coord_signatures = {
            tuple(np.asarray(da.coords[dim], dtype=float).round(18)) for da in arrays
        }
        size_signatures = {da.sizes[dim] for da in arrays}
        if len(coord_signatures) <= 1 and len(size_signatures) <= 1:
            continue

        target = _center_coords_for_simulation(simulation, dim)
        if target is None:
            continue
        lower = max(
            float(np.nanmin(np.asarray(da.coords[dim], dtype=float))) for da in arrays
        )
        upper = min(
            float(np.nanmax(np.asarray(da.coords[dim], dtype=float))) for da in arrays
        )
        target = target[(target >= lower) & (target <= upper)]
        if target.size:
            targets[dim] = target

    if not targets:
        return data_vars

    colocated = {}
    for name, da in data_vars.items():
        selectors = {dim: values for dim, values in targets.items() if dim in da.dims}
        colocated[name] = da.interp(selectors) if selectors else da
    return colocated


def simulation_fields_dataset(
    simulation,
    fields,
    *,
    field_times=None,
    field_steps=None,
):
    """Create the public xarray Dataset for saved simulation fields."""
    if fields is None:
        data_vars = {}
    elif isinstance(fields, xr.Dataset):
        return fields
    else:
        data_vars = {
            name: field_data_array(
                values,
                name=name,
                simulation=simulation,
                times=field_times,
                steps=field_steps,
            )
            for name, values in fields.items()
        }
        data_vars = _colocate_field_arrays_for_dataset(simulation, data_vars)

    sim = simulation
    attrs = {
        "beamz_kind": "SimulationResults",
        "resolution": float(getattr(sim, "resolution", np.nan)),
        "plane_2d": getattr(sim, "plane_2d", None),
        "design_width": float(getattr(sim.design, "width", np.nan)),
        "design_height": float(getattr(sim.design, "height", np.nan)),
        "design_depth": float(getattr(sim.design, "depth", 0.0) or 0.0),
    }
    return xr.Dataset(data_vars=data_vars, attrs=attrs)


def simulation_dataset(results):
    """Return ``SimulationResults`` saved fields as an xarray Dataset."""
    return simulation_fields_dataset(
        results.simulation,
        results.fields,
        field_times=getattr(results, "field_times", None),
        field_steps=getattr(results, "field_steps", None),
    )


def _monitor_field_data_array(monitor, component, values, times):
    arr = np.asarray(values)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr[:, None]

    if times is not None and len(times) == arr.shape[0]:
        dims = ("t", *_spatial_dims(arr.ndim - 1))
        coords = {"t": ("t", np.asarray(times, dtype=float), {"units": "s"})}
    else:
        dims = ("frame", *_spatial_dims(arr.ndim - 1))
        coords = {"frame": ("frame", np.arange(arr.shape[0], dtype=int))}

    spatial = dims[1:]
    coords.update(_monitor_spatial_coords(monitor, spatial, arr.shape[1:]))
    attrs = {
        "component": str(component),
        "units": _FIELD_UNITS.get(str(component), ""),
        "monitor_type": getattr(monitor, "monitor_type", None),
        "monitor_name": getattr(monitor, "name", None),
    }
    return xr.DataArray(arr, dims=dims, coords=coords, name=str(component), attrs=attrs)


def _line_monitor_s_coords(monitor, length):
    start = getattr(monitor, "start", None)
    end = getattr(monitor, "end", None)
    if start is None or end is None:
        return np.arange(int(length), dtype=float)
    start_arr = np.asarray(start, dtype=float)
    end_arr = np.asarray(end, dtype=float)
    if start_arr.shape != end_arr.shape:
        return np.arange(int(length), dtype=float)
    if int(length) <= 1:
        return np.asarray([0.0], dtype=float)
    return np.linspace(0.0, float(np.linalg.norm(end_arr - start_arr)), int(length))


def _monitor_spatial_coords(monitor, dims, shape):
    if dims == ("s",):
        return {
            "s": (
                "s",
                _line_monitor_s_coords(monitor, shape[0]),
                {"units": "m", "long_name": "distance along monitor"},
            )
        }

    resolution = getattr(monitor, "_resolution", None)
    if getattr(monitor, "is_3d", False) and resolution is not None:
        try:
            field_shape = getattr(monitor, "_field_shape", None)
            if field_shape is None:
                field_shape = tuple(max(int(v), 1) for v in shape)
            coords0, coords1 = monitor.get_analysis_plane_coords_3d(
                dx=float(resolution),
                dy=float(resolution),
                dz=float(resolution),
                field_shape=tuple(int(v) for v in field_shape),
            )
            values_by_dim = {
                dim: values
                for dim, values in zip(dims, (coords0, coords1), strict=False)
            }
            if all(
                len(values_by_dim.get(dim, ())) == size
                for dim, size in zip(dims, shape, strict=True)
            ):
                return {
                    dim: (
                        dim,
                        np.asarray(values_by_dim[dim], dtype=float),
                        {"units": "m"},
                    )
                    for dim in dims
                }
        except Exception:
            pass

    return _axis_coords(dims, shape, resolution=resolution)


def _monitor_dft_data_array(monitor, component):
    if not hasattr(monitor, "get_dft_component"):
        return None
    try:
        values = np.asarray(monitor.get_dft_component(component), dtype=np.complex128)
        freqs = np.asarray(monitor.get_dft_frequencies(), dtype=float)
    except Exception:
        return None
    if values.size == 0 or values.ndim != 2 or freqs.size != values.shape[0]:
        return None
    coords = {
        "f": ("f", freqs, {"units": "Hz"}),
        "s": ("s", _line_monitor_s_coords(monitor, values.shape[1]), {"units": "m"}),
    }
    return xr.DataArray(
        values,
        dims=("f", "s"),
        coords=coords,
        name=f"dft_{component}",
        attrs={
            "component": str(component),
            "units": _FIELD_UNITS.get(str(component), ""),
            "monitor_type": getattr(monitor, "monitor_type", None),
            "monitor_name": getattr(monitor, "name", None),
            "domain": "frequency",
        },
    )


def monitor_dataset(monitor_or_result):
    """Convert a ``Monitor`` or ``MonitorResults`` object to an xarray Dataset."""
    source_monitor = getattr(monitor_or_result, "monitor", monitor_or_result)
    monitor = _as_monitor_like(monitor_or_result)
    fields = getattr(monitor, "fields", {}) or {}
    times = np.asarray(fields.get("t", ()), dtype=float)
    times = times if times.size else None

    data_vars = {}
    for component, values in fields.items():
        if component == "t":
            continue
        data = _monitor_field_data_array(monitor, component, values, times)
        if data is not None:
            data_vars[component] = data

    power_history = np.asarray(getattr(monitor, "power_history", ()), dtype=float)
    if power_history.size:
        power_t = np.asarray(getattr(monitor, "power_timestamps", ()), dtype=float)
        if (
            power_t.size == power_history.size
            and times is not None
            and times.size == power_t.size
            and np.allclose(times, power_t)
        ):
            coords = {"t": ("t", power_t, {"units": "s"})}
            dims = ("t",)
        elif power_t.size == power_history.size:
            coords = {"power_t": ("power_t", power_t, {"units": "s"})}
            dims = ("power_t",)
        else:
            coords = {"power_index": ("power_index", np.arange(power_history.size))}
            dims = ("power_index",)
        data_vars["power"] = xr.DataArray(
            power_history,
            dims=dims,
            coords=coords,
            attrs={"units": "a.u.", "monitor_name": getattr(monitor, "name", None)},
        )

    power_spectrum = np.asarray(getattr(monitor, "power_spectrum", ()))
    power_freqs = np.asarray(getattr(monitor, "power_spectrum_frequencies", ()))
    if power_spectrum.size and power_freqs.size == power_spectrum.size:
        data_vars["power_spectrum"] = xr.DataArray(
            power_spectrum,
            dims=("f",),
            coords={"f": ("f", power_freqs, {"units": "Hz"})},
            attrs={"monitor_name": getattr(monitor, "name", None)},
        )

    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        data = _monitor_dft_data_array(source_monitor, component)
        if data is not None:
            data_vars[f"dft_{component}"] = data

    if source_monitor is not monitor_or_result:
        frequency_flux = np.asarray(
            getattr(monitor_or_result, "frequency_flux_spectrum", ())
        )
        dft_freqs = np.asarray(
            getattr(source_monitor, "dft_frequencies", ()), dtype=float
        )
        if frequency_flux.size and dft_freqs.size == frequency_flux.size:
            data_vars["frequency_flux_spectrum"] = xr.DataArray(
                frequency_flux,
                dims=("f",),
                coords={"f": ("f", dft_freqs, {"units": "Hz"})},
                attrs={"monitor_name": getattr(monitor, "name", None)},
            )

    return xr.Dataset(
        data_vars=data_vars,
        attrs={
            "beamz_kind": type(monitor_or_result).__name__,
            "monitor_type": getattr(monitor, "monitor_type", None),
            "monitor_name": getattr(monitor, "name", None),
        },
    )


def source_signal_data_array(source, *, t=None):
    """Convert a source signal to an xarray ``DataArray``."""
    signal = getattr(source, "signal", None)
    if signal is None:
        raise RuntimeError(f"{type(source).__name__} has no signal attribute.")

    if callable(signal):
        if t is None:
            raise ValueError("t must be provided for callable source signals.")
        time = np.asarray(t, dtype=float)
        values = np.asarray([signal(float(value)) for value in time])
        dims = ("t",)
        coords = {"t": ("t", time, {"units": "s"})}
    else:
        values = np.asarray(signal)
        if t is not None:
            time = np.asarray(t, dtype=float)
            if time.shape[0] != values.shape[0]:
                raise ValueError(
                    "t and source signal must have the same length: "
                    f"{time.shape[0]} != {values.shape[0]}"
                )
            dims = ("t",)
            coords = {"t": ("t", time, {"units": "s"})}
        else:
            dims = ("sample",)
            coords = {"sample": ("sample", np.arange(values.shape[0], dtype=int))}

    return xr.DataArray(
        values,
        dims=dims,
        coords=coords,
        name="signal",
        attrs={"source_type": type(source).__name__},
    )


def source_dataset(source, *, t=None):
    """Convert source signal data to a Dataset."""
    return xr.Dataset(
        data_vars={"signal": source_signal_data_array(source, t=t)},
        attrs={"beamz_kind": type(source).__name__},
    )


def _center_coords_from_attrs(ds, dim):
    resolution = ds.attrs.get("resolution")
    design_key = {"x": "design_width", "y": "design_height", "z": "design_depth"}.get(
        dim
    )
    span = ds.attrs.get(design_key) if design_key is not None else None
    if (
        resolution is None
        or span is None
        or not np.isfinite(float(span))
        or float(span) <= 0
    ):
        return None
    count = max(1, int(round(float(span) / float(resolution))))
    return (np.arange(count, dtype=float) + 0.5) * float(resolution)


def colocate_dataset(dataset, *, where="centers", coords=None):
    """Return a Dataset with spatial variables interpolated onto common coordinates."""
    ds = dataset.to_xarray() if hasattr(dataset, "to_xarray") else dataset
    if not isinstance(ds, xr.Dataset):
        raise TypeError(
            "colocate_dataset expects an xarray Dataset or object with to_xarray()."
        )
    if where not in {"centers"}:
        raise ValueError("Only where='centers' is currently supported.")

    target_coords = dict(coords or {})
    for dim in ("x", "y", "z"):
        if dim in target_coords:
            continue
        inferred = _center_coords_from_attrs(ds, dim)
        if inferred is not None:
            target_coords[dim] = inferred

    for dim, values in tuple(target_coords.items()):
        bounds = []
        for da in ds.data_vars.values():
            if dim in da.dims and dim in da.coords:
                coord = np.asarray(da.coords[dim], dtype=float)
                if coord.size:
                    bounds.append((float(np.nanmin(coord)), float(np.nanmax(coord))))
        if not bounds:
            continue
        lower = max(bound[0] for bound in bounds)
        upper = min(bound[1] for bound in bounds)
        clipped = np.asarray(values, dtype=float)
        clipped = clipped[(clipped >= lower) & (clipped <= upper)]
        if clipped.size:
            target_coords[dim] = clipped

    data_vars = {}
    for name, da in ds.data_vars.items():
        selectors = {
            dim: values
            for dim, values in target_coords.items()
            if dim in da.dims and dim in da.coords
        }
        data_vars[name] = da.interp(selectors) if selectors else da
    return xr.Dataset(data_vars=data_vars, attrs={**ds.attrs, "colocated": where})


def field_intensity(dataset, *, components=("Ex", "Ey", "Ez"), colocate=True):
    """Return ``sum(abs(E_i)**2)`` as an xarray DataArray."""
    ds = dataset.to_xarray() if hasattr(dataset, "to_xarray") else dataset
    ds = colocate_dataset(ds) if colocate else ds
    terms = [np.abs(ds[name]) ** 2 for name in components if name in ds]
    if not terms:
        raise ValueError(
            f"None of the requested components are present: {components!r}"
        )
    out = terms[0]
    for term in terms[1:]:
        out = out + term
    out.name = "intensity"
    out.attrs.update({"long_name": "field intensity", "units": "field^2"})
    return out


def poynting_vector(dataset, *, colocate=True, phasor=None):
    """Return the electromagnetic Poynting vector as an xarray Dataset."""
    ds = dataset.to_xarray() if hasattr(dataset, "to_xarray") else dataset
    ds = colocate_dataset(ds) if colocate else ds
    required = {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}
    missing = sorted(required.difference(ds.data_vars))
    if missing:
        raise ValueError(f"Missing field components for Poynting vector: {missing}")

    complex_data = any(np.iscomplexobj(ds[name].data) for name in required)
    use_phasor = complex_data if phasor is None else bool(phasor)
    hx = ds["Hx"].conj() if use_phasor else ds["Hx"]
    hy = ds["Hy"].conj() if use_phasor else ds["Hy"]
    hz = ds["Hz"].conj() if use_phasor else ds["Hz"]
    scale = 0.5 if use_phasor else 1.0
    data_vars = {
        "Sx": scale * (ds["Ey"] * hz - ds["Ez"] * hy),
        "Sy": scale * (ds["Ez"] * hx - ds["Ex"] * hz),
        "Sz": scale * (ds["Ex"] * hy - ds["Ey"] * hx),
    }
    if use_phasor:
        data_vars = {name: np.real(value) for name, value in data_vars.items()}
    for name, da in data_vars.items():
        da.name = name
        da.attrs.update({"long_name": "Poynting vector", "units": "W/m^2"})
    return xr.Dataset(
        data_vars=data_vars, attrs={**ds.attrs, "beamz_kind": "PoyntingVector"}
    )


def mode_dataset(source, *, t=None):
    """Convert a ModeSource profile and signal to an xarray Dataset."""
    from beamz.visual.data import mode_profile_data

    data_vars = {}
    try:
        payload = mode_profile_data(source)
    except RuntimeError:
        payload = None

    if payload is not None:
        profile = np.asarray(payload["profile"])
        dims = _spatial_dims(profile.ndim)
        coords = _axis_coords(dims, profile.shape)
        attrs = {
            "neff": payload.get("neff"),
            "direction": payload.get("direction"),
            "title": payload.get("title"),
        }
        data_vars["profile"] = xr.DataArray(
            profile,
            dims=dims,
            coords=coords,
            attrs=attrs,
        )
        data_vars["amplitude"] = xr.DataArray(
            np.asarray(payload["amplitude"]),
            dims=dims,
            coords=coords,
            attrs=attrs,
        )

    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        profile = getattr(source, f"_{component}_profile", None)
        if profile is None:
            continue
        arr = np.squeeze(np.asarray(profile))
        dims = _spatial_dims(arr.ndim)
        coords = _axis_coords(dims, arr.shape)
        data_vars[component] = xr.DataArray(
            arr,
            dims=dims,
            coords=coords,
            attrs={"component": component, "units": _FIELD_UNITS.get(component, "")},
        )

    eps_profile = getattr(source, "_eps_profile_2d", None)
    if eps_profile is not None:
        arr = np.squeeze(np.asarray(eps_profile))
        dims = _spatial_dims(arr.ndim)
        data_vars["permittivity"] = xr.DataArray(
            arr,
            dims=dims,
            coords=_axis_coords(dims, arr.shape),
            attrs={"component": "permittivity", "units": "relative"},
        )

    if hasattr(source, "signal"):
        data_vars["signal"] = source_signal_data_array(source, t=t)

    return xr.Dataset(data_vars=data_vars, attrs={"beamz_kind": type(source).__name__})


def proxy_monitor(
    *,
    fields,
    power_history=(),
    power_timestamps=(),
    power_spectrum=(),
    power_spectrum_frequencies=(),
    monitor_type="line",
    name=None,
):
    """Build a minimal monitor-like namespace for internal conversions."""
    return SimpleNamespace(
        fields=fields,
        power_history=power_history,
        power_timestamps=power_timestamps,
        power_spectrum=power_spectrum,
        power_spectrum_frequencies=power_spectrum_frequencies,
        monitor_type=monitor_type,
        name=name,
    )


__all__ = [
    "field_data_array",
    "colocate_dataset",
    "field_intensity",
    "mode_dataset",
    "monitor_dataset",
    "poynting_vector",
    "proxy_monitor",
    "simulation_dataset",
    "simulation_fields_dataset",
    "source_dataset",
    "source_signal_data_array",
]
