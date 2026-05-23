"""Plain xarray conversion helpers for BeamZ objects.

This module is intentionally factory-only. BeamZ keeps solver/runtime objects as
NumPy/JAX arrays and exposes xarray at the result boundary for labeled slicing,
metadata, and plotting.
"""

from __future__ import annotations

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
        coords.update(_axis_coords(spatial, arr.shape[1:], resolution=resolution))
    else:
        dims = _spatial_dims(arr.ndim, plane_2d=plane_2d)
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
    coords.update(_axis_coords(spatial, arr.shape[1:]))
    attrs = {
        "component": str(component),
        "units": _FIELD_UNITS.get(str(component), ""),
        "monitor_type": getattr(monitor, "monitor_type", None),
        "monitor_name": getattr(monitor, "name", None),
    }
    return xr.DataArray(arr, dims=dims, coords=coords, name=str(component), attrs=attrs)


def monitor_dataset(monitor_or_result):
    """Convert a ``Monitor`` or ``MonitorResults`` object to an xarray Dataset."""
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
    "mode_dataset",
    "monitor_dataset",
    "proxy_monitor",
    "simulation_dataset",
    "simulation_fields_dataset",
    "source_dataset",
    "source_signal_data_array",
]
