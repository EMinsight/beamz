"""Shared Yee-lattice constants used across package layers."""

_COMPONENT_AXIS_OFFSETS_3D = {
    "Ex": {"z": 0.0, "y": 0.0, "x": 0.5},
    "Ey": {"z": 0.0, "y": 0.5, "x": 0.0},
    "Ez": {"z": 0.5, "y": 0.0, "x": 0.0},
    "Hx": {"z": 0.5, "y": 0.5, "x": 0.0},
    "Hy": {"z": 0.5, "y": 0.0, "x": 0.5},
    "Hz": {"z": 0.0, "y": 0.5, "x": 0.5},
}


def component_axis_offsets_3d(component: str) -> dict[str, float]:
    """Return physical Yee offsets, in grid-cell units, for a 3D component."""
    try:
        return dict(_COMPONENT_AXIS_OFFSETS_3D[component])
    except KeyError as exc:
        raise ValueError(f"Unsupported component {component!r}") from exc
