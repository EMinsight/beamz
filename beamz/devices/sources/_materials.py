"""Material sampling helpers for source injection on Yee components."""

from __future__ import annotations


def component_permittivity_at(fields, component: str, index):
    """Return permittivity sampled on the target E-field component lattice."""
    if component not in {"Ex", "Ey", "Ez"}:
        raise ValueError(f"Unsupported E component {component!r}")

    if fields.permittivity.ndim == 3:
        material = getattr(fields, f"eps_{component[-1].lower()}", None)
        if material is None:
            from beamz.simulation.yee import sample_voxel_grid_at_component_3d

            material = sample_voxel_grid_at_component_3d(
                fields.permittivity,
                component,
                stored_shape=tuple(getattr(fields, component).shape),
            )
        return material[index]

    if (
        getattr(fields, "plane_2d", None) == "xy"
        and component == "Ez"
        and hasattr(fields, "eps_tm_ez")
    ):
        return fields.eps_tm_ez[index]

    from beamz.simulation.yee import sample_voxel_grid_at_component_2d

    material = sample_voxel_grid_at_component_2d(
        fields.permittivity,
        component,
        getattr(fields, "plane_2d", "xy"),
        stored_shape=tuple(getattr(fields, component).shape),
    )
    return material[index]


def component_permeability_at(fields, component: str, index):
    """Return permeability sampled on the target H-field component lattice."""
    if component not in {"Hx", "Hy", "Hz"}:
        raise ValueError(f"Unsupported H component {component!r}")

    if fields.permittivity.ndim == 3:
        from beamz.simulation.yee import sample_voxel_grid_at_component_3d

        material = sample_voxel_grid_at_component_3d(
            fields.permeability,
            component,
            stored_shape=tuple(getattr(fields, component).shape),
        )
        return material[index]

    if getattr(fields, "plane_2d", None) == "xy" and component in {"Hx", "Hy"}:
        attr = f"mu_tm_h{component[-1].lower()}"
        if hasattr(fields, attr):
            return getattr(fields, attr)[index]

    from beamz.simulation.yee import sample_voxel_grid_at_component_2d

    material = sample_voxel_grid_at_component_2d(
        fields.permeability,
        component,
        getattr(fields, "plane_2d", "xy"),
        stored_shape=tuple(getattr(fields, component).shape),
    )
    return material[index]
