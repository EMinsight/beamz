"""BeamZ-native equivalent of low-level PML material coupling tests.

BeamZ does not copy material tensors into a separate PML extension region. Its
equivalent contract is how PML conductivity shells merge into
``Fields.total_conductivity`` while CPML leaves the base material update
conductivity unchanged.
"""

import numpy as np
import pytest

from beamz import Design, Material, PML
from beamz.simulation.fields import Fields

pytestmark = pytest.mark.unit


def _make_fields(shape=(6, 8), *, resolution=0.1, base_sigma=0.0):
    permittivity = np.ones(shape, dtype=np.float32)
    conductivity = np.full(shape, base_sigma, dtype=np.float32)
    permeability = np.ones(shape, dtype=np.float32)
    return Fields(
        permittivity=permittivity,
        conductivity=conductivity,
        permeability=permeability,
        resolution=resolution,
        plane_2d="xy",
    )


def _make_design(shape=(6, 8), *, resolution=0.1):
    return Design(
        width=shape[1] * resolution,
        height=shape[0] * resolution,
        material=Material(permittivity=1.0, permeability=1.0, conductivity=0.0),
    )


def _attach_pml(fields, pml, design, *, resolution=0.1, dt=1e-15):
    payload = pml.create_pml_regions(
        fields, design, resolution=resolution, dt=dt, plane_2d="xy"
    )
    fields.has_pml = True
    fields.has_cpml = pml.formulation == "cpml"
    fields.pml_data = payload
    fields._init_material_parameters()
    return payload


def test_sigma_pml_merges_left_shell_into_total_conductivity():
    fields = _make_fields()
    design = _make_design()
    pml = PML(edges=["left"], thickness=0.2, sigma_max=6.0, formulation="sigma")

    payload = _attach_pml(fields, pml, design)
    total_sigma = np.asarray(fields.total_conductivity, dtype=np.float64)
    mask = np.asarray(payload["mask"], dtype=bool)
    sigma_shell = np.asarray(payload["sigma_x"] + payload["sigma_y"], dtype=np.float64)

    np.testing.assert_allclose(total_sigma, sigma_shell)
    assert np.any(mask[:, 0])
    assert not np.any(mask[:, -1])
    assert float(np.max(total_sigma[:, 0])) > 0.0
    assert float(np.max(total_sigma[:, -1])) == pytest.approx(0.0)


def test_sigma_pml_merges_right_shell_into_total_conductivity():
    fields = _make_fields()
    design = _make_design()
    pml = PML(edges=["right"], thickness=0.2, sigma_max=6.0, formulation="sigma")

    payload = _attach_pml(fields, pml, design)
    total_sigma = np.asarray(fields.total_conductivity, dtype=np.float64)
    mask = np.asarray(payload["mask"], dtype=bool)

    assert np.any(mask[:, -1])
    assert not np.any(mask[:, 0])
    assert float(np.max(total_sigma[:, -1])) > 0.0
    assert float(np.max(total_sigma[:, 0])) == pytest.approx(0.0)


def test_sigma_pml_preserves_larger_base_conductivity():
    fields = _make_fields(base_sigma=0.0)
    design = _make_design()
    fields.conductivity = fields.conductivity.at[3, 4].set(9.0)
    pml = PML(edges=["left"], thickness=0.2, sigma_max=4.0, formulation="sigma")

    _attach_pml(fields, pml, design)
    total_sigma = np.asarray(fields.total_conductivity, dtype=np.float64)

    assert total_sigma[3, 4] == pytest.approx(9.0)


def test_cpml_auxiliary_profiles_do_not_modify_total_conductivity():
    fields = _make_fields(base_sigma=0.25)
    design = _make_design()
    pml = PML(
        edges=["left", "right"],
        thickness=0.2,
        sigma_max=6.0,
        alpha_max=0.5,
        formulation="cpml",
    )

    _attach_pml(fields, pml, design)

    np.testing.assert_allclose(
        np.asarray(fields.total_conductivity, dtype=np.float64),
        np.asarray(fields.conductivity, dtype=np.float64),
    )
