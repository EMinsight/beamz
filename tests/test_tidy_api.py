import numpy as np
import pytest

import beamz as bz


def test_modal_notebook_api_surface_is_available():
    for name in (
        "BoundarySpec",
        "Box",
        "GaussianPulse",
        "GridSpec",
        "ModeData",
        "ModeSolver",
        "ModeSpec",
        "Structure",
        "inf",
    ):
        assert hasattr(bz, name)


def test_box_rejects_negative_sizes():
    with pytest.raises(ValueError, match="non-negative"):
        bz.Box(center=(0.0, 0.0, 0.0), size=(-1.0, 1.0, 1.0))


def test_notebook_style_simulation_construction_with_design_domain_and_specs():
    si = bz.Material(permittivity=12.0)
    sio2 = bz.Material(permittivity=2.0)
    design = bz.Design(background=sio2)
    design += bz.Box(
        center=(0.0, 0.0, 0.0),
        size=(1.0, 0.5, 0.2),
        material=si,
    )
    sim = bz.Simulation(
        domain=(2.0, 2.0, 1.0),
        grid_spec=bz.GridSpec.uniform(0.5),
        design=design,
        sources=[],
        monitors=[],
        boundary_spec=bz.BoundarySpec.all_sides(),
        run_time=2e-9,
    )

    assert sim.domain == (2.0, 2.0, 1.0)
    assert sim.time.size >= 2


def test_mode_solver_and_mode_data_notebook_helpers_import_without_pandas_dependency():
    mode_data = bz.ModeData(
        frequencies=np.asarray([2.0e14]),
        neffs=np.asarray([[2.0 + 0.0j]]),
        e_fields=np.ones((1, 1, 3, 2, 2), dtype=np.complex128),
        h_fields=np.ones((1, 1, 3, 2, 2), dtype=np.complex128),
        eps_profiles=np.ones((1, 2, 2), dtype=float),
        resolution=0.1,
    )

    pytest.importorskip("pandas")
    df = mode_data.to_dataframe()
    assert list(df.index.names) == ["f", "mode_index"]
