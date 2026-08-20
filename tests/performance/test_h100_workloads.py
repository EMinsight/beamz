from __future__ import annotations

import pytest

from tests.performance.h100_workloads import H100Workload


def test_realistic_h100_workload_compiles_all_timed_features():
    workload = H100Workload(
        name="realistic_3d_smoke",
        shape_zyx=(8, 10, 12),
        timesteps=3,
        resolution=100e-9,
        pml_cells=2,
        heterogeneous=True,
        cpml=True,
        source=True,
        monitor=True,
    )

    simulation = workload.build()
    program = simulation.compile()

    assert simulation.num_steps == 3
    assert program.grid.material_grid.shape == (8, 10, 12)
    assert program.boundary.cpml.enabled
    assert program.sources
    assert program.monitors
    assert workload.feature_labels == {
        "boundaries": ("CPML(all)",),
        "sources": ("GaussianSource(Ez)",),
        "monitors": ("FieldMonitor(plane,DFT)",),
    }


def test_h100_workload_rejects_cpml_that_consumes_domain():
    with pytest.raises(ValueError, match="CPML"):
        H100Workload(
            name="invalid",
            shape_zyx=(8, 10, 12),
            timesteps=3,
            pml_cells=4,
            cpml=True,
        )
