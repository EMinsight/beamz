"""Tag-only local execution evidence for the packaged GDSFactory workflow."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from beamz.design.gdsfactory import Settings, prepare

pytestmark = [
    pytest.mark.release,
    pytest.mark.skipif(
        importlib.util.find_spec("gdsfactory") is None,
        reason="gdsfactory not installed",
    ),
    pytest.mark.filterwarnings(
        "ignore:Support for class-based `config` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:Implicitly cleaning up <TemporaryDirectory.*:ResourceWarning"
    ),
    pytest.mark.filterwarnings("ignore:unclosed file .*gdsfactory.*:ResourceWarning"),
]


def test_packaged_gdsfactory_straight_executes_one_native_beamz_column():
    """A built package can prepare and execute the documented local workflow."""
    setup = prepare(
        "straight",
        settings=Settings(
            wavelengths=(1.55e-6, 1.550001e-6),
            wavelength_points=1,
            xy_padding=1.5e-6,
            z_padding=0.5e-6,
            pml_thickness=0.5e-6,
            run_time=4e-15,
        ),
        component_settings={"length": 1.0},
    )

    result = setup.run_sparameters(["o1"])
    through = np.asarray(result.sparameters.s_matrix[("o2", "o1")])

    assert through.shape == (1,)
    assert np.isfinite(through).all()
    assert result.provenance["source_ports"] == ("o1",)
