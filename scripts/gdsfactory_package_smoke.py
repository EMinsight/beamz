#!/usr/bin/env python3
"""Execute one packaged local BeamZ GDSFactory S-parameter column."""

from __future__ import annotations

import numpy as np

from beamz.design.gdsfactory import Settings, prepare


def main() -> int:
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
    if through.shape != (1,) or not np.isfinite(through).all():
        raise RuntimeError("GDSFactory BeamZ smoke simulation returned invalid S21.")
    print("GDSFactory BeamZ packaged smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
