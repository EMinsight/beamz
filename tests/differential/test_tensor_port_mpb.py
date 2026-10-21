"""Independent Meep/MPB 1.34.0 reference at 80 cells/um.

A 0.6 x 0.6 um guide in vacuum, wavelength 1.55 um, epsilon diagonal
(6, 5, 4), off-diagonal (xy, xz, yz) = (0.2, 0.1, 0.3). Periodic transverse
supercells of 4 and 2 um give the fundamental effective indices below.
MPB find_k used three bands, NO_PARITY, and eigensolver tolerance 1e-10.
"""

from tests.validation.analytical.test_tensor_port_maxwell import mode_for


def test_tensor_guide_refines_toward_independent_mpb_reference():
    reference = 1.7430003206223845
    smaller_box = 1.7430624096798615
    assert abs(reference - smaller_box) < 1e-4
    errors = [abs(mode_for(cells=n).neff.real - reference) for n in (20, 40, 80)]
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] / reference < 0.002


def test_rectilinear_tensor_guide_refines_toward_independent_mpb_reference():
    from tests.validation.analytical.test_rectilinear_tensor_modes import (
        rectilinear_mode,
    )

    reference = 1.7430003206223845
    errors = [
        abs(rectilinear_mode(cells=n).neff.real - reference) for n in (20, 40, 80)
    ]
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] / reference < 0.002
