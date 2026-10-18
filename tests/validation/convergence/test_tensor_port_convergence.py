"""Refinement of the full-tensor port eigenproblem."""

import numpy as np

from tests.validation.analytical.test_tensor_port_maxwell import mode_for


def test_tensor_guided_mode_has_second_order_grid_convergence():
    values = [mode_for(cells=n).neff.real for n in (20, 40, 80)]
    order = np.log2(abs((values[0] - values[1]) / (values[1] - values[2])))
    assert 1.7 < order < 2.3


def test_rectilinear_tensor_guided_mode_has_second_order_grid_convergence():
    from tests.validation.analytical.test_rectilinear_tensor_modes import (
        rectilinear_mode,
    )

    values = [rectilinear_mode(cells=n).neff.real for n in (20, 40, 80)]
    order = np.log2(abs((values[0] - values[1]) / (values[1] - values[2])))
    assert 1.7 < order < 2.3
