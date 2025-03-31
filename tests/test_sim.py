"""
Test basic 2D FDTD simulation functionality.
"""

import numpy as np
import pytest
from beamz.sim import Simulation

def test_2D_sim_init():
    """Test that FDTD can be initialized with 2D parameters."""
    # Physical parameters
    wavelength = 1550e-9  # m (1.55 μm)
    c = 299792458  # m/s (speed of light)
    f = c / wavelength  # Hz (frequency)
    
    # Grid parameters
    # For accurate FDTD, we need at least 10-20 points per wavelength
    points_per_wavelength = 20
    cell_size = wavelength / points_per_wavelength  # m
    
    # Calculate grid dimensions (make it large enough to see propagation)
    grid_width = wavelength * 10  # 10 wavelengths wide
    grid_height = wavelength * 10  # 10 wavelengths high
    nx = int(grid_width / cell_size)  # number of cells in x
    ny = int(grid_height / cell_size)  # number of cells in y
    
    # Time step calculation
    # For 2D FDTD, the CFL condition is: dt <= 1/(c * sqrt(1/dx^2 + 1/dy^2))
    # For uniform grid (dx = dy), this simplifies to: dt <= dx/(c * sqrt(2))
    dt = 0.99 * cell_size / (c * np.sqrt(2))  # s (99% of CFL limit for stability)
    
    # Simulation duration (enough time for wave to propagate across grid)
    sim_duration = grid_width / c * 2  # s (2x time to cross grid)
    num_steps = int(sim_duration / dt)
    
    # Act
    sim = Simulation(
        type="2D",
        size=(nx, ny),
        cell_size=cell_size,
        dt=dt,
        time=sim_duration
    )
    
    # Assert
    assert sim.nx == nx
    assert sim.ny == ny
    assert sim.dx == cell_size
    assert sim.dy == cell_size
    assert sim.dt == dt
    assert sim.num_steps == num_steps
    
    # Check that CFL condition is satisfied
    cfl = c * dt * np.sqrt(1/cell_size**2 + 1/cell_size**2)
    assert cfl <= 1.0, "CFL condition violated"
    
    # Check that wavelength is properly resolved
    assert cell_size <= wavelength/10, "Grid resolution too coarse"
    
    # Check field array shapes
    assert sim.Ez.shape == (nx, ny)
    assert sim.Hx.shape == (nx, ny-1)
    assert sim.Hy.shape == (nx-1, ny)