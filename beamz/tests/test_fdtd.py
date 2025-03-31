"""
Test basic 2D FDTD simulation functionality.
"""

import numpy as np
import pytest
from beamz.fdtd import FDTD

def test_fdtd_2d_initialization():
    """Test that FDTD can be initialized with 2D parameters."""
    # Arrange
    w, h = 100, 100  # Grid size (µm)
    dx = 0.1  # Cell size (µm)
    dt = 0.99 * np.sqrt(1/(2/dx**2))  # CFL condition
    
    # Act
    fdtd = FDTD(w=w, h=h, dx=dx, dt=dt)
    
    # Assert
    assert fdtd.w == w
    assert fdtd.h == h
    assert fdtd.dx == dx
    assert fdtd.dt == dt
    assert fdtd.Ez.shape == (w//dx, h//dx)
    assert fdtd.Hx.shape == (w//dx, h//dx-1)
    assert fdtd.Hy.shape == (w//dx-1, h//dx)

def test_fdtd_2d_source():
    """Test that a source can be added to the simulation."""
    # Arrange
    fdtd = FDTD(nx=100, ny=100, dx=0.1, dy=0.1)
    source_x, source_y = 50, 50
    wavelength = 1.55  # micrometers
    
    # Act
    fdtd.add_source(source_x, source_y, wavelength)
    
    # Assert
    assert fdtd.source_x == source_x
    assert fdtd.source_y == source_y
    assert fdtd.wavelength == wavelength
    assert fdtd.source is not None

def test_fdtd_2d_propagation():
    """Test that fields propagate correctly in 2D."""
    # Arrange
    fdtd = FDTD(nx=100, ny=100, dx=0.1, dy=0.1)
    fdtd.add_source(50, 50, 1.55)
    initial_energy = np.sum(fdtd.Ez**2)
    
    # Act
    fdtd.run(steps=10)
    final_energy = np.sum(fdtd.Ez**2)
    
    # Assert
    assert final_energy > 0  # Energy should be non-zero
    assert final_energy != initial_energy  # Energy should change during propagation

def test_fdtd_2d_boundary_conditions():
    """Test that boundary conditions are properly applied."""
    # Arrange
    fdtd = FDTD(nx=100, ny=100, dx=0.1, dy=0.1)
    fdtd.add_source(50, 50, 1.55)
    
    # Act
    fdtd.run(steps=10)
    
    # Assert
    # Check that fields at boundaries are properly handled
    assert np.all(np.isfinite(fdtd.Ez[0, :]))  # Top boundary
    assert np.all(np.isfinite(fdtd.Ez[-1, :]))  # Bottom boundary
    assert np.all(np.isfinite(fdtd.Ez[:, 0]))   # Left boundary
    assert np.all(np.isfinite(fdtd.Ez[:, -1]))  # Right boundary 