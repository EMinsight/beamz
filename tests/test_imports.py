"""
Tests for verifying all imports from the beamz package are working correctly.
"""

import pytest
from typing import List, Dict

# Test main package imports
def test_main_package_import():
    """Test that the main beamz package can be imported."""
    import beamz
    assert beamz.__version__ is not None

# Test design module imports
@pytest.mark.parametrize("name", [
    'Material', 'VarMaterial',
    'Design', 'Rectangle', 'Circle', 'Ring',
    'CircularBend', 'Polygon', 'Taper',
    'ModeSource', 'GaussianSource',
    'Monitor',
    'ramped_cosine', 'plot_signal',
    'solve_modes', 'slab_mode_source'
])
def test_design_imports(name: str):
    """Test that all design module components can be imported."""
    exec(f"from beamz.design import {name}")

# Test simulation module imports
@pytest.mark.parametrize("name", [
    'RegularGrid', 'FDTD', 'get_backend'
])
def test_simulation_imports(name: str):
    """Test that all simulation module components can be imported."""
    exec(f"from beamz.simulation import {name}")

# Test constant imports
@pytest.mark.parametrize("name", [
    'LIGHT_SPEED', 'VAC_PERMITTIVITY', 'VAC_PERMEABILITY', 
    'EPS_0', 'MU_0', 'um', 'nm', 'µm', 'μm'
])
def test_constant_imports(name: str):
    """Test that all constants can be imported."""
    exec(f"from beamz import {name}")

# Test that imported components are of the correct type
def test_imported_types():
    """Test that imported components are of the correct type."""
    from beamz.design import Material, Design, Rectangle
    from beamz.simulation import FDTD, RegularGrid
    
    # Test some basic type checks
    assert isinstance(Material, type)
    assert isinstance(Design, type)
    assert isinstance(Rectangle, type)
    assert isinstance(FDTD, type)
    assert isinstance(RegularGrid, type)
    
    # Test that constants are numbers
    from beamz import LIGHT_SPEED, EPS_0, MU_0
    assert isinstance(LIGHT_SPEED, (int, float))
    assert isinstance(EPS_0, (int, float))
    assert isinstance(MU_0, (int, float)) 