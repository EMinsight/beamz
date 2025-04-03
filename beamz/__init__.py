"""
BeamZ - A Python package for electromagnetic simulations.
"""

__version__ = "0.1.0"

# Import core components
from .sim import Simulation, StandardGrid, PML, Boundaries
from .materials import Material, MaterialLibrary
from .sources import PointSource, Wave
from .const import LIGHT_SPEED
from .viz import animate_field
from .helpers import progress_bar

__all__ = [
    "Simulation", "StandardGrid", "PML", "Boundaries",
    "Material", "MaterialLibrary",
    "PointSource", "Wave",
    "LIGHT_SPEED",
    "animate_field",
    "progress_bar"
] 