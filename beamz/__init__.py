"""
BeamZ - A Python package for electromagnetic simulations.
"""

__version__ = "0.1.0"

# Import core components
from .sim import Simulation
from .materials import MaterialLibrary
from .helpers import progress_bar

__all__ = ["Simulation", "MaterialLibrary", "progress_bar"] 