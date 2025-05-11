"""
BeamZ - A Python package for electromagnetic simulations.
"""

# Import constants
from beamz.const import *

# Import design-related classes and functions
from beamz.design.materials import Material, VarMaterial
from beamz.design.structures import (
    Design, Rectangle, Circle, Ring, 
    CircularBend, Polygon, Taper
)
from beamz.design.sources import ModeSource, GaussianSource
from beamz.design.monitors import Monitor

# Import simulation-related classes and functions
from beamz.design.signals import ramped_cosine
from beamz.simulation.meshing import RegularGrid
from beamz.simulation.fdtd import FDTD

# Version information
__version__ = "0.1.0"