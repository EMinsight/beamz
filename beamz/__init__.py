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
from beamz.design.signals import ramped_cosine, plot_signal
from beamz.simulation.meshing import RegularGrid
from beamz.simulation.fdtd import FDTD

# Import optimization-related classes


# Import UI helpers
from beamz.helpers import (
    display_header, display_status, create_rich_progress,
    display_parameters, display_results,
    display_simulation_status, display_optimization_progress,
    display_time_elapsed, tree_view, code_preview, get_si_scale_and_label, calc_optimal_fdtd_params
)

# Version information
__version__ = "0.1.0"