"""
BeamZ - Create inverse designs for your photonic devices with ease and efficiency.
"""

__version__ = "0.1.0"

# Import main components
from .fdtd import FDTD
from .materials import MaterialLibrary

__all__ = ["FDTD","MaterialLibrary"] 