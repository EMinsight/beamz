"""
BeamZ - Create inverse designs for your photonic devices with ease and efficiency.
"""

__version__ = "0.0.1"

# Import main components
from .fdtd import FDTD
from .materials import MaterialLibrary
from .sources import SourceLibrary
from .structures import StructureLibrary
from .detectors import DetectorLibrary

__all__ = ["FDTD","MaterialLibrary","SourceLibrary","StructureLibrary","DetectorLibrary"] 