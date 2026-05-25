"""
Design module for BEAMZ - Contains components for designing photonic structures.
"""

from beamz.design import io
from beamz.design.core import Design
from beamz.design.materials import CustomMaterial, Material
from beamz.design.meshing import RegularGrid, RegularGrid3D, create_mesh
from beamz.design.structures import (
    Box,
    Circle,
    CircularBend,
    Polygon,
    Rectangle,
    Ring,
    Structure,
    Taper,
)

__all__ = [
    "Material",
    "CustomMaterial",
    "Design",
    "Box",
    "Structure",
    "Rectangle",
    "Circle",
    "Ring",
    "CircularBend",
    "Polygon",
    "Taper",
    "RegularGrid",
    "RegularGrid3D",
    "create_mesh",
    "io",
]
