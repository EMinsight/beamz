#!/usr/bin/env python3
"""
Test script for 3D visualization in beamz design structures.
This script demonstrates the new 3D visualization capabilities using Plotly.
"""

import sys
import os

# Add the beamz module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from beamz.design.structures import Design, Rectangle, Circle, Ring
from beamz.design.materials import Material
from beamz.const import µm

def test_2d_visualization():
    """Test 2D visualization (should use matplotlib)."""
    print("Creating 2D design...")
    
    # Create 2D design
    design = Design(width=10*µm, height=8*µm)
    
    # Add some structures
    silicon = Material(permittivity=12.0)
    design.add(Rectangle(position=(2*µm, 2*µm), width=3*µm, height=2*µm, material=silicon))
    design.add(Circle(position=(6*µm, 4*µm), radius=1*µm, material=silicon))
    design.add(Ring(position=(3*µm, 5*µm), inner_radius=0.5*µm, outer_radius=1*µm, material=silicon))
    
    print(f"Design is 3D: {design.is_3d}")
    print("Showing 2D visualization...")
    design.show()

def test_3d_visualization():
    """Test 3D visualization (should use plotly)."""
    print("Creating 3D design...")
    
    # Create 3D design
    design = Design(width=10*µm, height=8*µm, depth=2*µm)
    
    # Add some 3D structures
    silicon = Material(permittivity=12.0)
    design.add(Rectangle(position=(2*µm, 2*µm), width=3*µm, height=2*µm, depth=1*µm, z=0.5*µm, material=silicon))
    design.add(Circle(position=(6*µm, 4*µm), radius=1*µm, depth=0.8*µm, z=0.2*µm, material=silicon))
    design.add(Ring(position=(3*µm, 5*µm), inner_radius=0.5*µm, outer_radius=1*µm, depth=1.5*µm, material=silicon))
    
    print(f"Design is 3D: {design.is_3d}")
    print("Showing 3D visualization...")
    design.show()

def test_mixed_2d_3d():
    """Test design with mixed 2D and 3D structures."""
    print("Creating design with mixed 2D/3D structures...")
    
    # Create design that becomes 3D when 3D structures are added
    design = Design(width=10*µm, height=8*µm)
    
    # Add 2D structures first
    silicon = Material(permittivity=12.0)
    design.add(Rectangle(position=(1*µm, 1*µm), width=2*µm, height=2*µm, material=silicon))
    
    print(f"After adding 2D rectangle, design is 3D: {design.is_3d}")
    
    # Add 3D structure (should trigger 3D mode)
    design.add(Rectangle(position=(4*µm, 4*µm), width=2*µm, height=2*µm, depth=1*µm, material=silicon))
    
    print(f"After adding 3D rectangle, design is 3D: {design.is_3d}")
    print("Showing visualization...")
    design.show()

if __name__ == "__main__":
    print("Testing beamz 3D visualization capabilities...")
    print("="*50)
    
    # Test 2D visualization
    test_2d_visualization()
    
    print("\n" + "="*50)
    
    # Test 3D visualization
    test_3d_visualization()
    
    print("\n" + "="*50)
    
    # Test mixed 2D/3D
    test_mixed_2d_3d()
    
    print("\nAll tests completed!") 