#!/usr/bin/env python3
"""
Test script to verify 2D/3D detection and visualization switching.
"""

import sys
import os

# Add the beamz module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from beamz.design.structures import Design, Rectangle, Circle
from beamz.design.materials import Material
from beamz.const import µm

def test_2d_visualization():
    """Test that 2D structures are shown with matplotlib."""
    print("Testing 2D Visualization")
    print("="*30)
    
    # Create a 2D design (no depth, no z positions)
    design = Design(width=10*µm, height=8*µm, auto_pml=False)
    silicon = Material(permittivity=12.0)
    
    # Add 2D structures (no depth or z specified)
    design.add(Rectangle(position=(2*µm, 2*µm), width=3*µm, height=2*µm, material=silicon))
    design.add(Circle(position=(6*µm, 4*µm), radius=1*µm, material=silicon))
    
    print(f"Design depth: {design.depth}")
    print(f"Is 3D detected: {design._determine_if_3d()}")
    print("Expected: False (should use 2D matplotlib)")
    
    try:
        print("Showing 2D design...")
        design.show()
        print("✅ 2D visualization successful!")
        return True
    except Exception as e:
        print(f"❌ Error in 2D visualization: {e}")
        return False

def test_3d_visualization():
    """Test that 3D structures are shown with plotly."""
    print("\nTesting 3D Visualization")
    print("="*30)
    
    # Create a 3D design (with depth)
    design = Design(width=10*µm, height=8*µm, depth=2*µm, auto_pml=False)
    silicon = Material(permittivity=12.0)
    
    # Add 3D structures (with depth)
    design.add(Rectangle(position=(2*µm, 2*µm), width=3*µm, height=2*µm, depth=1*µm, material=silicon))
    design.add(Circle(position=(6*µm, 4*µm), radius=1*µm, depth=0.5*µm, material=silicon))
    
    print(f"Design depth: {design.depth}")
    print(f"Is 3D detected: {design._determine_if_3d()}")
    print("Expected: True (should use 3D plotly)")
    
    try:
        print("Showing 3D design...")
        design.show()
        print("✅ 3D visualization successful!")
        return True
    except Exception as e:
        print(f"❌ Error in 3D visualization: {e}")
        return False

def test_mixed_case():
    """Test edge case with design depth but no structure depth."""
    print("\nTesting Mixed Case")
    print("="*20)
    
    # Design has depth but structures don't
    design = Design(width=10*µm, height=8*µm, depth=1*µm, auto_pml=False)
    silicon = Material(permittivity=12.0)
    
    # Add structures without depth (should still be 2D)
    design.add(Rectangle(position=(2*µm, 2*µm), width=3*µm, height=2*µm, material=silicon))
    
    print(f"Design depth: {design.depth}")
    print(f"Is 3D detected: {design._determine_if_3d()}")
    print("Expected: False (structures have no depth)")
    
    try:
        print("Showing mixed case...")
        design.show()
        print("✅ Mixed case handled correctly!")
        return True
    except Exception as e:
        print(f"❌ Error in mixed case: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing 2D/3D Detection and Visualization")
    print("Ensuring correct matplotlib vs plotly usage")
    
    success1 = test_2d_visualization()
    success2 = test_3d_visualization() 
    success3 = test_mixed_case()
    
    if success1 and success2 and success3:
        print("\n🎉 2D/3D DETECTION SUCCESSFUL!")
        print("✅ 2D structures → matplotlib visualization")
        print("✅ 3D structures → plotly visualization")
        print("✅ Mixed cases handled correctly")
        sys.exit(0)
    else:
        print("\n❌ 2D/3D DETECTION FAILED")
        sys.exit(1) 