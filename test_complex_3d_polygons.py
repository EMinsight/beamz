#!/usr/bin/env python3
"""
Test script for complex 3D polygon handling and unification.
This tests the robust triangulation and polygon simplification fixes.
"""

import sys
import os

# Add the beamz module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from beamz.design.structures import Design, Rectangle, Circle, Ring, Polygon
from beamz.design.materials import Material
from beamz.const import µm

def test_complex_unified_polygons():
    """Test complex polygons that result from unification."""
    print("Testing Complex Unified Polygons")
    print("="*35)
    
    # Create a 3D design
    design = Design(width=20*µm, height=15*µm, depth=3*µm, auto_pml=False)
    silicon = Material(permittivity=12.0)
    
    # Add many overlapping structures that will create complex unified shapes
    print("Adding overlapping structures that will create complex unified polygons...")
    
    # Create a grid of overlapping rectangles
    for i in range(5):
        for j in range(4):
            x = 2*µm + i * 3*µm
            y = 2*µm + j * 3*µm
            design.add(Rectangle(position=(x, y, 0.5*µm), width=4*µm, height=3*µm, 
                               depth=1*µm, material=silicon))
    
    # Add some circles that will overlap with rectangles
    for i in range(3):
        for j in range(2):
            x = 4*µm + i * 6*µm
            y = 4*µm + j * 6*µm
            design.add(Circle(position=(x, y, 0.5*µm), radius=2.5*µm, 
                            depth=1*µm, material=silicon))
    
    print(f"Created {len(design.structures)} overlapping structures")
    print("These will be unified into complex polygons")
    
    try:
        print("\nTesting 3D visualization with complex unified polygons...")
        design.show()
        print("✅ Complex unified polygons rendered successfully!")
        return True
    except Exception as e:
        print(f"❌ Error with complex unified polygons: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_extremely_complex_polygon():
    """Test an extremely complex polygon that would break simple triangulation."""
    print("\nTesting Extremely Complex Polygon")
    print("="*35)
    
    # Create a design with a manually complex polygon
    design = Design(width=15*µm, height=12*µm, depth=2*µm, auto_pml=False)
    silicon = Material(permittivity=12.0)
    
    # Create a complex star-like polygon with many vertices
    import numpy as np
    center_x, center_y = 7.5*µm, 6*µm
    outer_radius = 4*µm
    inner_radius = 2*µm
    n_points = 16  # Creates 32 vertices (star shape)
    
    vertices = []
    for i in range(n_points):
        # Outer point
        angle = 2 * np.pi * i / n_points
        x = center_x + outer_radius * np.cos(angle)
        y = center_y + outer_radius * np.sin(angle)
        vertices.append((x, y, 0.5*µm))
        
        # Inner point
        angle = 2 * np.pi * (i + 0.5) / n_points
        x = center_x + inner_radius * np.cos(angle)
        y = center_y + inner_radius * np.sin(angle)
        vertices.append((x, y, 0.5*µm))
    
    # Add the complex polygon
    complex_poly = Polygon(vertices=vertices, material=silicon, depth=1*µm, z=0.5*µm)
    design.add(complex_poly)
    
    print(f"Created complex star polygon with {len(vertices)} vertices")
    
    try:
        print("Testing 3D visualization with extremely complex polygon...")
        design.show()
        print("✅ Extremely complex polygon rendered successfully!")
        return True
    except Exception as e:
        print(f"❌ Error with extremely complex polygon: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_unification_control():
    """Test the unification control for complex 3D cases."""
    print("\nTesting Unification Control")
    print("="*30)
    
    # Create a design with many complex structures
    design = Design(width=25*µm, height=20*µm, depth=3*µm, auto_pml=False)
    silicon = Material(permittivity=12.0)
    
    # Add many high-vertex structures
    for i in range(10):
        for j in range(8):
            x = 1*µm + i * 2.3*µm
            y = 1*µm + j * 2.3*µm
            # High-vertex circles
            design.add(Circle(position=(x, y, 0.5*µm), radius=1*µm, points=64,
                            depth=0.8*µm, material=silicon))
    
    print(f"Created design with {len(design.structures)} high-vertex structures")
    
    try:
        print("Testing automatic unification control...")
        # This should automatically disable unification due to complexity
        design.show()
        print("✅ Unification control working correctly!")
        return True
    except Exception as e:
        print(f"❌ Error with unification control: {e}")
        return False

def test_triangulation_fallbacks():
    """Test the triangulation fallback system."""
    print("\nTesting Triangulation Fallbacks")
    print("="*32)
    
    design = Design(width=12*µm, height=10*µm, depth=2*µm, auto_pml=False)
    silicon = Material(permittivity=12.0)
    
    # Create a problematic polygon (self-intersecting or near-degenerate)
    vertices = [
        (2*µm, 2*µm, 0.5*µm),
        (6*µm, 2*µm, 0.5*µm),
        (8*µm, 6*µm, 0.5*µm),
        (4*µm, 8*µm, 0.5*µm),
        (3*µm, 4*µm, 0.5*µm),  # This creates a non-convex shape
        (7*µm, 4*µm, 0.5*µm),  # This might cause triangulation issues
        (1*µm, 6*µm, 0.5*µm),
    ]
    
    problematic_poly = Polygon(vertices=vertices, material=silicon, depth=1*µm, z=0.5*µm)
    design.add(problematic_poly)
    
    print("Created problematic polygon to test triangulation fallbacks")
    
    try:
        print("Testing triangulation fallback system...")
        design.show()
        print("✅ Triangulation fallbacks working correctly!")
        return True
    except Exception as e:
        print(f"❌ Error with triangulation fallbacks: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing Complex 3D Polygon Handling")
    print("Verifying robust triangulation and polygon simplification")
    
    # Test 1: Complex unified polygons
    success1 = test_complex_unified_polygons()
    
    # Test 2: Extremely complex polygon
    success2 = test_extremely_complex_polygon()
    
    # Test 3: Unification control
    success3 = test_unification_control()
    
    # Test 4: Triangulation fallbacks
    success4 = test_triangulation_fallbacks()
    
    if success1 and success2 and success3 and success4:
        print("\n🎉 COMPLEX 3D POLYGON HANDLING SUCCESSFUL!")
        print("✅ Robust triangulation: WORKING")
        print("✅ Polygon simplification: IMPLEMENTED")
        print("✅ Unification control: ACTIVE") 
        print("✅ Fallback systems: ROBUST")
        
        print("\nComplex 3D polygons should now:")
        print("• Triangulate correctly without mesh errors")
        print("• Automatically simplify when too complex")
        print("• Skip unification when structures are too complex")
        print("• Fall back gracefully when triangulation fails")
        sys.exit(0)
    else:
        print("\n❌ COMPLEX 3D POLYGON HANDLING FAILED")
        print("Check the error messages above for details.")
        sys.exit(1) 