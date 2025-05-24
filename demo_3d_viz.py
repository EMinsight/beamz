#!/usr/bin/env python3
"""
Comprehensive demo of beamz 3D visualization capabilities.
This script demonstrates both 2D and 3D visualization using matplotlib and plotly.
"""

import sys
import os

# Add the beamz module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from beamz.design.structures import Design, Rectangle, Circle, Ring, Taper
from beamz.design.materials import Material
from beamz.const import µm

def demo_2d_photonic_device():
    """Demo 2D photonic device visualization."""
    print("="*60)
    print("DEMO 1: 2D Photonic Waveguide with Ring Resonator")
    print("="*60)
    
    # Create 2D design
    design = Design(width=20*µm, height=12*µm, auto_pml=True)
    
    # Materials
    silicon = Material(permittivity=12.0, name="Silicon")
    sio2 = Material(permittivity=2.25, name="SiO2")
    
    # Waveguide structures
    design.add(Rectangle(position=(2*µm, 5*µm), width=15*µm, height=0.5*µm, material=silicon))
    design.add(Rectangle(position=(2*µm, 6.5*µm), width=15*µm, height=0.5*µm, material=silicon))
    
    # Ring resonator
    design.add(Ring(position=(10*µm, 6*µm), inner_radius=2*µm, outer_radius=2.25*µm, material=silicon))
    
    # Coupling regions
    design.add(Rectangle(position=(8*µm, 5.75*µm), width=4*µm, height=0.5*µm, material=silicon))
    
    print(f"Design is 3D: {design.is_3d}")
    print("Showing 2D visualization with matplotlib...")
    design.show()

def demo_3d_photonic_device():
    """Demo 3D photonic device visualization."""
    print("="*60)
    print("DEMO 2: 3D Multilayer Photonic Structure")
    print("="*60)
    
    # Create 3D design
    design = Design(width=20*µm, height=12*µm, depth=3*µm, auto_pml=True)
    
    # Materials
    silicon = Material(permittivity=12.0, name="Silicon")
    sin = Material(permittivity=7.0, name="SiN") 
    sio2 = Material(permittivity=2.25, name="SiO2")
    
    # Bottom layer - Silicon substrate
    design.add(Rectangle(position=(0*µm, 0*µm), width=20*µm, height=12*µm, 
                        depth=0.5*µm, z=0*µm, material=silicon))
    
    # Middle layer - SiO2 isolation
    design.add(Rectangle(position=(0*µm, 0*µm), width=20*µm, height=12*µm, 
                        depth=0.5*µm, z=0.5*µm, material=sio2))
    
    # Top layer - SiN waveguides at different heights
    design.add(Rectangle(position=(2*µm, 5*µm), width=15*µm, height=0.5*µm, 
                        depth=0.22*µm, z=1.5*µm, material=sin))
    design.add(Rectangle(position=(2*µm, 6.5*µm), width=15*µm, height=0.5*µm, 
                        depth=0.22*µm, z=2*µm, material=sin))
    
    # 3D Ring resonator with varying height
    design.add(Ring(position=(10*µm, 6*µm), inner_radius=2*µm, outer_radius=2.25*µm, 
                   depth=0.3*µm, z=1.8*µm, material=sin))
    
    # Tapered coupler
    design.add(Taper(position=(7*µm, 5.75*µm), input_width=0.5*µm, output_width=1*µm, 
                    length=3*µm, depth=0.22*µm, z=1.75*µm, material=sin))
    
    print(f"Design is 3D: {design.is_3d}")
    print("Showing 3D visualization with plotly...")
    design.show()

def demo_complex_3d_structure():
    """Demo complex 3D structure with multiple layers."""
    print("="*60)
    print("DEMO 3: Complex 3D Photonic Crystal Structure")
    print("="*60)
    
    # Create 3D design
    design = Design(width=15*µm, height=15*µm, depth=2*µm, auto_pml=True)
    
    # Materials
    silicon = Material(permittivity=12.0, name="Silicon")
    air = Material(permittivity=1.0, name="Air")
    
    # Create a photonic crystal lattice
    lattice_constant = 2*µm
    hole_radius = 0.3*µm
    
    # Background silicon slab
    design.add(Rectangle(position=(0*µm, 0*µm), width=15*µm, height=15*µm, 
                        depth=0.5*µm, z=0.75*µm, material=silicon))
    
    # Create array of air holes at different heights
    for i in range(7):
        for j in range(7):
            x = (i + 0.5) * lattice_constant
            y = (j + 0.5) * lattice_constant
            if x < 15*µm and y < 15*µm:
                # Varying hole depths to create 3D effect
                hole_depth = 0.3*µm + 0.1*µm * (i + j) / 12
                z_pos = 0.75*µm + (0.25*µm - hole_depth/2)
                design.add(Circle(position=(x, y), radius=hole_radius, 
                                 depth=hole_depth, z=z_pos, material=air))
    
    # Central defect - larger structure
    design.add(Circle(position=(7.5*µm, 7.5*µm), radius=0.8*µm, 
                     depth=0.8*µm, z=0.6*µm, material=air))
    
    print(f"Design is 3D: {design.is_3d}")
    print("Showing 3D visualization with plotly...")
    design.show()

if __name__ == "__main__":
    print("Beamz 3D Visualization Demo")
    print("This demo showcases both 2D and 3D visualization capabilities")
    print("Press Enter to continue through each demo...")
    
    input("\nPress Enter to start Demo 1 (2D visualization)...")
    demo_2d_photonic_device()
    
    input("\nPress Enter to start Demo 2 (3D visualization)...")
    demo_3d_photonic_device()
    
    input("\nPress Enter to start Demo 3 (Complex 3D structure)...")
    demo_complex_3d_structure()
    
    print("\n" + "="*60)
    print("Demo completed! Key features demonstrated:")
    print("- 2D visualization using matplotlib (faster, better for simple designs)")
    print("- 3D visualization using plotly (interactive, web-based)")
    print("- Automatic detection of 2D vs 3D structures")
    print("- Support for depth and z-positioning of structures")
    print("- No window closing issues (plotly is web-based)")
    print("="*60) 