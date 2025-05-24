#!/usr/bin/env python3
"""
Demo of the enhanced beamz 3D visualization with modern styling.
This script showcases the improved 3D visualization with black outlines, better lighting, and modern UI.
"""

import sys
import os

# Add the beamz module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from beamz.design.structures import Design, Rectangle, Circle, Ring, Taper
from beamz.design.materials import Material
from beamz.const import µm

def demo_modern_3d_styling():
    """Demo modern 3D styling with enhanced visuals."""
    print("="*70)
    print("DEMO: Modern 3D Visualization with Enhanced Styling")
    print("="*70)
    
    # Create 3D design
    design = Design(width=16*µm, height=12*µm, depth=3*µm, auto_pml=True)
    
    # Create materials for better visualization
    silicon = Material(permittivity=12.0)
    silicon_nitride = Material(permittivity=7.0) 
    silicon_dioxide = Material(permittivity=2.25)
    polymer = Material(permittivity=2.5)
    air = Material(permittivity=1.0)
    
    # Substrate layer
    design.add(Rectangle(position=(0*µm, 0*µm), width=16*µm, height=12*µm, 
                        depth=0.8*µm, z=0*µm, material=silicon))
    
    # Oxide isolation layer
    design.add(Rectangle(position=(0*µm, 0*µm), width=16*µm, height=12*µm, 
                        depth=0.4*µm, z=0.8*µm, material=silicon_dioxide))
    
    # Waveguide core structures at different heights
    design.add(Rectangle(position=(2*µm, 4*µm), width=10*µm, height=0.6*µm, 
                        depth=0.22*µm, z=1.5*µm, material=silicon_nitride))
    design.add(Rectangle(position=(2*µm, 7*µm), width=10*µm, height=0.6*µm, 
                        depth=0.22*µm, z=1.8*µm, material=silicon_nitride))
    
    # Ring resonators with different materials and heights
    design.add(Ring(position=(8*µm, 6*µm), inner_radius=1.8*µm, outer_radius=2.1*µm, 
                   depth=0.25*µm, z=1.7*µm, material=silicon_nitride))
    design.add(Ring(position=(8*µm, 6*µm), inner_radius=1.2*µm, outer_radius=1.5*µm, 
                   depth=0.15*µm, z=2.1*µm, material=polymer))
    
    # Tapered couplers
    design.add(Taper(position=(5*µm, 4.2*µm), input_width=0.6*µm, output_width=1.2*µm, 
                    length=2*µm, depth=0.22*µm, z=1.6*µm, material=silicon_nitride))
    design.add(Taper(position=(5*µm, 6.8*µm), input_width=0.6*µm, output_width=1.2*µm, 
                    length=2*µm, depth=0.22*µm, z=1.9*µm, material=silicon_nitride))
    
    # Air-filled photonic crystal holes
    hole_positions = [
        (4*µm, 2*µm), (6*µm, 2*µm), (8*µm, 2*µm), (10*µm, 2*µm),
        (4*µm, 9.5*µm), (6*µm, 9.5*µm), (8*µm, 9.5*µm), (10*µm, 9.5*µm)
    ]
    
    for pos in hole_positions:
        design.add(Circle(position=pos, radius=0.3*µm, 
                         depth=0.6*µm, z=1.2*µm, material=air))
    
    # Metal contact pads (using high permittivity to simulate metal)
    metal = Material(permittivity=100.0)
    design.add(Rectangle(position=(1*µm, 1*µm), width=1.5*µm, height=1*µm, 
                        depth=0.1*µm, z=2.5*µm, material=metal))
    design.add(Rectangle(position=(13.5*µm, 10*µm), width=1.5*µm, height=1*µm, 
                        depth=0.1*µm, z=2.5*µm, material=metal))
    
    print(f"Design is 3D: {design.is_3d}")
    print("Features demonstrated:")
    print("✅ Black outlines on all 3D objects")
    print("✅ Enhanced lighting and materials")
    print("✅ Modern color palette")
    print("✅ Material-based consistent coloring")
    print("✅ Interactive hover information")
    print("✅ Improved camera positioning")
    print("✅ Professional styling and typography")
    print("✅ Ground plane for elevated structures")
    print("\nShowing enhanced 3D visualization...")
    design.show()

def demo_complex_multilayer():
    """Demo complex multilayer structure with the new styling."""
    print("="*70)
    print("DEMO: Complex Multilayer Photonic Device")
    print("="*70)
    
    # Create 3D design
    design = Design(width=20*µm, height=15*µm, depth=4*µm, auto_pml=True)
    
    # Materials
    substrate = Material(permittivity=11.8)
    active = Material(permittivity=12.5)
    cladding = Material(permittivity=10.2)
    contact = Material(permittivity=50.0)
    
    # Substrate
    design.add(Rectangle(position=(0*µm, 0*µm), width=20*µm, height=15*µm, 
                        depth=1.5*µm, z=0*µm, material=substrate))
    
    # Active quantum wells
    for i in range(3):
        z_pos = 1.5*µm + i * 0.3*µm
        design.add(Rectangle(position=(3*µm, 3*µm), width=14*µm, height=9*µm, 
                            depth=0.1*µm, z=z_pos, material=active))
        design.add(Rectangle(position=(3*µm, 3*µm), width=14*µm, height=9*µm, 
                            depth=0.1*µm, z=z_pos + 0.15*µm, material=cladding))
    
    # Ridge waveguide
    design.add(Rectangle(position=(8*µm, 6*µm), width=4*µm, height=3*µm, 
                        depth=0.8*µm, z=2.4*µm, material=active))
    
    # Contact layers
    design.add(Rectangle(position=(2*µm, 1*µm), width=16*µm, height=2*µm, 
                        depth=0.2*µm, z=3.5*µm, material=contact))
    design.add(Rectangle(position=(2*µm, 12*µm), width=16*µm, height=2*µm, 
                        depth=0.2*µm, z=3.5*µm, material=contact))
    
    # Etched features
    etch = Material(permittivity=1.0)
    for i in range(5):
        x_pos = 4*µm + i * 2.5*µm
        design.add(Circle(position=(x_pos, 7.5*µm), radius=0.4*µm, 
                         depth=1.2*µm, z=2.0*µm, material=etch))
    
    print(f"Design is 3D: {design.is_3d}")
    print("Showing complex multilayer device...")
    design.show()

if __name__ == "__main__":
    print("Enhanced Beamz 3D Visualization Demo")
    print("Showcasing modern styling, black outlines, and improved aesthetics")
    print("="*70)
    
    input("\nPress Enter to start Modern 3D Styling Demo...")
    demo_modern_3d_styling()
    
    input("\nPress Enter to start Complex Multilayer Demo...")
    demo_complex_multilayer()
    
    print("\n" + "="*70)
    print("Enhanced 3D Visualization Features:")
    print("🎨 Modern color palette with material-based consistency")
    print("⚫ Black outlines for clear structure definition")
    print("💡 Enhanced lighting with ambient, diffuse, and specular components") 
    print("📐 Professional typography and layout")
    print("🏠 Subtle ground plane for elevated structures")
    print("🔍 Rich hover information with material properties")
    print("📷 Optimized camera positioning for best viewing angle")
    print("🖱️  Interactive controls: rotate, zoom, pan")
    print("🌐 Web-based visualization (no window closing issues)")
    print("="*70) 