from beamz import *
import numpy as np

# Define materials for different layers
# In a real photonic design, these would be your actual materials like silicon, oxide, etc.
LAYER_MATERIALS = {
    0: Material(permittivity=11.67),    # Silicon
    1: Material(permittivity=2.085),      # Silicon Oxide
    2: Material(permittivity=3.9)       # Silicon Nitride
}
# Define layer properties
LAYER_PROPERTIES = {
    1: {"depth": 0.22*µm, "z": 0.0},      # Silicon layer at bottom
    0: {"depth": 2.0*µm, "z": 0.22*µm},   # Oxide layer in middle
    2: {"depth": 0.4*µm, "z": 2.22*µm}    # Nitride layer on top
}
# Import a GDS file
print("Importing GDS file...")
gds_design = Design.import_gds("mmi.gds")

# Create a new 3D design with appropriate size
# Size should be determined by the imported design's dimensions
max_width = max(max(v[0] for v in s.vertices) for l in gds_design.layers.values() for s in l)
max_height = max(max(v[1] for v in s.vertices) for l in gds_design.layers.values() for s in l)
total_depth = sum(prop["depth"] for prop in LAYER_PROPERTIES.values())

design = Design(
    width=max_width*1.2,  # Add 20% margin
    height=max_height*1.2,
    depth=total_depth*1.2,
    material=Material(permittivity=1.0),  # Air cladding
    pml_size=None,
    auto_pml=False
)

# Add structures from each layer with proper materials and z-positions
print("\nImporting layers:")
for layer_num, structures in gds_design.layers.items():
    # Get material and properties for this layer
    material = LAYER_MATERIALS.get(layer_num, Material(permittivity=1.0))
    properties = LAYER_PROPERTIES.get(layer_num, {"depth": 0.5*µm, "z": 0.0})
    
    print(f"Layer {layer_num}")
    print(f"  - Material: {material.name if hasattr(material, 'name') else 'Unknown'}")
    print(f"  - Depth: {properties['depth']/µm:.2f} µm")
    print(f"  - Z-position: {properties['z']/µm:.2f} µm")
    print(f"  - Number of structures: {len(structures)}")
    
    # Add each structure from this layer
    for structure in structures:
        # Create polygon with proper material and z-position
        # Vertex ordering is now handled automatically by the Polygon class
        polygon = Polygon(
            vertices=structure.vertices,
            material=material,
            depth=properties["depth"],
            z=properties["z"]
        )
        design += polygon

print("\nVisualization:")
print(f"Total design size: {design.width/µm:.1f} x {design.height/µm:.1f} x {design.depth/µm:.1f} µm")
design.show()  # This will show the 3D visualization