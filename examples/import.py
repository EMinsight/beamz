# import a gds file
from beamz.design import Design, Material # Polygon is not directly used here anymore for instantiation
from beamz.design.io import import_gds # Updated import
from beamz.const import µm

# Create a design
# The overall design material can be a default or background material.
# Specific materials can be assigned to polygons based on layer info.
design = Design(width=5*µm, height=5*µm, material=Material(1), pml_size=0)

# Import gds-file using the new import_gds function
gds_design = import_gds("test.gds") # "test.gds" is a placeholder

# Add the polygons from the GDSDesign object to the main design
if gds_design and gds_design.layers:
    for layer_number, polygons_in_layer in gds_design.layers.items():
        # Example: Assign different materials based on layer number
        # material_for_layer = Material(permittivity=1.444**2 + layer_number * 0.1) 
        print(f"Processing layer: {layer_number} with {len(polygons_in_layer)} polygons")
        for gds_polygon in polygons_in_layer:
            # Scale the polygon (assuming original GDS units need scaling to simulation units)
            scaled_polygon = gds_polygon.scale(1*µm)
            
            # Assign a material to the polygon before adding it to the design.
            # For this example, we'll use a default material, but you can use material_for_layer.
            # scaled_polygon.material = material_for_layer # Uncomment to use layer-specific material
            scaled_polygon.material = Material(permittivity=1.444**2) # Example material
            
            design += scaled_polygon
else:
    print("No layers or polygons found in the GDS file.")

# Show the design
design.show()