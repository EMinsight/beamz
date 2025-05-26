import os
import pytest
from beamz.design.io import import_gds, GDSDesign
from beamz.design.structures import Polygon, Rectangle # Imported Rectangle
from beamz.design import Design # For beamz.design.Design

# Define the path to the test GDS file
# Assuming the test runs from the repository root, or paths are relative to it.
# The GDS file was created by create_gds_script.py in tests/test_data/
TEST_GDS_FILE = "tests/test_data/test_import.gds"

def test_import_gds_functionality():
    """
    Tests the GDS import functionality:
    - Loads a GDS file created by a helper script.
    - Checks if the returned object is a GDSDesign instance.
    - Verifies layer numbers and polygon presence.
    - Validates vertex data for polygons on specific layers.
    - Checks integration with the main Design object.
    """
    assert os.path.exists(TEST_GDS_FILE), f"Test GDS file not found: {TEST_GDS_FILE}"

    # Call import_gds to load the test GDS file
    gds_design_instance = import_gds(TEST_GDS_FILE)

    # Assert that the returned object is an instance of GDSDesign
    assert isinstance(gds_design_instance, GDSDesign), "import_gds should return a GDSDesign instance."

    # Assert that the gds_design.layers dictionary contains the correct layer numbers
    expected_layers = {1, 2}
    imported_layers = set(gds_design_instance.layers.keys())
    assert imported_layers == expected_layers, \
        f"Expected layers {expected_layers}, but got {imported_layers}."

    # For each layer, assert that the list of polygons is not empty
    for layer_num in expected_layers:
        assert layer_num in gds_design_instance.layers, f"Layer {layer_num} not found in imported design."
        assert len(gds_design_instance.layers[layer_num]) > 0, f"Layer {layer_num} should contain polygons."
        assert isinstance(gds_design_instance.layers[layer_num][0], Polygon), \
            f"Objects in layer {layer_num} should be Polygon instances."

    # Assert vertex data for layer 1
    # Vertices defined in create_gds_script.py for layer 1: [(0,0), (1,0), (1,1), (0,1)]
    # GDS units were 1e-6 (micrometers). beamz.Polygon stores them as is.
    expected_vertices_layer1 = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    imported_poly_layer1 = gds_design_instance.layers[1][0]
    assert imported_poly_layer1.vertices == expected_vertices_layer1, \
        f"Vertex mismatch for layer 1. Expected {expected_vertices_layer1}, got {imported_poly_layer1.vertices}."

    # Assert vertex data for layer 2
    # Vertices defined in create_gds_script.py for layer 2: [(2,2), (3,2), (3,3), (2,3)]
    expected_vertices_layer2 = [(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)]
    imported_poly_layer2 = gds_design_instance.layers[2][0]
    assert imported_poly_layer2.vertices == expected_vertices_layer2, \
        f"Vertex mismatch for layer 2. Expected {expected_vertices_layer2}, got {imported_poly_layer2.vertices}."

    # Create a beamz.design.Design object
    # Initialize with auto_pml=False to prevent PML visualization structures from being added by default.
    # This way, the design starts with 1 structure (the background).
    main_design = Design(auto_pml=False)

    # Iterate through the imported gds_design.layers and add all polygons to the main_design
    total_polygons_added = 0
    for layer_num, polygons_in_layer in gds_design_instance.layers.items():
        for poly in polygons_in_layer:
            # Optionally assign material or other properties if needed before adding
            # For this test, directly adding the polygon structure is sufficient
            main_design += poly 
            total_polygons_added += 1
    
    # Assert that the main_design.structures list now contains the correct number of polygons
    # It should be the initial background structure (1) + the polygons added from GDS (total_polygons_added)
    expected_structure_count = 1 + total_polygons_added
    assert len(main_design.structures) == expected_structure_count, \
        f"Expected {expected_structure_count} structures in main_design, but got {len(main_design.structures)}."
    # This checks that we indeed processed 2 polygons from GDS
    assert total_polygons_added == 2, \
        f"Expected to add 2 polygons from GDS, but added {total_polygons_added}. Check GDS file or import logic."

    # Verify the type of the added structures in main_design.structures
    # The first one is the background, the rest should be the imported polygons.
    assert isinstance(main_design.structures[0], Rectangle) # Background
    for i in range(total_polygons_added):
        assert isinstance(main_design.structures[1 + i], Polygon), \
            f"Structure {1+i} in main_design should be a Polygon instance, but got {type(main_design.structures[1+i])}."

    print(f"Test test_import_gds_functionality passed successfully. Added {total_polygons_added} polygons to design.")

# To run this test (assuming pytest is installed and in the root of the repo):
# pytest tests/design/test_io.py
# Or, if pytest.ini is set up:
# pytest
