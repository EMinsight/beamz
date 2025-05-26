import gdspy
from itertools import groupby
import os
import numpy as np
# Import the Polygon class from structures.py
from beamz.design.structures import Polygon
from beamz.const import µm
# TODO: Add support for gltf for 3D models!

class GDSDesign:
    """Represents a GDS design with multiple layers."""
    def __init__(self):
        """Initializes a GDSDesign object.

        The `layers` attribute is an empty dictionary that will store layer numbers
        (integers) as keys and lists of `beamz.design.structures.Polygon`
        objects as values.
        """
        self.layers: dict[int, list[Polygon]] = {}

def import_file(input_file):
    """Import a file with format detection based on file extension."""
    _, ext = os.path.splitext(input_file)
    ext = ext.lower()
    if ext == '.gds': return import_gds(input_file) # Updated to call import_gds
    else: raise ValueError(f"Unsupported file extension: {ext}")

def export_file(data, output_file):
    """Export data to a file with format detection based on file extension."""
    _, ext = os.path.splitext(output_file)
    ext = ext.lower()
    if ext == '.gds':
        if isinstance(data, gdspy.Polygon) or isinstance(data, gdspy.PolygonSet):
            export_polygon_as_gds(data, output_file)
        elif isinstance(data, np.ndarray):
            if data.ndim == 2 and np.array_equal(data, data.astype(bool)):
                export_bin_numpy_as_gds(data, output_file)
            else: raise ValueError("Numpy array must be 2D binary array for GDS export")
        else: raise ValueError(f"Unsupported data type for GDS export: {type(data)}")
    else: raise ValueError(f"Unsupported file extension: {ext}")

def export_polygon_as_gds(polygon, output_file):
    """Export a polygon to a GDS file."""
    lib = gdspy.GdsLibrary(unit=1, precision=1e-3)
    cell = lib.new_cell("main")
    cell.add(polygon)
    lib.write_gds(output_file)
    print(f"GDS file saved as '{output_file}'")

def import_gds(gds_file: str) -> GDSDesign:
    """Import a GDS file and return a GDSDesign object.

    Args:
        gds_file: Path to the GDS file.

    Returns:
        A GDSDesign object populated with polygons from the GDS file.
    """
    gds_lib = gdspy.GdsLibrary(infile=gds_file)
    design = GDSDesign()  # Create GDSDesign instance
    
    cells = gds_lib.cells  # Get all cells from the library
    
    total_polygons_imported = 0
    
    for _cell_name, cell in cells.items():
        # Get polygons by spec, which returns a dict: {(layer, datatype): [poly1_points, poly2_points,...]}
        # polyN_points is a numpy array of vertices, e.g., [[x1, y1], [x2, y2], ...]
        gdspy_polygons_by_spec = cell.get_polygons(by_spec=True)
        
        for (layer_num, _datatype), list_of_polygon_points in gdspy_polygons_by_spec.items():
            if layer_num not in design.layers:
                design.layers[layer_num] = []
            
            for polygon_points in list_of_polygon_points:
                # Convert gdspy.Polygon points (numpy array) to list of tuples for beamz.Polygon
                vertices = [(point[0], point[1]) for point in polygon_points]
                beamz_polygon = Polygon(vertices=vertices)
                design.layers[layer_num].append(beamz_polygon)
                total_polygons_imported += 1
                
    print(f"Imported {total_polygons_imported} polygons from '{gds_file}' into GDSDesign object.")
    return design

def export_bin_numpy_as_gds(image, output_file, scale=µm):
    """Convert a binary NumPy array to a GDS file.
    
    Args:
        image: Binary NumPy array representation of the image
        output_file: Output GDS file path
        scale: Scale factor for the dimensions (default: µm)
    """
    height, width = image.shape
    lib = gdspy.GdsLibrary(unit=1, precision=1e-3)
    cell = lib.new_cell("TOP")
    # Process each row of the image
    for i in range(height):
        # Map y-coordinates: row 0 (top) to y=height-1, row height-1 (bottom) to y=0
        y_bottom = (height - 1 - i)
        y_top = (height - i)
        # Get the current row
        row = image[i, :]
        # Find runs of consecutive 1's using groupby
        for key, group in groupby(enumerate(row), key=lambda x: x[1]):
            if key == 1:  # If the group is a run of 1's
                indices = [x[0] for x in group]
                j_start = indices[0]  # Start of the run
                j_end = indices[-1]   # End of the run
                # Create a rectangle from (j_start, y_bottom) to (j_end + 1, y_top)
                rect = gdspy.Rectangle(
                    (j_start, y_bottom),
                    ((j_end + 1), y_top),
                    layer=0) # Place all rectangles on layer 0
                cell.add(rect)
    # Write the library to a GDS file
    lib.write_gds(output_file)
    print(f"GDS file saved as '{output_file}'")