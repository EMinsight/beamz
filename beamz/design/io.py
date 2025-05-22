import gdspy
from itertools import groupby
import os
import numpy as np

def import_file(input_file):
    """Import a file with format detection based on file extension."""
    _, ext = os.path.splitext(input_file)
    ext = ext.lower()
    if ext == '.gds': return import_gds_as_polygons(input_file)
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

def import_gds_as_polygons(gds_file):
    """Import a GDS file and return a list of polygons."""
    gds_lib = gdspy.GdsLibrary(infile=gds_file)
    cells = gds_lib.cells # get all cells from the library
    polygons = []
    for cell_name, cell in cells.items():
        for polygon in cell.get_polygons():
            polygons.append(gdspy.Polygon(polygon))
    print(f"Imported {len(polygons)} polygons from '{gds_file}'")
    return polygons

def export_bin_numpy_as_gds(image, output_file):
    """Convert a binary NumPy array to a GDS file."""
    height, width = image.shape
    lib = gdspy.GdsLibrary(unit=1, precision=1e-3)
    cell = lib.new_cell("TOP")
    # Process each row of the image
    for i in range(height):
        # Map y-coordinates: row 0 (top) to y=height-1, row height-1 (bottom) to y=0
        y_bottom = height - 1 - i
        y_top = height - i
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
                    (j_end + 1, y_top),
                    layer=0) # Place all rectangles on layer 0
                cell.add(rect)
    # Write the library to a GDS file
    lib.write_gds(output_file)
    print(f"GDS file saved as '{output_file}'")