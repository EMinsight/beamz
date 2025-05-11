import gdspy
from itertools import groupby

def rgb_to_hex(r, g, b):
    """Convert RGB values to the hex color format used in get_random_color()."""
    return f'#{(r << 16) + (g << 8) + b:06x}'

def is_dark(color):
    """Check if a color is dark using relative luminance calculation. """
    color = color.lstrip('#')
    r = int(color[0:2], 16) / 255.0
    g = int(color[2:4], 16) / 255.0
    b = int(color[4:6], 16) / 255.0
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luminance < 0.5

def get_si_scale_and_label(value):
    """Convert a value to appropriate SI unit and return scale factor and label."""
    if value >= 1e-3: return 1e3, 'mm'
    elif value >= 1e-6: return 1e6, 'µm'
    elif value >= 1e-9: return 1e9, 'nm'
    else: return 1e12, 'pm'

def image_to_gds(image, output_file):
    """
    Convert a binary NumPy array to a GDS file.
    
    Parameters:
    - image: 2D NumPy array with binary values (0 or 1), shape (height, width)
    - output_file: String, the path to save the GDS file (e.g., 'output.gds')
    """
    # Get the dimensions of the image
    height, width = image.shape
    # Create a new GDS library with unit=1 (user unit) and precision=0.001
    lib = gdspy.GdsLibrary(unit=1, precision=1e-3)
    # Create a new cell named "TOP" to hold the geometry
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