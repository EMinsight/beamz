import sys
import numpy as np
import gdspy
from itertools import groupby

def progress_bar(progress:int, total:int, length:int=50):
    """Print a progress bar to the console.

    Args:
        progress (int): The current progress.
        total (int): The total progress.
        length (int): The length of the progress bar.
    """
    percent = 100 * (progress / float(total))
    filled_length = int(length * progress // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r|{bar}| {percent:.2f}%')
    sys.stdout.flush()

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
                    layer=0  # Place all rectangles on layer 0
                )
                cell.add(rect)
    
    # Write the library to a GDS file
    lib.write_gds(output_file)
    print(f"GDS file saved as '{output_file}'")

