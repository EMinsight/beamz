from beamz.structures import image_to_gds
import numpy as np

# Sample 3x3 binary image
sample_image = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=np.uint8)

# Convert and export to GDS
image_to_gds(sample_image, "output.gds")