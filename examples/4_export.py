# create a numpy array of 10x10 pixels
import numpy as np

image = np.zeros((5, 5))

# set the center pixel to 1
image[2, 2] = 1
# set the corner pixels to 1
image[0, 0] = 1
image[0, 4] = 1
image[4, 0] = 1
image[4, 4] = 1

# save the image to a gds file
from beamz.design.io import export_bin_numpy_as_gds
export_bin_numpy_as_gds(image, "test.gds")