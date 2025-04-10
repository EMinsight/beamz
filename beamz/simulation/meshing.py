import numpy as np
import matplotlib.pyplot as plt

class RegularGrid:
    """Takes in a design and resolution and returns a rasterized grid of that design."""
    def __init__(self, design, resolution):
        self.design = design
        self.resolution = resolution
        self.grid = None
        self.rasterize()
        self.shape = self.grid.shape

    def rasterize(self):
        """Rasterize the design into a grid with the given resolution using fully vectorized operations."""
        width, height = self.design.width, self.design.height
        grid_width, grid_height = int(width / self.resolution), int(height / self.resolution)
        cell_size = self.resolution
        # Create grid of cell centers
        x_centers = (np.arange(grid_width) + 0.5) * cell_size
        y_centers = (np.arange(grid_height) + 0.5) * cell_size
        X_centers, Y_centers = np.meshgrid(x_centers, y_centers)
        # Create sample offsets for anti-aliasing (9 points per cell)
        offsets = np.array([-0.25, 0, 0.25]) * cell_size
        dx, dy = np.meshgrid(offsets, offsets)
        dx = dx.flatten()
        dy = dy.flatten()
        # Create arrays of all sample points
        X_samples = X_centers[:, :, np.newaxis] + dx
        Y_samples = Y_centers[:, :, np.newaxis] + dy
        # Reshape for vectorized material value calculation
        X_flat = X_samples.reshape(-1)
        Y_flat = Y_samples.reshape(-1)
        # Get material values for all points at once
        material_values = np.array([self.design.get_material_value(x, y) for x, y in zip(X_flat, Y_flat)])
        # Reshape back to grid structure with 9 samples per cell
        material_values = material_values.reshape(grid_height, grid_width, 9)
        # Average the 9 samples for each cell
        self.grid = np.mean(material_values, axis=2)

    def show(self):
        """Display the rasterized grid."""
        if self.grid is not None:
            plt.imshow(self.grid, origin='lower', cmap='viridis', extent=(0, self.design.width, 0, self.design.height))
            plt.colorbar(label='Permittivity')
            plt.title('Rasterized Design Grid')
            plt.xlabel('Width')
            plt.ylabel('Height')
            plt.show()
        else:
            print("Grid not rasterized yet.")



# RectilinearGrid

# ================================ UnstructuredGrid