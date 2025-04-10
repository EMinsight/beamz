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
        """Display the rasterized grid with properly scaled SI units."""
        if self.grid is not None:
            # Determine appropriate SI unit and scale
            max_dim = max(self.design.width, self.design.height)
            if max_dim >= 1e-3:
                scale, unit = 1e3, 'mm'
            elif max_dim >= 1e-6:
                scale, unit = 1e6, 'µm'
            elif max_dim >= 1e-9:
                scale, unit = 1e9, 'nm'
            else:
                scale, unit = 1e12, 'pm'

            # Calculate figure size based on grid dimensions
            grid_height, grid_width = self.grid.shape
            aspect_ratio = grid_width / grid_height
            base_size = 2.5  # Base size for the smaller dimension
            if aspect_ratio > 1: figsize = (base_size * aspect_ratio, base_size)
            else: figsize = (base_size, base_size / aspect_ratio)

            plt.figure(figsize=figsize)
            plt.imshow(self.grid, origin='lower', cmap='Grays', 
                      extent=(0, self.design.width, 0, self.design.height))
            plt.colorbar(label='Permittivity')
            plt.title('Rasterized Design Grid')
            plt.xlabel(f'X ({unit})')
            plt.ylabel(f'Y ({unit})')

            # Update tick labels with scaled values
            plt.gca().xaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
            plt.gca().yaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')

            plt.tight_layout()
            plt.show()
        else:
            print("Grid not rasterized yet.")



# RectilinearGrid

# ================================ UnstructuredGrid