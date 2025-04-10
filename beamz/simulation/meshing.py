import numpy as np
import matplotlib.pyplot as plt

class RegularGrid:
    """Takes in a design and resolution and returns a rasterized grid of that design."""
    def __init__(self, design, resolution):
        self.design = design
        self.resolution = resolution
        self.grid = None
        self.rasterize()

    def rasterize(self):
        """Rasterize the design into a grid with the given resolution."""
        width, height = self.design.width, self.design.height
        print(width, height)
        grid_width, grid_height = int(width / self.resolution), int(height / self.resolution)
        cell_size = self.resolution
        print(grid_width, grid_height)
        self.grid = np.zeros((grid_height, grid_width))

        # Iterate over each grid cell
        for i in range(grid_height):
            for j in range(grid_width):
                # Calculate the center of the grid cell
                x_center = (j + 0.5) * cell_size
                y_center = (i + 0.5) * cell_size
                # Sample 9 points within the grid cell for anti-aliasing
                samples = []
                for dx in [-0.25, 0, 0.25]:
                    for dy in [-0.25, 0, 0.25]:
                        x_sample = x_center + dx * cell_size
                        y_sample = y_center + dy * cell_size
                        print(x_sample, y_sample)
                        samples.append(self.design.get_material_value(x_sample, y_sample))
                # Average the samples to get the final value for the grid cell
                self.grid[i, j] = np.mean(samples)

    def print(self):
        """Print the rasterized grid."""
        if self.grid is not None: print(self.grid)
        else: print("Grid not rasterized yet.")

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