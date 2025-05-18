import numpy as np
import matplotlib.pyplot as plt
from beamz.helpers import display_status, create_rich_progress
from beamz.helpers import get_si_scale_and_label

class RegularGrid:
    """Takes in a design and resolution and returns a rasterized grid of that design."""
    def __init__(self, design, resolution):
        self.design = design
        self.resolution = resolution
        # Calculate grid dimensions in order to initialize the grids
        width, height = self.design.width, self.design.height
        grid_width, grid_height = int(width / self.resolution), int(height / self.resolution)
        # We have three grids of the same shape: permittivity, permeability, and conductivity
        self.permittivity = np.zeros((grid_height, grid_width))
        self.permeability = np.zeros((grid_height, grid_width))
        self.conductivity = np.zeros((grid_height, grid_width))
        self.__rasterize__()
        self.shape = self.permittivity.shape
        self.dx = self.resolution
        self.dy = self.resolution

    def __rasterize__(self):
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
        
        # Estimate dt for PML calculations (based on Courant stability criterion)
        # Using a conservative factor of 0.5 times the Courant limit
        c = 3e8  # Speed of light
        dt_estimate = 0.5 * self.resolution / (c * np.sqrt(2))
        
        # Process each point in the grid
        display_status("Rasterizing design as a regular grid...")
        
        # Use a single progress bar for the entire simulation
        with create_rich_progress() as progress:
            task = progress.add_task("Rasterizing design...", total=grid_height*grid_width)
            for i in range(grid_height):
                for j in range(grid_width):
                    # Get the center point
                    x_center = X_centers[i, j]
                    y_center = Y_centers[i, j]
                    # Create sample points around this center
                    x_samples = x_center + dx
                    y_samples = y_center + dy
                    # Get material values for all sample points, passing grid parameters
                    values = [self.design.get_material_value(x, y, dx=self.resolution, dt=dt_estimate) 
                            for x, y in zip(x_samples, y_samples)]
                    # Calculate the mean permittivity, permeability, and conductivity
                    permittivity = np.mean([value[0] for value in values])
                    permeability = np.mean([value[1] for value in values])
                    conductivity = np.mean([value[2] for value in values])
                    # Average the values
                    self.permittivity[i, j] = permittivity
                    self.permeability[i, j] = permeability
                    self.conductivity[i, j] = conductivity
                    # Update the progress bar
                    progress.update(task, advance=1)

    def show(self, field: str = "permittivity"):
        """Display the rasterized grid with properly scaled SI units."""
        if field == "permittivity": grid = self.permittivity
        elif field == "permeability": grid = self.permeability
        elif field == "conductivity": grid = self.conductivity
        if grid is not None:
            scale, unit = get_si_scale_and_label(max(self.design.width, self.design.height))
            # Calculate figure size based on grid dimensions
            grid_height, grid_width = grid.shape
            aspect_ratio = grid_width / grid_height
            base_size = 2.5  # Base size for the smaller dimension
            if aspect_ratio > 1: figsize = (base_size * aspect_ratio, base_size)
            else: figsize = (base_size, base_size / aspect_ratio)
            # PLot the figure
            plt.figure(figsize=figsize)
            plt.imshow(grid, origin='lower', cmap='Grays', extent=(0, self.design.width, 0, self.design.height))
            plt.colorbar(label=field)
            plt.title('Rasterized Design Grid')
            plt.xlabel(f'X ({unit})')
            plt.ylabel(f'Y ({unit})')
            plt.gca().xaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
            plt.gca().yaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
            plt.tight_layout()
            plt.show()
        else: print("Grid not rasterized yet.")