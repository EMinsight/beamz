import numpy as np
import matplotlib.pyplot as plt
from beamz.design.structures import Rectangle
from beamz.helpers import create_rich_progress

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
        """Painters algorithm to rasterize the design into a grid using super-sampling
        by utilizing the ordered nature of the structures and their bounding boxes."
        We iterate through the sorted list of objects:
        1. First, draw the background layer without any anti-aliasing or boundary box consideration.
        2. Then take the boundary box of the next object and create a mask for the material arrays.
        3. Then use super-sampling over that boundary box to draw this object.
        4. Do this until all objects are drawn.
        """
        width, height = self.design.width, self.design.height
        grid_width, grid_height = int(width / self.resolution), int(height / self.resolution)
        cell_size = self.resolution
        
        # Create grid of cell centers
        x_centers = (np.arange(grid_width) + 0.5) * cell_size
        y_centers = (np.arange(grid_height) + 0.5) * cell_size
        X_centers, Y_centers = np.meshgrid(x_centers, y_centers)
        
        # Create sample offsets for super-sampling (9 points per cell)
        offsets = np.array([-0.25, 0, 0.25]) * cell_size
        dx, dy = np.meshgrid(offsets, offsets)
        dx = dx.flatten()
        dy = dy.flatten()
        num_samples = len(dx)
        
        # Estimate dt for PML calculations
        c = 3e8  # Speed of light
        dt_estimate = 0.5 * self.resolution / (c * np.sqrt(2))
        
        # Initialize material grids with vacuum
        permittivity = np.ones((grid_height, grid_width))
        permeability = np.ones((grid_height, grid_width))
        conductivity = np.zeros((grid_height, grid_width))
        
        # Process structures in original order (background first, top structure last)
        with create_rich_progress() as progress:
            task = progress.add_task("Rasterizing structures...", total=len(self.design.structures))
            
            # Step 1: Process the first structure (background) without anti-aliasing
            if len(self.design.structures) > 0:
                background = self.design.structures[0]
                if hasattr(background, 'material') and background.material is not None:
                    # Fill the entire grid with background material
                    permittivity.fill(background.material.permittivity)
                    permeability.fill(background.material.permeability)
                    conductivity.fill(background.material.conductivity)
                progress.update(task, advance=1)
            
            # Step 2-4: Process each remaining structure using bounding boxes
            for idx in range(1, len(self.design.structures)):
                structure = self.design.structures[idx]
                # Skip PML visualization structures
                if hasattr(structure, 'is_pml') and structure.is_pml:
                    progress.update(task, advance=1)
                    continue
                # Skip if no material
                if not hasattr(structure, 'material') or structure.material is None:
                    progress.update(task, advance=1)
                    continue
                # Get material properties
                mat_perm = structure.material.permittivity
                mat_permb = structure.material.permeability
                mat_cond = structure.material.conductivity
                
                # Get structure's bounding box
                try:
                    bbox = structure.get_bounding_box()
                    if bbox is None: raise AttributeError("Bounding box is None")
                    min_x, min_y, max_x, max_y = bbox
                    # Convert to grid indices with slight padding for anti-aliasing
                    min_i = max(0, int(min_y / cell_size) - 1)
                    min_j = max(0, int(min_x / cell_size) - 1)
                    max_i = min(grid_height, int(np.ceil(max_y / cell_size)) + 1)
                    max_j = min(grid_width, int(np.ceil(max_x / cell_size)) + 1)
                    # Create a mask for this structure's bounding box
                    # (0 = outside structure, 0-1 = partial coverage, 1 = fully inside)
                    mask = np.zeros((max_i - min_i, max_j - min_j))
                    # Determine structure's type for point-in-structure testing
                    if hasattr(structure, 'point_in_polygon'):
                        contains_func = lambda x, y: structure.point_in_polygon(x, y)
                    elif isinstance(structure, Rectangle) and all(v == 0 for v in [
                            structure.vertices[0][0] - structure.position[0],
                            structure.vertices[0][1] - structure.position[1]]):
                        # Fast path for axis-aligned rectangles
                        contains_func = lambda x, y: (
                            structure.position[0] <= x < structure.position[0] + structure.width and
                            structure.position[1] <= y < structure.position[1] + structure.height
                        )
                    elif hasattr(structure, 'radius'):  # Circle
                        contains_func = lambda x, y: (
                            np.hypot(x - structure.position[0], y - structure.position[1]) <= structure.radius
                        )
                    elif hasattr(structure, 'inner_radius') and hasattr(structure, 'outer_radius'):  # Ring
                        contains_func = lambda x, y: (
                            structure.inner_radius <= 
                            np.hypot(x - structure.position[0], y - structure.position[1]) <= 
                            structure.outer_radius
                        )
                    else:
                        # Fallback to using design's material check
                        contains_func = lambda x, y: any(
                            val != def_val for val, def_val in zip(
                                self.design.get_material_value(x, y),
                                [1.0, 1.0, 0.0]  # Default values
                            )
                        )
                    
                    # Super-sample within the bounding box to create the mask
                    for i_rel in range(max_i - min_i):
                        for j_rel in range(max_j - min_j):
                            i, j = i_rel + min_i, j_rel + min_j
                            # Get cell center
                            x_center = X_centers[i, j]
                            y_center = Y_centers[i, j]
                            # Count samples inside structure
                            samples_inside = 0
                            for k in range(num_samples):
                                x_sample = x_center + dx[k]
                                y_sample = y_center + dy[k]
                                
                                if contains_func(x_sample, y_sample):
                                    samples_inside += 1
                            
                            # Calculate coverage (0-1)
                            mask[i_rel, j_rel] = samples_inside / num_samples
                    
                    # Apply the mask to update material values
                    for i_rel in range(max_i - min_i):
                        for j_rel in range(max_j - min_j):
                            i, j = i_rel + min_i, j_rel + min_j
                            
                            # Skip cells with no coverage
                            if mask[i_rel, j_rel] <= 0:
                                continue
                            
                            # Blend materials based on coverage
                            blend_factor = mask[i_rel, j_rel]
                            
                            # Update material values using painter's algorithm
                            permittivity[i, j] = permittivity[i, j] * (1 - blend_factor) + mat_perm * blend_factor
                            permeability[i, j] = permeability[i, j] * (1 - blend_factor) + mat_permb * blend_factor
                            conductivity[i, j] = conductivity[i, j] * (1 - blend_factor) + mat_cond * blend_factor
                
                except (AttributeError, TypeError) as e:
                    print(f"Warning: Structure {type(structure)} doesn't have proper bounding box: {e}")
                
                progress.update(task, advance=1)
        
        # Process PML separately 
        with create_rich_progress() as progress:
            task = progress.add_task("Processing PML...", total=len(self.design.boundaries))
            
            # Process each PML boundary
            for boundary in self.design.boundaries:
                # Get the PML region
                if hasattr(boundary, 'position'):
                    if hasattr(boundary, 'width') and hasattr(boundary, 'height'):
                        # Rectangular PML
                        min_i = max(0, int(boundary.position[1] / cell_size))
                        min_j = max(0, int(boundary.position[0] / cell_size))
                        max_i = min(grid_height, int(np.ceil((boundary.position[1] + boundary.height) / cell_size)))
                        max_j = min(grid_width, int(np.ceil((boundary.position[0] + boundary.width) / cell_size)))
                    elif hasattr(boundary, 'radius'):
                        # Corner PML
                        center_i = int(boundary.position[1] / cell_size)
                        center_j = int(boundary.position[0] / cell_size)
                        radius = int(np.ceil(boundary.radius / cell_size))
                        
                        min_i = max(0, center_i - radius)
                        max_i = min(grid_height, center_i + radius + 1)
                        min_j = max(0, center_j - radius)
                        max_j = min(grid_width, center_j + radius + 1)
                    else:
                        # Skip if we can't determine the region
                        progress.update(task, advance=1)
                        continue
                else:
                    # Skip if no position
                    progress.update(task, advance=1)
                    continue
                
                # Process cells within the PML region
                for i in range(min_i, max_i):
                    for j in range(min_j, max_j):
                        x_center = X_centers[i, j]
                        y_center = Y_centers[i, j]
                        
                        # Super-sample for PML conductivity
                        pml_values = []
                        for k in range(num_samples):
                            x_sample = x_center + dx[k]
                            y_sample = y_center + dy[k]
                            
                            pml_conductivity = boundary.get_conductivity(
                                x_sample, y_sample,
                                dx=self.resolution,
                                dt=dt_estimate,
                                eps_avg=permittivity[i, j]
                            )
                            
                            pml_values.append(pml_conductivity)
                        
                        # Add average PML conductivity to current value
                        conductivity[i, j] += np.mean(pml_values)
                
                progress.update(task, advance=1)
        
        # Assign final arrays to class instance
        self.permittivity = permittivity
        self.permeability = permeability
        self.conductivity = conductivity
        
        # Log material ranges for debugging
        print(f"Permittivity range: [{np.min(permittivity):.2f}, {np.max(permittivity):.2f}]")
        print(f"Permeability range: [{np.min(permeability):.2f}, {np.max(permeability):.2f}]")
        print(f"Conductivity range: [{np.min(conductivity):.2e}, {np.max(conductivity):.2e}]")

    def show(self, field: str = "permittivity"):
        """Display the rasterized grid with properly scaled SI units."""
        if field == "permittivity":
            grid = self.permittivity
        elif field == "permeability":
            grid = self.permeability
        elif field == "conductivity":
            grid = self.conductivity
        if grid is not None:
            # Determine appropriate SI unit and scale
            max_dim = max(self.design.width, self.design.height)
            if max_dim >= 1e-3: scale, unit = 1e3, 'mm'
            elif max_dim >= 1e-6: scale, unit = 1e6, 'µm'
            elif max_dim >= 1e-9: scale, unit = 1e9, 'nm'
            else: scale, unit = 1e12, 'pm'

            # Calculate figure size based on grid dimensions
            grid_height, grid_width = grid.shape
            aspect_ratio = grid_width / grid_height
            base_size = 2.5  # Base size for the smaller dimension
            if aspect_ratio > 1: figsize = (base_size * aspect_ratio, base_size)
            else: figsize = (base_size, base_size / aspect_ratio)

            plt.figure(figsize=figsize)
            plt.imshow(grid, origin='lower', cmap='Grays', extent=(0, self.design.width, 0, self.design.height))
            plt.colorbar(label=field)
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