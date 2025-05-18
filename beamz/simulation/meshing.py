import numpy as np
import matplotlib.pyplot as plt
from beamz.helpers import display_status, create_rich_progress
from beamz.helpers import get_si_scale_and_label
from beamz.design.structures import Rectangle

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
        """Rasterize the design into a grid with the given resolution using a painter's algorithm."""
        width, height = self.design.width, self.design.height
        grid_width, grid_height = int(width / self.resolution), int(height / self.resolution)
        cell_size = self.resolution

        # Create grid of cell centers
        x_centers = (np.arange(grid_width) + 0.5) * cell_size
        y_centers = (np.arange(grid_height) + 0.5) * cell_size
        X_centers, Y_centers = np.meshgrid(x_centers, y_centers)
        
        # Estimate dt for PML calculations (based on Courant stability criterion)
        # Using a conservative factor of 0.5 times the Courant limit
        c = 3e8  # Speed of light
        dt_estimate = 0.5 * self.resolution / (c * np.sqrt(2))
        
        # Initialize grid with background material (air)
        permittivity = np.ones((grid_height, grid_width))
        permeability = np.ones((grid_height, grid_width))
        conductivity = np.zeros((grid_height, grid_width))
        
        # Create mask to track which cells have been filled
        filled_mask = np.zeros((grid_height, grid_width), dtype=bool)
        
        # Process structures from top to bottom (reversed order)
        with create_rich_progress() as progress:
            task = progress.add_task("Rasterizing structures...", total=len(self.design.structures))
            
            for structure in reversed(self.design.structures):
                # Skip PML visualization structures
                if hasattr(structure, 'is_pml') and structure.is_pml:
                    progress.update(task, advance=1)
                    continue
                    
                # Get structure's bounding box
                try:
                    bbox = structure.get_bounding_box()
                    if bbox is None:
                        raise AttributeError("Bounding box is None")
                        
                    min_x, min_y, max_x, max_y = bbox
                    
                    # Convert to grid indices
                    min_i = max(0, int(min_y / self.resolution))
                    min_j = max(0, int(min_x / self.resolution))
                    max_i = min(grid_height, int(np.ceil(max_y / self.resolution)))
                    max_j = min(grid_width, int(np.ceil(max_x / self.resolution)))
                    
                    # Only process cells within bounding box that aren't already filled
                    for i in range(min_i, max_i):
                        for j in range(min_j, max_j):
                            if filled_mask[i, j]:
                                continue
                                
                            # Get cell center
                            x = X_centers[i, j]
                            y = Y_centers[i, j]
                            
                            # Check if point is inside structure
                            if hasattr(structure, 'material'):
                                # Direct check for Rectangle without rotation
                                if isinstance(structure, Rectangle) and not any(angle != 0 for angle in [structure.vertices[0][0] - structure.position[0], 
                                                            structure.vertices[0][1] - structure.position[1]]):
                                    # Fast path for axis-aligned rectangles
                                    if (structure.position[0] <= x < structure.position[0] + structure.width and
                                        structure.position[1] <= y < structure.position[1] + structure.height):
                                        permittivity[i, j] = structure.material.permittivity
                                        permeability[i, j] = structure.material.permeability
                                        conductivity[i, j] = structure.material.conductivity
                                        filled_mask[i, j] = True
                                # For polygons and shapes
                                elif hasattr(structure, '_point_in_polygon'):
                                    if structure._point_in_polygon(x, y, structure.vertices):
                                        permittivity[i, j] = structure.material.permittivity
                                        permeability[i, j] = structure.material.permeability
                                        conductivity[i, j] = structure.material.conductivity
                                        filled_mask[i, j] = True
                                # For circles
                                elif hasattr(structure, 'radius'):
                                    distance = np.hypot(x - structure.position[0], y - structure.position[1])
                                    if distance <= structure.radius:
                                        permittivity[i, j] = structure.material.permittivity
                                        permeability[i, j] = structure.material.permeability
                                        conductivity[i, j] = structure.material.conductivity
                                        filled_mask[i, j] = True
                                # For rings
                                elif hasattr(structure, 'inner_radius') and hasattr(structure, 'outer_radius'):
                                    distance = np.hypot(x - structure.position[0], y - structure.position[1])
                                    if structure.inner_radius <= distance <= structure.outer_radius:
                                        permittivity[i, j] = structure.material.permittivity
                                        permeability[i, j] = structure.material.permeability
                                        conductivity[i, j] = structure.material.conductivity
                                        filled_mask[i, j] = True
                
                except (AttributeError, TypeError) as e:
                    print(f"Warning: Structure {type(structure)} doesn't have proper bounding box: {e}")
                    # Fallback for structures without proper bounding box implementation
                    # This is the old method but only for this specific structure
                    for i in range(grid_height):
                        for j in range(grid_width):
                            if filled_mask[i, j]:
                                continue
                                
                            x = X_centers[i, j]
                            y = Y_centers[i, j]
                            
                            # Use the design's get_material_value 
                            if hasattr(structure, 'material') and hasattr(structure, 'position'):
                                # Check if the point is within this structure
                                if isinstance(structure, Rectangle):
                                    # Fast path for rectangles
                                    if (structure.position[0] <= x < structure.position[0] + structure.width and
                                        structure.position[1] <= y < structure.position[1] + structure.height):
                                        permittivity[i, j] = structure.material.permittivity
                                        permeability[i, j] = structure.material.permeability
                                        conductivity[i, j] = structure.material.conductivity
                                        filled_mask[i, j] = True
                                        
                            # Fallback to design's method
                            if not filled_mask[i, j]:
                                material_values = self.design.get_material_value(x, y)
                                if material_values[0] != 1.0 or material_values[1] != 1.0 or material_values[2] != 0.0:
                                    permittivity[i, j] = material_values[0]
                                    permeability[i, j] = material_values[1]
                                    conductivity[i, j] = material_values[2]
                                    filled_mask[i, j] = True
                
                progress.update(task, advance=1)
        
        # Process PML separately (only add conductivity)
        with create_rich_progress() as progress:
            task = progress.add_task("Processing PML...", total=len(self.design.boundaries))
            
            # Create a mask for PML regions to avoid checking all grid points
            pml_mask = np.zeros((grid_height, grid_width), dtype=bool)
            
            # Mark PML regions based on boundary definitions
            for boundary in self.design.boundaries:
                if hasattr(boundary, 'position') and hasattr(boundary, 'width') and hasattr(boundary, 'height'):
                    # Rectangular PML
                    min_i = max(0, int(boundary.position[1] / self.resolution))
                    min_j = max(0, int(boundary.position[0] / self.resolution))
                    max_i = min(grid_height, int((boundary.position[1] + boundary.height) / self.resolution))
                    max_j = min(grid_width, int((boundary.position[0] + boundary.width) / self.resolution))
                    pml_mask[min_i:max_i, min_j:max_j] = True
                elif hasattr(boundary, 'position') and hasattr(boundary, 'radius'):
                    # Corner PML
                    center_i = int(boundary.position[1] / self.resolution)
                    center_j = int(boundary.position[0] / self.resolution)
                    radius_cells = int(boundary.radius / self.resolution)
                    
                    # Simple circle mask
                    i_indices, j_indices = np.ogrid[-radius_cells:radius_cells+1, -radius_cells:radius_cells+1]
                    circle_mask = i_indices**2 + j_indices**2 <= radius_cells**2
                    
                    # Apply mask at the correct position with bounds checking
                    min_i = max(0, center_i - radius_cells)
                    max_i = min(grid_height, center_i + radius_cells + 1)
                    min_j = max(0, center_j - radius_cells)
                    max_j = min(grid_width, center_j + radius_cells + 1)
                    
                    # Calculate indices for the cropped mask
                    mask_min_i = max(0, -center_i + radius_cells)
                    mask_max_i = mask_min_i + (max_i - min_i)
                    mask_min_j = max(0, -center_j + radius_cells)
                    mask_max_j = mask_min_j + (max_j - min_j)
                    
                    # Apply mask at the correct position with bounds checking
                    cropped_mask = circle_mask[mask_min_i:mask_max_i, mask_min_j:mask_max_j]
                    pml_mask[min_i:max_i, min_j:max_j] |= cropped_mask
                
                progress.update(task, advance=1)
            
            # Apply PML conductivity where needed
            pml_task = progress.add_task("Applying PML conductivity...", 
                                         total=np.sum(pml_mask))
            
            # Only process cells within PML regions
            pml_indices = np.where(pml_mask)
            for idx in range(len(pml_indices[0])):
                i, j = pml_indices[0][idx], pml_indices[1][idx]
                
                x = X_centers[i, j]
                y = Y_centers[i, j]
                
                # Apply all PML boundaries
                pml_conductivity = 0.0
                for boundary in self.design.boundaries:
                    pml_conductivity += boundary.get_conductivity(
                        x, y, 
                        dx=self.resolution, 
                        dt=dt_estimate, 
                        eps_avg=permittivity[i, j])
                # Add PML conductivity to existing conductivity
                conductivity[i, j] += pml_conductivity
                progress.update(pml_task, advance=1)
        
        # Make sure to assign the final arrays to the class instance variables
        self.permittivity = permittivity
        self.permeability = permeability
        self.conductivity = conductivity
        
        # Log min/max values for debugging
        print(f"Permittivity range: [{np.min(permittivity):.2f}, {np.max(permittivity):.2f}]")
        print(f"Permeability range: [{np.min(permeability):.2f}, {np.max(permeability):.2f}]")
        print(f"Conductivity range: [{np.min(conductivity):.2e}, {np.max(conductivity):.2e}]")

    def show(self, field: str = "permittivity"):
        """Display the rasterized grid with properly scaled SI units."""
        if field == "permittivity": grid = self.permittivity
        elif field == "permeability": grid = self.permeability
        elif field == "conductivity": grid = self.conductivity
        print(f"Min: {np.min(grid):.3f}, Max: {np.max(grid):.3f}")
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