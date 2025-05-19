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
        x_centers = np.linspace(0.5 * cell_size, width - 0.5 * cell_size, grid_width)
        y_centers = np.linspace(0.5 * cell_size, height - 0.5 * cell_size, grid_height)
        
        # Precompute offsets for all 9 sample points
        offsets = np.array([-0.25, 0, 0.25]) * cell_size
        dx, dy = np.meshgrid(offsets, offsets)
        dx = dx.flatten()
        dy = dy.flatten()
        num_samples = len(dx)
        
        # Estimate dt for PML calculations
        c = 3e8  # Speed of light
        dt_estimate = 0.5 * self.resolution / (c * np.sqrt(2))
        
        # Initialize material grids with vacuum (air) properties
        permittivity = np.ones((grid_height, grid_width))
        permeability = np.ones((grid_height, grid_width))
        conductivity = np.zeros((grid_height, grid_width))
        
        # Start with the background (first structure)
        if len(self.design.structures) > 0:
            background = self.design.structures[0]
            if hasattr(background, 'material') and background.material is not None:
                # Fast fill for background
                permittivity.fill(background.material.permittivity)
                permeability.fill(background.material.permeability)
                conductivity.fill(background.material.conductivity)
        
        # Process remaining structures in reverse order (foreground objects last)
        # Note: we process in ORIGINAL order, not reversed, because we want background first
        with create_rich_progress() as progress:
            task = progress.add_task("Rasterizing structures...", total=len(self.design.structures))
            progress.update(task, advance=1)  # Skip the background we already processed
            # Process each structure (except background)
            for idx in range(1, len(self.design.structures)):
                structure = self.design.structures[idx]
                # Skip PML visualization structures
                if hasattr(structure, 'is_pml') and structure.is_pml:
                    progress.update(task, advance=1)
                    continue
                # Skip structures without material
                if not hasattr(structure, 'material') or structure.material is None:
                    progress.update(task, advance=1)
                    continue
                # Cache material properties for performance
                mat_perm = structure.material.permittivity
                mat_permb = structure.material.permeability
                mat_cond = structure.material.conductivity
                try:
                    # Get bounding box of the structure
                    bbox = structure.get_bounding_box()
                    if bbox is None: raise AttributeError("Bounding box is None")
                    min_x, min_y, max_x, max_y = bbox
                    # Convert to grid indices
                    min_i = max(0, int(min_y / cell_size) - 1)
                    min_j = max(0, int(min_x / cell_size) - 1)
                    max_i = min(grid_height, int(np.ceil(max_y / cell_size)) + 1)
                    max_j = min(grid_width, int(np.ceil(max_x / cell_size)) + 1)
                    # Skip if bounding box is outside grid
                    if min_i >= grid_height or min_j >= grid_width or max_i <= 0 or max_j <= 0:
                        progress.update(task, advance=1)
                        continue
                    # Fast paths for different structure types
                    if isinstance(structure, Rectangle) and all(v == 0 for v in [
                            structure.vertices[0][0] - structure.position[0], 
                            structure.vertices[0][1] - structure.position[1]]):
                        # FAST PATH: Axis-aligned rectangle
                        # Define rectangle bounds for grid indices
                        rect_min_j = max(0, int(structure.position[0] / cell_size))
                        rect_min_i = max(0, int(structure.position[1] / cell_size))
                        rect_max_j = min(grid_width, int(np.ceil((structure.position[0] + structure.width) / cell_size)))
                        rect_max_i = min(grid_height, int(np.ceil((structure.position[1] + structure.height) / cell_size)))
                        # Identify interior and boundary cells
                        inner_min_j = max(0, int((structure.position[0] + 0.25 * cell_size) / cell_size))
                        inner_min_i = max(0, int((structure.position[1] + 0.25 * cell_size) / cell_size))
                        inner_max_j = min(grid_width, int(np.floor((structure.position[0] + structure.width - 0.25 * cell_size) / cell_size)))
                        inner_max_i = min(grid_height, int(np.floor((structure.position[1] + structure.height - 0.25 * cell_size) / cell_size)))
                        # Fast fill interior cells (fully covered, no need for sampling)
                        if inner_max_i > inner_min_i and inner_max_j > inner_min_j:
                            permittivity[inner_min_i:inner_max_i, inner_min_j:inner_max_j] = mat_perm
                            permeability[inner_min_i:inner_max_i, inner_min_j:inner_max_j] = mat_permb
                            conductivity[inner_min_i:inner_max_i, inner_min_j:inner_max_j] = mat_cond
                        # Calculate boundary region cells (those that need super-sampling)
                        # This is more efficient than checking each cell individually
                        boundary_mask = np.zeros((rect_max_i - rect_min_i, rect_max_j - rect_min_j), dtype=bool)
                        # Top and bottom boundaries
                        if rect_min_i < inner_min_i: boundary_mask[:inner_min_i-rect_min_i, :] = True 
                        if inner_max_i < rect_max_i: boundary_mask[inner_max_i-rect_min_i:, :] = True
                        # Left and right boundaries
                        if rect_min_j < inner_min_j: boundary_mask[:, :inner_min_j-rect_min_j] = True 
                        if inner_max_j < rect_max_j: boundary_mask[:, inner_max_j-rect_min_j:] = True
                        # Process boundary cells with super-sampling
                        boundary_indices = np.where(boundary_mask)
                        for idx in range(len(boundary_indices[0])):
                            i_rel, j_rel = boundary_indices[0][idx], boundary_indices[1][idx]
                            i, j = i_rel + rect_min_i, j_rel + rect_min_j
                            # Cell center
                            center_x = x_centers[j]
                            center_y = y_centers[i]
                            # Count samples inside rectangle
                            samples_inside = 0
                            for k in range(num_samples):
                                x_sample = center_x + dx[k]
                                y_sample = center_y + dy[k]
                                if (structure.position[0] <= x_sample < structure.position[0] + structure.width and
                                    structure.position[1] <= y_sample < structure.position[1] + structure.height):
                                    samples_inside += 1
                            if samples_inside > 0:
                                # Calculate blend factor
                                blend_factor = samples_inside / num_samples
                                # Update material properties
                                permittivity[i, j] = permittivity[i, j] * (1 - blend_factor) + mat_perm * blend_factor
                                permeability[i, j] = permeability[i, j] * (1 - blend_factor) + mat_permb * blend_factor
                                conductivity[i, j] = conductivity[i, j] * (1 - blend_factor) + mat_cond * blend_factor
                    
                    elif hasattr(structure, 'radius'):  # Circle
                        # FAST PATH: Circle
                        # Get circle parameters
                        center_x, center_y = structure.position
                        radius = structure.radius
                        # Create local coordinate arrays for the bounding box region
                        j_indices = np.arange(min_j, max_j)
                        i_indices = np.arange(min_i, max_i)
                        local_x = x_centers[j_indices]
                        local_y = y_centers[i_indices]
                        # Create a grid of coordinates
                        X, Y = np.meshgrid(local_x, local_y)
                        # Calculate distances from center
                        distances = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
                        # Find cells fully inside circle (all sample points inside)
                        fully_inside = distances + 0.3536 * cell_size <= radius  # sqrt(2)/4 ≈ 0.3536 for diagonal
                        # Find cells potentially on the boundary (need super-sampling)
                        boundary = (distances - 0.3536 * cell_size <= radius) & ~fully_inside
                        # Fast update for fully inside cells
                        local_i, local_j = np.where(fully_inside)
                        global_i, global_j = local_i + min_i, local_j + min_j
                        if len(global_i) > 0:
                            permittivity[global_i, global_j] = mat_perm
                            permeability[global_i, global_j] = mat_permb
                            conductivity[global_i, global_j] = mat_cond
                        # Super-sample for boundary cells
                        boundary_i, boundary_j = np.where(boundary)
                        for idx in range(len(boundary_i)):
                            i, j = boundary_i[idx] + min_i, boundary_j[idx] + min_j
                            # Cell center
                            center_x_cell = x_centers[j]
                            center_y_cell = y_centers[i]
                            # Count samples inside circle
                            samples_inside = 0
                            for k in range(num_samples):
                                x_sample = center_x_cell + dx[k]
                                y_sample = center_y_cell + dy[k]
                                if np.hypot(x_sample - center_x, y_sample - center_y) <= radius:
                                    samples_inside += 1
                            if samples_inside > 0:
                                # Calculate blend factor
                                blend_factor = samples_inside / num_samples
                                # Update material properties
                                permittivity[i, j] = permittivity[i, j] * (1 - blend_factor) + mat_perm * blend_factor
                                permeability[i, j] = permeability[i, j] * (1 - blend_factor) + mat_permb * blend_factor
                                conductivity[i, j] = conductivity[i, j] * (1 - blend_factor) + mat_cond * blend_factor
                    
                    elif hasattr(structure, 'inner_radius') and hasattr(structure, 'outer_radius'):  # Ring
                        # FAST PATH: Ring
                        # Get ring parameters
                        center_x, center_y = structure.position
                        inner_radius = structure.inner_radius
                        outer_radius = structure.outer_radius
                        # Create local coordinate arrays for the bounding box region
                        j_indices = np.arange(min_j, max_j)
                        i_indices = np.arange(min_i, max_i)
                        local_x = x_centers[j_indices]
                        local_y = y_centers[i_indices]
                        # Create a grid of coordinates
                        X, Y = np.meshgrid(local_x, local_y)
                        # Calculate distances from center
                        distances = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
                        # Find cells fully inside ring (all sample points inside)
                        fully_inside = (distances - 0.3536 * cell_size >= inner_radius) & (distances + 0.3536 * cell_size <= outer_radius)
                        # Find cells potentially on the boundary (need super-sampling)
                        inner_boundary = (distances - 0.3536 * cell_size <= inner_radius) & (distances + 0.3536 * cell_size >= inner_radius)
                        outer_boundary = (distances - 0.3536 * cell_size <= outer_radius) & (distances + 0.3536 * cell_size >= outer_radius)
                        boundary = inner_boundary | outer_boundary
                        # Fast update for fully inside cells
                        local_i, local_j = np.where(fully_inside)
                        global_i, global_j = local_i + min_i, local_j + min_j
                        if len(global_i) > 0:
                            permittivity[global_i, global_j] = mat_perm
                            permeability[global_i, global_j] = mat_permb
                            conductivity[global_i, global_j] = mat_cond
                        # Super-sample for boundary cells
                        boundary_i, boundary_j = np.where(boundary)
                        for idx in range(len(boundary_i)):
                            i, j = boundary_i[idx] + min_i, boundary_j[idx] + min_j
                            # Cell center
                            center_x_cell = x_centers[j]
                            center_y_cell = y_centers[i]
                            # Count samples inside ring
                            samples_inside = 0
                            for k in range(num_samples):
                                x_sample = center_x_cell + dx[k]
                                y_sample = center_y_cell + dy[k]
                                distance = np.hypot(x_sample - center_x, y_sample - center_y)
                                if inner_radius <= distance <= outer_radius:
                                    samples_inside += 1
                            if samples_inside > 0:
                                # Calculate blend factor
                                blend_factor = samples_inside / num_samples
                                # Update material properties
                                permittivity[i, j] = permittivity[i, j] * (1 - blend_factor) + mat_perm * blend_factor
                                permeability[i, j] = permeability[i, j] * (1 - blend_factor) + mat_permb * blend_factor
                                conductivity[i, j] = conductivity[i, j] * (1 - blend_factor) + mat_cond * blend_factor
                    else:
                        # GENERAL PATH: For polygons and complex shapes
                        # Select the appropriate containment function
                        if hasattr(structure, 'point_in_polygon'):
                            contains_func = lambda x, y: structure.point_in_polygon(x, y)
                        else:
                            # Fallback method using material values
                            contains_func = lambda x, y: any(val != def_val for val, def_val in zip(
                                self.design.get_material_value(x, y), [1.0, 1.0, 0.0]))
                        # First, try to identify fully inside cells if possible to minimize super-sampling
                        if hasattr(structure, 'vertices') and len(getattr(structure, 'vertices', [])) > 0:
                            # Sample a center grid of points in each cell to detect likely inside areas
                            # This is a heuristic to identify cells likely fully inside
                            inside_mask = np.zeros((max_i - min_i, max_j - min_j), dtype=bool)
                            boundary_mask = np.zeros((max_i - min_i, max_j - min_j), dtype=bool)
                            # Sample 5 points per cell (center and corners) to identify inside/boundary cells
                            sample_points = [(0, 0), (-0.4, -0.4), (-0.4, 0.4), (0.4, -0.4), (0.4, 0.4)]
                            for i_rel in range(max_i - min_i):
                                for j_rel in range(max_j - min_j):
                                    i, j = i_rel + min_i, j_rel + min_j
                                    # Get cell center
                                    center_x = x_centers[j]
                                    center_y = y_centers[i]
                                    # Track points inside/outside
                                    points_inside = 0
                                    center_inside = False
                                    # Check center point first
                                    if contains_func(center_x, center_y):
                                        center_inside = True
                                        points_inside += 1
                                    # Check corner points
                                    for dx_pt, dy_pt in sample_points[1:]:
                                        x_pt = center_x + dx_pt * cell_size
                                        y_pt = center_y + dy_pt * cell_size
                                        if contains_func(x_pt, y_pt):
                                            points_inside += 1
                                    # If center is inside and all sample points are inside
                                    if center_inside and points_inside == len(sample_points):
                                        inside_mask[i_rel, j_rel] = True
                                    # If some points are inside and some are outside
                                    elif points_inside > 0:
                                        boundary_mask[i_rel, j_rel] = True
                            
                            # Fast update for fully inside cells
                            inside_i, inside_j = np.where(inside_mask)
                            for idx in range(len(inside_i)):
                                i, j = inside_i[idx] + min_i, inside_j[idx] + min_j
                                permittivity[i, j] = mat_perm
                                permeability[i, j] = mat_permb
                                conductivity[i, j] = mat_cond
                            
                            # Super-sample for boundary cells
                            boundary_i, boundary_j = np.where(boundary_mask)
                            for idx in range(len(boundary_i)):
                                i, j = boundary_i[idx] + min_i, boundary_j[idx] + min_j
                                # Cell center
                                center_x = x_centers[j]
                                center_y = y_centers[i]
                                # Count samples inside shape
                                samples_inside = 0
                                for k in range(num_samples):
                                    x_sample = center_x + dx[k]
                                    y_sample = center_y + dy[k]
                                    if contains_func(x_sample, y_sample):
                                        samples_inside += 1
                                
                                if samples_inside > 0:
                                    # Calculate blend factor
                                    blend_factor = samples_inside / num_samples
                                    # Update material properties
                                    permittivity[i, j] = permittivity[i, j] * (1 - blend_factor) + mat_perm * blend_factor
                                    permeability[i, j] = permeability[i, j] * (1 - blend_factor) + mat_permb * blend_factor
                                    conductivity[i, j] = conductivity[i, j] * (1 - blend_factor) + mat_cond * blend_factor
                            
                            # Check remaining cells not marked as inside or boundary
                            remaining_i, remaining_j = np.where(~inside_mask & ~boundary_mask)
                            for idx in range(len(remaining_i)):
                                i, j = remaining_i[idx] + min_i, remaining_j[idx] + min_j
                                # Cell center
                                center_x = x_centers[j]
                                center_y = y_centers[i]
                                # Super-sample
                                samples_inside = 0
                                for k in range(num_samples):
                                    x_sample = center_x + dx[k]
                                    y_sample = center_y + dy[k]
                                    if contains_func(x_sample, y_sample):
                                        samples_inside += 1
                                
                                if samples_inside > 0:
                                    # Calculate blend factor
                                    blend_factor = samples_inside / num_samples
                                    # Update material properties
                                    permittivity[i, j] = permittivity[i, j] * (1 - blend_factor) + mat_perm * blend_factor
                                    permeability[i, j] = permeability[i, j] * (1 - blend_factor) + mat_permb * blend_factor
                                    conductivity[i, j] = conductivity[i, j] * (1 - blend_factor) + mat_cond * blend_factor
                        else:
                            # Direct super-sampling for all cells in bounding box
                            for i in range(min_i, max_i):
                                for j in range(min_j, max_j):
                                    # Cell center
                                    center_x = x_centers[j]
                                    center_y = y_centers[i]
                                    # Super-sample
                                    samples_inside = 0
                                    for k in range(num_samples):
                                        x_sample = center_x + dx[k]
                                        y_sample = center_y + dy[k]
                                        if contains_func(x_sample, y_sample):
                                            samples_inside += 1
                                    
                                    if samples_inside > 0:
                                        # Calculate blend factor
                                        blend_factor = samples_inside / num_samples
                                        # Update material properties
                                        permittivity[i, j] = permittivity[i, j] * (1 - blend_factor) + mat_perm * blend_factor
                                        permeability[i, j] = permeability[i, j] * (1 - blend_factor) + mat_permb * blend_factor
                                        conductivity[i, j] = conductivity[i, j] * (1 - blend_factor) + mat_cond * blend_factor
                    
                except (AttributeError, TypeError) as e:
                    print(f"Warning: Structure {type(structure)} doesn't have proper bounding box: {e}")
                
                progress.update(task, advance=1)
        
        # Process PML separately (only add conductivity to existing material values)
        with create_rich_progress() as progress:
            task = progress.add_task("Processing PML boundaries...", total=len(self.design.boundaries))
            for boundary in self.design.boundaries:
                # Get boundary region
                if hasattr(boundary, 'position'):
                    # Calculate PML region
                    if hasattr(boundary, 'width') and hasattr(boundary, 'height'):
                        # Rectangular PML
                        pos_x, pos_y = boundary.position
                        width, height = boundary.width, boundary.height
                        # Convert to grid indices
                        min_i = max(0, int(pos_y / cell_size))
                        min_j = max(0, int(pos_x / cell_size))
                        max_i = min(grid_height, int(np.ceil((pos_y + height) / cell_size)))
                        max_j = min(grid_width, int(np.ceil((pos_x + width) / cell_size)))
                    elif hasattr(boundary, 'radius'):
                        # Corner PML
                        center_x, center_y = boundary.position
                        radius = boundary.radius
                        # Calculate bounding box
                        min_i = max(0, int((center_y - radius) / cell_size))
                        min_j = max(0, int((center_x - radius) / cell_size))
                        max_i = min(grid_height, int(np.ceil((center_y + radius) / cell_size)))
                        max_j = min(grid_width, int(np.ceil((center_x + radius) / cell_size)))
                    else:
                        progress.update(task, advance=1)
                        continue
                else:
                    progress.update(task, advance=1)
                    continue
                
                # Fast calculation for entire PML region
                for i in range(min_i, max_i):
                    for j in range(min_j, max_j):
                        x = x_centers[j]
                        y = y_centers[i]
                        # Add PML conductivity (single sample at center is sufficient)
                        pml_conductivity = boundary.get_conductivity(
                            x, y, 
                            dx=self.resolution, 
                            dt=dt_estimate, 
                            eps_avg=permittivity[i, j]
                        )
                        if pml_conductivity > 0:
                            conductivity[i, j] += pml_conductivity
                
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
        if field == "permittivity": grid = self.permittivity
        elif field == "permeability": grid = self.permeability
        elif field == "conductivity": grid = self.conductivity
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