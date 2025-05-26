import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MatplotlibRectangle, PathPatch, Circle as MatplotlibCircle
from matplotlib.path import Path
import random
import numpy as np
from beamz.design.materials import Material
from beamz.const import µm, EPS_0, MU_0
from beamz.design.sources import ModeSource, GaussianSource
from beamz.design.monitors import Monitor
from beamz.helpers import display_header, display_status, tree_view, console, get_si_scale_and_label
import colorsys
import gdspy

class Design:
    def __init__(self, width=1, height=1, depth=None, material=None, color=None, border_color="black", auto_pml=True, pml_size=None):
        if material is None: material = Material(permittivity=1.0, permeability=1.0, conductivity=0.0)
        self.structures = [Rectangle(position=(0,0), width=width, height=height, material=material, color=color)]
        self.sources = []
        self.monitors = []
        self.boundaries = []
        self.width = width
        self.height = height
        self.depth = depth
        self.border_color = border_color
        self.time = 0
        self.is_3d = False if depth is None else True
        self.layers: dict[int, list[Polygon]] = {}
        if auto_pml: self.init_boundaries(pml_size)
        display_status(f"Created design with size: {self.width:.2e} x {self.height:.2e} m")

    def import_gds(gds_file: str):
        """Import a GDS file and return polygon and layer data."""
        gds_lib = gdspy.GdsLibrary(infile=gds_file)
        design = Design()  # Create GDSDesign instance
        cells = gds_lib.cells  # Get all cells from the library
        total_polygons_imported = 0
        for _cell_name, cell in cells.items():
            # Get polygons by spec, which returns a dict: {(layer, datatype): [poly1_points, poly2_points,...]}
            # polyN_points is a numpy array of vertices, e.g., [[x1, y1], [x2, y2], ...]
            gdspy_polygons_by_spec = cell.get_polygons(by_spec=True)
            for (layer_num, _datatype), list_of_polygon_points in gdspy_polygons_by_spec.items():
                if layer_num not in design.layers:
                    design.layers[layer_num] = []
                for polygon_points in list_of_polygon_points:
                    # Convert gdspy.Polygon points (numpy array) to list of tuples for beamz.Polygon
                    vertices = [(point[0], point[1]) for point in polygon_points]
                    beamz_polygon = Polygon(vertices=vertices)
                    design.layers[layer_num].append(beamz_polygon)
                    total_polygons_imported += 1
        print(f"Imported {total_polygons_imported} polygons from '{gds_file}' into GDSDesign object.")
        return design
    
    def export_gds(polygon, output_file):
        """Export a polygon to a GDS file."""
        lib = gdspy.GdsLibrary(unit=1, precision=1e-3)
        cell = lib.new_cell("main")
        cell.add(polygon)
        lib.write_gds(output_file)
        print(f"GDS file saved as '{output_file}'")
        
    def add(self, structure):
        """Core add function for adding structures on top of the design."""
        if isinstance(structure, ModeSource):
            self.sources.append(structure)
            self.structures.append(structure)
        elif isinstance(structure, GaussianSource):
            self.sources.append(structure)
            self.structures.append(structure)
        elif isinstance(structure, Monitor):
            self.monitors.append(structure)
            self.structures.append(structure)
        else: self.structures.append(structure)
        # Check for 3D structures
        if hasattr(structure, 'z') or hasattr(structure, 'depth'): self.is_3d = True

    def __iadd__(self, structure):
        """Implement += operator for adding structures."""
        self.add(structure)
        return self
    
    def unify_polygons(self):
        """If polygons are the same material and overlap spatially, unify them into a single, simplified polygon."""
        try:
            from shapely.geometry import Polygon as ShapelyPolygon
            from shapely.ops import unary_union
        except ImportError:
            display_status("Shapely library is required for polygon unification. Please install with: pip install shapely", "error")
            return False
            
        # Group structures by material properties
        material_groups = {}
        non_polygon_structures = []
        
        # Track which structures to remove later
        structures_to_remove = []
        
        # First pass: group polygons by material
        for structure in self.structures:
            # Skip PML visualizations, sources, monitors
            if hasattr(structure, 'is_pml') and structure.is_pml:
                non_polygon_structures.append(structure)
                continue
            if isinstance(structure, ModeSource) or isinstance(structure, GaussianSource) or isinstance(structure, Monitor):
                non_polygon_structures.append(structure)
                continue
                
            # Only process polygon-like structures with vertices
            if not hasattr(structure, 'vertices') or not hasattr(structure, 'material'):
                non_polygon_structures.append(structure)
                continue
                
            # Create a material key based on material properties
            material = structure.material
            if not material:
                non_polygon_structures.append(structure)
                continue
                
            material_key = (
                getattr(material, 'permittivity', None),
                getattr(material, 'permeability', None),
                getattr(material, 'conductivity', None)
            )
            
            # Add to the appropriate group
            if material_key not in material_groups:
                material_groups[material_key] = []
            
            # Convert to Shapely polygon
            try:
                # Handle polygons that might have interiors defined
                if hasattr(structure, 'interiors') and structure.interiors:
                    # Ensure vertices are not empty, and interiors are lists of coordinates
                    valid_interiors = [list(i_path) for i_path in structure.interiors if i_path]
                    if structure.vertices and valid_interiors:
                         shapely_polygon = ShapelyPolygon(shell=structure.vertices, holes=valid_interiors)
                    elif structure.vertices: # Only exterior is valid
                         shapely_polygon = ShapelyPolygon(shell=structure.vertices)
                    else: # No valid exterior
                        display_status(f"Skipping structure with no valid exterior vertices: {structure}", "warning")
                        non_polygon_structures.append(structure)
                        continue
                elif hasattr(structure, 'vertices') and structure.vertices: # Simple polygon with only exterior vertices
                    shapely_polygon = ShapelyPolygon(shell=structure.vertices)
                else: # Not a valid polygon structure
                    non_polygon_structures.append(structure)
                    continue # Skip if no vertices

                if shapely_polygon.is_valid:
                    material_groups[material_key].append((structure, shapely_polygon))
                    structures_to_remove.append(structure)
                else:
                    display_status(f"Skipping invalid polygon: {structure}", "warning")
                    non_polygon_structures.append(structure)
            except Exception as e:
                display_status(f"Error converting structure to Shapely polygon: {e}", "warning")
                non_polygon_structures.append(structure)
        
        # Second pass: Check for Ring objects that don't touch other objects
        rings_to_preserve = []
        
        for material_key, structure_group in material_groups.items():
            if len(structure_group) <= 1:
                continue  # Skip material groups with only one structure
                
            rings_in_group = [(idx, s) for idx, s in enumerate(structure_group) if isinstance(s[0], Ring)]
            if not rings_in_group:
                continue  # No rings in this group
                
            # For each Ring, check if it touches any other object
            for ring_idx, (ring, ring_shapely) in rings_in_group:
                touches_other_object = False
                
                # Check against other polygons in the same material group
                for other_idx, (other_struct, other_shapely) in enumerate(structure_group):
                    if ring_idx == other_idx:
                        continue  # Skip self-comparison
                        
                    if ring_shapely.intersects(other_shapely):
                        touches_other_object = True
                        break
                        
                # If the Ring doesn't touch any other object, preserve it
                if not touches_other_object:
                    rings_to_preserve.append(ring)
                    # Remove it from the material group to prevent it from being unified
                    material_groups[material_key].pop(ring_idx)
                    if ring in structures_to_remove:
                        structures_to_remove.remove(ring)
                        
        # Third pass: unify polygons within each material group
        new_structures = []
        for material_key, structure_group in material_groups.items():
            if len(structure_group) <= 1:
                # Only one structure with this material, no merging needed
                new_structures.extend([s[0] for s in structure_group])
                for s in structure_group:
                    if s[0] in structures_to_remove:
                        structures_to_remove.remove(s[0])
                continue
                
            # Extract shapely polygons for merging
            shapely_polygons = [p[1] for p in structure_group]
            
            # Get the material from the first structure in the group
            material = structure_group[0][0].material
            
            try:
                # Unify the polygons
                merged = unary_union(shapely_polygons)
                
                # The result could be a single polygon or a multipolygon
                if merged.geom_type == 'Polygon':
                    # Don't slice off the last vertex - our add_to_plot method needs complete vertices
                    exterior_coords = list(merged.exterior.coords[:-1])  # Keep [:-1] to remove duplicate closing vertex from Shapely
                    interior_coords_lists = [list(interior.coords[:-1]) for interior in merged.interiors]
                    
                    if exterior_coords:
                        new_poly = Polygon(vertices=exterior_coords, interiors=interior_coords_lists, material=material)
                        new_structures.append(new_poly)
                        display_status(f"Unified {len(structure_group)} polygons with permittivity={material_key[0]}", "success")
                    else:
                        display_status(f"Failed to convert merged polygon for material {material_key[0]} (no exterior), keeping original {len(structure_group)} structures.", "warning")
                        new_structures.extend([s[0] for s in structure_group])
                        for s_tuple in structure_group: # Ensure these are not removed
                            if s_tuple[0] in structures_to_remove:
                                structures_to_remove.remove(s_tuple[0])

                elif merged.geom_type == 'MultiPolygon':
                    all_geoms_converted_successfully = True
                    temp_new_polys_for_multipolygon = []
                    for geom in merged.geoms:
                        # Keep [:-1] to remove duplicate closing vertex from Shapely (our add_to_plot will add it back)
                        exterior_coords = list(geom.exterior.coords[:-1])
                        interior_coords_lists = [list(interior.coords[:-1]) for interior in geom.interiors]

                        if exterior_coords:
                            new_poly = Polygon(vertices=exterior_coords, interiors=interior_coords_lists, material=material)
                            temp_new_polys_for_multipolygon.append(new_poly)
                        else: # A sub-geometry had no exterior
                            all_geoms_converted_successfully = False
                            display_status(f"Failed to convert a geometry (no exterior) within MultiPolygon for material {material_key[0]}.", "warning")
                            break 
                    
                    if all_geoms_converted_successfully:
                        new_structures.extend(temp_new_polys_for_multipolygon)
                        display_status(f"Unified into {len(merged.geoms)} separate polygons with permittivity={material_key[0]}", "success")
                    else:
                        display_status(f"Reverting unification for material {material_key[0]} due to conversion error in MultiPolygon, keeping original {len(structure_group)} structures.", "warning")
                        new_structures.extend([s[0] for s in structure_group])
                        for s_tuple in structure_group: # Ensure these are not removed
                            if s_tuple[0] in structures_to_remove:
                                structures_to_remove.remove(s_tuple[0])
                else:
                    # If the result is something unexpected, keep the original structures
                    display_status(f"Unexpected geometry type after union: {merged.geom_type} for material {material_key[0]}, keeping original structures", "warning")
                    new_structures.extend([s[0] for s in structure_group])
                    for s in structure_group:
                        if s[0] in structures_to_remove:
                            structures_to_remove.remove(s[0])
            except Exception as e:
                display_status(f"Error unifying polygons: {e}", "error")
                # Keep original structures if unification fails
                new_structures.extend([s[0] for s in structure_group])
                for s in structure_group:
                    if s[0] in structures_to_remove:
                        structures_to_remove.remove(s[0])
        
        # Remove the structures that were unified
        for structure in structures_to_remove:
            if structure in self.structures:
                self.structures.remove(structure)
        
        # Add the unified structures, non-polygon structures, and preserved rings back
        self.structures.extend(new_structures)
        self.structures.extend(rings_to_preserve)
        
        # Final report
        display_status(f"Polygon unification complete: {len(structures_to_remove)} structures merged into {len(new_structures)} unified shapes, {len(rings_to_preserve)} isolated rings preserved", "success")
        return True

    def scatter(self, structure, n=1000, xyrange=(-5*µm, 5*µm), scale_range=(0.05, 1)):
        """Randomly distribute a given object over the design domain."""
        display_status(f"Scattering {n} instances of {structure.__class__.__name__}", "info")
        for _ in range(n):
            new_structure = structure.copy()
            new_structure.shift(random.uniform(xyrange[0], xyrange[1]), random.uniform(xyrange[0], xyrange[1]))
            new_structure.rotate(random.uniform(0, 360))
            new_structure.scale(random.uniform(scale_range[0], scale_range[1]))
            self.add(new_structure)
        display_status(f"Completed scattering {n} structures", "success")

    def init_boundaries(self, pml_size=None):
        """Add boundary conditions to the design area (using PML)."""
        # Calculate PML size more intelligently if not specified
        if pml_size is None:
            # Find max permittivity in design for wavelength calculation
            max_permittivity = 1.0
            for structure in self.structures:
                if hasattr(structure, 'material') and hasattr(structure.material, 'permittivity'):
                    max_permittivity = max(max_permittivity, structure.material.permittivity)
            # Estimate minimum wavelength
            wavelength_estimate = 1.55e-6 / np.sqrt(max_permittivity)
            # Make PML thicker to allow for more gradual absorption
            min_size = 1.5 * wavelength_estimate  # Increased from 1.0
            max_size = min(self.width, self.height) * 0.3  # Increased thickness for gradual absorption
            pml_size = max(min_size, min(max_size, min(self.width, self.height) / 3))
            display_status(f"Auto-selected PML size: {pml_size:.2e} m (~{pml_size/wavelength_estimate:.1f} wavelengths)", "info")
        
        # Create transparent material for PML outlines (for visualization only)
        pml_material = Material(permittivity=1.0, permeability=1.0, conductivity=0.0)
        
        # Create unified PML regions with optimized parameters for gradual absorption
        # Rectangular edge PMLs
        self.boundaries.append(PML("rect", (0, 0), (pml_size, self.height), "left"))
        self.boundaries.append(PML("rect", (self.width - pml_size, 0), (pml_size, self.height), "right"))
        self.boundaries.append(PML("rect", (0, self.height - pml_size), (self.width, pml_size), "top"))
        self.boundaries.append(PML("rect", (0, 0), (self.width, pml_size), "bottom"))
        
        # Corner PMLs
        self.boundaries.append(PML("corner", (0, 0), pml_size, "bottom-left"))
        self.boundaries.append(PML("corner", (self.width, 0), pml_size, "bottom-right"))
        self.boundaries.append(PML("corner", (0, self.height), pml_size, "top-left"))
        self.boundaries.append(PML("corner", (self.width, self.height), pml_size, "top-right"))
        
        # Add visual representations of PML regions to the structures list (for display only)
        # These are just visualization helpers and don't affect the actual simulation
        left_pml = Rectangle(
            position=(0, 0),
            width=pml_size,
            height=self.height,
            material=pml_material,
            color='none',
            is_pml=True  # Flag to identify it as a visual PML marker
        )
        self.structures.append(left_pml)
        # Right PML region
        right_pml = Rectangle(
            position=(self.width - pml_size, 0),
            width=pml_size,
            height=self.height,
            material=pml_material,
            color='none',
            is_pml=True
        )
        self.structures.append(right_pml)
        # Bottom PML region
        bottom_pml = Rectangle(
            position=(0, 0),
            width=self.width,
            height=pml_size,
            material=pml_material,
            color='none',
            is_pml=True
        )
        self.structures.append(bottom_pml)
        # Top PML region
        top_pml = Rectangle(
            position=(0, self.height - pml_size),
            width=self.width,
            height=pml_size,
            material=pml_material,
            color='none',
            is_pml=True
        )
        self.structures.append(top_pml)

    def show(self, unify_structures=True):
        """Display the design visually."""
        # Determine appropriate SI unit and scale
        max_dim = max(self.width, self.height)
        scale, unit = get_si_scale_and_label(max_dim)

        # Calculate figure size based on domain dimensions
        aspect_ratio = self.width / self.height
        base_size = 5
        if aspect_ratio > 1: figsize = (base_size * aspect_ratio, base_size)
        else: figsize = (base_size, base_size / aspect_ratio)

        # Do we want to show the indiviudal structures or a unified shape?
        if unify_structures: self.unify_polygons()

        # Create a single figure for all structures
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')
        
        # Now plot each structure
        for structure in self.structures:
            # Use dashed lines for PML regions
            if hasattr(structure, 'is_pml') and structure.is_pml:
                structure.add_to_plot(ax, edgecolor=self.border_color, linestyle='--', facecolor='none', alpha=0.5)
            else:
                structure.add_to_plot(ax)
        
        # Plot PML boundaries explicitly with dashed lines
        for boundary in self.boundaries:
            if hasattr(boundary, 'add_to_plot'):
                boundary.add_to_plot(ax, edgecolor=self.border_color, linestyle='--', facecolor='none', alpha=0.5)
        
        # Set proper limits, title and label, and ensure the full design is visible
        ax.set_title('Design Layout')
        ax.set_xlabel(f'X ({unit})')
        ax.set_ylabel(f'Y ({unit})')
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        
        # Update tick labels with scaled values
        ax.xaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
        ax.yaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
        
        # Adjust layout for clean appearance
        plt.tight_layout()
        plt.show()
        
    def __str__(self):
        return f"Design with {len(self.structures)} structures ({'3D' if self.is_3d else '2D'})"

    def get_material_value(self, x, y, dx=None, dt=None):
        """Return the material value at a given (x, y) coordinate, prioritizing the topmost structure."""
        # First get material values from underlying structures
        # Start with default background material 
        epsilon = 1.0
        mu = 1.0
        sigma_base = 0.0
        
        # Find the material values from the structures (outside PML calculation)
        for structure in reversed(self.structures):
            if isinstance(structure, Rectangle):
                if structure.is_pml:
                    # Skip visual PML structures - they're just for display
                    continue
                if self._point_in_polygon(x, y, structure.vertices):
                    epsilon = structure.material.permittivity
                    mu = structure.material.permeability
                    sigma_base = structure.material.conductivity
                    break
            elif isinstance(structure, Circle):
                if np.hypot(x - structure.position[0], y - structure.position[1]) <= structure.radius:
                    epsilon = structure.material.permittivity
                    mu = structure.material.permeability
                    sigma_base = structure.material.conductivity
                    break
            elif isinstance(structure, Ring):
                distance = np.hypot(x - structure.position[0], y - structure.position[1])
                if structure.inner_radius <= distance <= structure.outer_radius:
                    epsilon = structure.material.permittivity
                    mu = structure.material.permeability
                    sigma_base = structure.material.conductivity
                    break
            elif isinstance(structure, CircularBend):
                distance = np.hypot(x - structure.position[0], y - structure.position[1])
                if structure.inner_radius <= distance <= structure.outer_radius:
                    epsilon = structure.material.permittivity
                    mu = structure.material.permeability
                    sigma_base = structure.material.conductivity
                    break
            elif isinstance(structure, Polygon):
                if self._point_in_polygon(x, y, structure.vertices):
                    epsilon = structure.material.permittivity
                    mu = structure.material.permeability
                    sigma_base = structure.material.conductivity
                    break
        
        # Calculate PML conductivity based on the UNDERLYING material
        # This is crucial for proper absorption without reflection
        pml_conductivity = 0.0
        if dx is not None:
            eps_avg = epsilon  # Use the actual permittivity at this point
            # Apply all PML boundaries
            for boundary in self.boundaries:
                pml_conductivity += boundary.get_conductivity(x, y, dx=dx, dt=dt, eps_avg=eps_avg)
        
        # Return with the permittivity of the underlying structure plus PML conductivity
        return [epsilon, mu, sigma_base + pml_conductivity]

    def _point_in_polygon(self, x, y, vertices):
        """Check if a point is inside a polygon using the ray-casting algorithm."""
        n = len(vertices)
        inside = False
        p1x, p1y = vertices[0]
        for i in range(n + 1):
            p2x, p2y = vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def get_tree_view(self):
        """Return a structured view of the design as a tree"""
        design_data = {
            "Properties": {
                "Width": self.width,
                "Height": self.height,
                "Depth": self.depth,
                "Dimension": "3D" if self.is_3d else "2D"
            },
            "Structures": {},
            "Sources": {},
            "Monitors": {}
        }
        
        # Add structure data
        for idx, structure in enumerate(self.structures):
            if isinstance(structure, ModeSource) or isinstance(structure, GaussianSource) or isinstance(structure, Monitor):
                continue
                
            struct_type = structure.__class__.__name__
            if struct_type not in design_data["Structures"]:
                design_data["Structures"][struct_type] = []
                
            struct_info = {"position": getattr(structure, "position", None)}
            if hasattr(structure, "material"):
                mat = structure.material
                struct_info["material"] = {
                    "permittivity": getattr(mat, "permittivity", None),
                    "permeability": getattr(mat, "permeability", None),
                    "conductivity": getattr(mat, "conductivity", None)
                }
            design_data["Structures"][struct_type].append(struct_info)
        
        # Add source data
        for idx, source in enumerate(self.sources):
            source_type = source.__class__.__name__
            if source_type not in design_data["Sources"]:
                design_data["Sources"][source_type] = []
            
            source_info = {
                "position": source.position,
                "wavelength": getattr(source, "wavelength", None)
            }
            design_data["Sources"][source_type].append(source_info)
        
        # Add monitor data
        for idx, monitor in enumerate(self.monitors):
            monitor_type = monitor.__class__.__name__
            if monitor_type not in design_data["Monitors"]:
                design_data["Monitors"][monitor_type] = []
            
            monitor_info = {
                "position": monitor.position,
                "size": getattr(monitor, "size", None)
            }
            design_data["Monitors"][monitor_type].append(monitor_info)
            
        return design_data
        
    def display_tree(self):
        """Display the design as a hierarchical tree"""
        design_data = self.get_tree_view()
        tree_view(design_data, "Design Structure")

class Polygon:
    def __init__(self, vertices=None, material=None, color=None, optimize=False, interiors=None):
        self.vertices = vertices if vertices is not None else [] # Exterior path
        self.interiors = interiors if interiors is not None else [] # List of interior paths
        self.material = material
        self.optimize = optimize
        self.color = color if color is not None else self.get_random_color_consistent()
    
    def get_random_color_consistent(self, saturation=0.6, value=0.7):
        """Generate a random color with consistent perceived brightness and saturation."""
        hue = random.random() # Generate random hue (0-1)
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
    
    def shift(self, x, y):
        """Shift the polygon by (x,y) and return self for method chaining."""
        if self.vertices: self.vertices = [(v[0] + x, v[1] + y) for v in self.vertices]
        new_interiors_paths = []
        for interior_path in self.interiors:
            if interior_path: # Ensure path is not empty
                new_interiors_paths.append([(v[0] + x, v[1] + y) for v in interior_path])
        self.interiors = new_interiors_paths
        return self
    
    def scale(self, s):
        """Scale the polygon around its center of mass and return self for method chaining."""
        if self.vertices:
            # Calculate center of mass of the exterior
            x_center = sum(v[0] for v in self.vertices) / len(self.vertices)
            y_center = sum(v[1] for v in self.vertices) / len(self.vertices)
            
            # Scale exterior
            self.vertices = [(x_center + (v[0] - x_center) * s,
                              y_center + (v[1] - y_center) * s)
                              for v in self.vertices]
            # Scale interiors
            new_interiors_paths = []
            for interior_path in self.interiors:
                if interior_path:
                    new_interiors_paths.append([(x_center + (v[0] - x_center) * s,
                                                 y_center + (v[1] - y_center) * s)
                                                 for v in interior_path])
            self.interiors = new_interiors_paths
        return self
    
    def rotate(self, angle, point=None):
        """Rotate the polygon around its center of mass or specified point.
        angle: Rotation angle in degrees
        point: Optional (x,y) point to rotate around. If None, rotates around center of exterior.
        """
        if self.vertices:
            angle_rad = np.radians(angle)
            if point is None:
                # Calculate center of mass of the exterior
                x_center = sum(v[0] for v in self.vertices) / len(self.vertices)
                y_center = sum(v[1] for v in self.vertices) / len(self.vertices)
            else:
                x_center, y_center = point

            # Rotate exterior
            self.vertices = [
                (x_center + (v[0] - x_center) * np.cos(angle_rad) - (v[1] - y_center) * np.sin(angle_rad),
                 y_center + (v[0] - x_center) * np.sin(angle_rad) + (v[1] - y_center) * np.cos(angle_rad))
                for v in self.vertices
            ]
            # Rotate interiors
            new_interiors_paths = []
            for interior_path in self.interiors:
                if interior_path:
                    new_interiors_paths.append([
                        (x_center + (v[0] - x_center) * np.cos(angle_rad) - (v[1] - y_center) * np.sin(angle_rad),
                         y_center + (v[0] - x_center) * np.sin(angle_rad) + (v[1] - y_center) * np.cos(angle_rad))
                        for v in interior_path
                    ])
            self.interiors = new_interiors_paths
        return self

    def add_to_plot(self, ax, facecolor=None, edgecolor="black", alpha=None, linestyle=None):
        """Add the polygon as a patch to the axis, handling holes correctly."""
        if facecolor is None: facecolor = self.color
        if alpha is None: alpha = 1.0 # Default alpha to 1.0 for visibility
        if linestyle is None: linestyle = '-'
        
        if not self.vertices: # No exterior to draw
            return

        # Path components: first is exterior, subsequent are interiors
        all_path_coords = []
        all_path_codes = []

        # Exterior path (assume CCW from shapely, Path will handle fill direction)
        # A Path needs N vertices and N codes. For a polygon segment: MOVETO, LINETO,...,LINETO, CLOSEPOLY
        if len(self.vertices) > 0:
            # Add all vertices
            all_path_coords.extend(self.vertices)
            # Add the first vertex again to close the path visually
            all_path_coords.append(self.vertices[0])
            # Set codes: MOVETO for first vertex, LINETO for middle vertices, CLOSEPOLY for last
            all_path_codes.append(Path.MOVETO)
            if len(self.vertices) > 1:
                all_path_codes.extend([Path.LINETO] * (len(self.vertices) - 1))
            all_path_codes.append(Path.CLOSEPOLY)

        # Interior paths (assume CW from shapely for holes)
        for interior_v_list in self.interiors:
            if interior_v_list and len(interior_v_list) > 0:
                # Add all interior vertices
                all_path_coords.extend(interior_v_list)
                # Add the first vertex again to close the interior path
                all_path_coords.append(interior_v_list[0])
                # Set codes: MOVETO for first vertex, LINETO for middle vertices, CLOSEPOLY for last
                all_path_codes.append(Path.MOVETO)
                if len(interior_v_list) > 1:
                    all_path_codes.extend([Path.LINETO] * (len(interior_v_list) - 1))
                all_path_codes.append(Path.CLOSEPOLY)
        
        if not all_path_coords or not all_path_codes: # Nothing to draw
            return
            
        path = Path(np.array(all_path_coords), np.array(all_path_codes))
        patch = PathPatch(path, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor, linestyle=linestyle)
        ax.add_patch(patch)

    def copy(self):
        # Ensure interiors are copied as lists of tuples/lists, not as references
        copied_interiors = [list(path) for path in self.interiors if path] if self.interiors else []
        return Polygon(vertices=list(self.vertices) if self.vertices else [], 
                       interiors=copied_interiors, 
                       material=self.material, # Material can be shared
                       color=self.color, 
                       optimize=self.optimize)
        
    def get_bounding_box(self):
        """Get the bounding box of the polygon as (min_x, min_y, max_x, max_y)"""
        if not self.vertices or len(self.vertices) == 0:
            return (0, 0, 0, 0)
        
        # Extract x and y coordinates
        x_coords = [v[0] for v in self.vertices]
        y_coords = [v[1] for v in self.vertices]
        
        # Calculate min and max
        min_x = min(x_coords)
        min_y = min(y_coords)
        max_x = max(x_coords)
        max_y = max(y_coords)
        
        return (min_x, min_y, max_x, max_y)
        
    def _point_in_polygon_single_path(self, x, y, path_vertices):
        """Check if a point is inside a single, simple polygon path using ray-casting."""
        if not path_vertices: return False
        n = len(path_vertices)
        inside = False
        p1x, p1y = path_vertices[0]
        for i in range(n + 1):
            p2x, p2y = path_vertices[i % n] # Ensure closure for ray casting
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y: # Avoid division by zero for horizontal lines
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        else: # Edge is horizontal
                            xinters = p1x # Doesn't matter, will compare x <= max(p1x, p2x)
                        
                        # For horizontal edge, if point.y is on edge, only count if x is to the left
                        if p1y == p2y and y == p1y and x > min(p1x,p2x) and x <= max(p1x,p2x):
                             #This case is tricky, often handled by convention (e.g. points on edge)
                             #For now, if on horizontal line, consider it an intersection if x is to the left.
                             #This can be problematic, robust point-in-polygon is hard.
                             pass # Let standard check handle, or needs specific rule for horizontal edges.

                        if p1x == p2x or x <= xinters: # For vertical edge or point left of intersection
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def point_in_polygon(self, x, y):
        """Check if a point (x,y) is inside this polygon (which may have holes).
        
        'vertices' argument is ignored here, uses self.vertices and self.interiors.
        """
        # Use self.vertices for exterior and self.interiors for holes
        exterior_path = self.vertices
        interior_paths = self.interiors
        if not exterior_path: return False
        # Check if point is in the exterior boundary
        if not self._point_in_polygon_single_path(x, y, exterior_path):
            return False # Not in exterior, so definitely not in polygom
        # If in exterior, check if it's in any of the holes
        # Winding order of interiors (CW) should make _point_in_polygon_single_path return true if inside.
        for interior_path_pts in interior_paths:
            if interior_path_pts and self._point_in_polygon_single_path(x, y, interior_path_pts):
                return False # Point is in a hole, so not in polygon (as part of the material)
        return True # In exterior and not in any hole

class Rectangle(Polygon):
    def __init__(self, position=(0,0), width=1, height=1, material=None, color=None, is_pml=False, optimize=False):
        # Calculate vertices for the rectangle
        x, y = position
        vertices = [(x, y),  # Bottom left
                    (x + width, y),  # Bottom right
                    (x + width, y + height),  # Top right
                    (x, y + height)]
        super().__init__(vertices=vertices, material=material, color=color, optimize=optimize)
        self.position = position
        self.width = width
        self.height = height
        self.is_pml = is_pml
        
    def get_bounding_box(self):
        """Get the axis-aligned bounding box for this rectangle."""
        # For non-rotated rectangles, this is straightforward
        if not hasattr(self, 'vertices') or len(self.vertices) == 0:
            x, y = self.position
            return (x, y, x + self.width, y + self.height)
        # For potentially rotated rectangles, use the vertices
        return super().get_bounding_box()

    def shift(self, x, y):
        """Shift the rectangle by (x,y) and return self for method chaining."""
        self.position = (self.position[0] + x, self.position[1] + y)
        super().shift(x, y)
        return self

    def rotate(self, angle, point=None):
        """Rotate the rectangle around its center of mass or specified point."""        
        # Use parent class rotation method (which now handles degree to radian conversion)
        super().rotate(angle, point)
        # Calculate new bounding box after rotation
        min_x = min(v[0] for v in self.vertices)
        min_y = min(v[1] for v in self.vertices)
        max_x = max(v[0] for v in self.vertices)
        max_y = max(v[1] for v in self.vertices)
        # Update position to be the bottom-left corner
        self.position = (min_x, min_y)
        self.width = max_x - min_x
        self.height = max_y - min_y
        return self

    def scale(self, s):
        """Scale the rectangle around its center of mass and return self for method chaining."""
        super().scale(s)
        self.width *= s; self.height *= s
        return self
    
    def copy(self):
        """Create a copy of this rectangle with the same attributes and vertices."""
        new_rect = Rectangle(self.position, self.width, self.height, 
                            self.material, self.color, self.is_pml, self.optimize)
        # Ensure vertices are copied exactly as they are (important for rotated rectangles)
        new_rect.vertices = [(x, y) for x, y in self.vertices]
        return new_rect

class Circle(Polygon):
    def __init__(self, position=(0,0), radius=1, points=32, material=None, color=None, optimize=False):
        theta = np.linspace(0, 2*np.pi, points, endpoint=False)
        vertices = [(position[0] + radius * np.cos(t), position[1] + radius * np.sin(t)) for t in theta]
        super().__init__(vertices=vertices, material=material, color=color, optimize=optimize)
        self.position = position
        self.radius = radius
    
    def shift(self, x, y):
        """Shift the circle by (x,y) and return self for method chaining."""
        self.position = (self.position[0] + x, self.position[1] + y)
        super().shift(x, y)
        return self
    
    def scale(self, s):
        """Scale the circle radius by s and return self for method chaining."""
        self.radius *= s
        # Regenerate vertices with new radius
        N = len(self.vertices)
        theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        self.vertices = [(self.position[0] + self.radius * np.cos(t), 
                         self.position[1] + self.radius * np.sin(t)) for t in theta]
        return self
    
    def copy(self):
        return Circle(position=self.position, radius=self.radius, points=len(self.vertices), 
                     material=self.material, color=self.color, optimize=self.optimize)

class Ring(Polygon):
    def __init__(self, position=(0,0), inner_radius=1, outer_radius=2, material=None, color=None, optimize=False, points=256):
        theta = np.linspace(0, 2*np.pi, points, endpoint=False) # CCW, N points
        # Exterior path (CCW, N points)
        # These are unclosed paths, as expected by Polygon.add_to_plot's Path logic
        outer_ext_vertices = [(position[0] + outer_radius * np.cos(t), 
                               position[1] + outer_radius * np.sin(t)) for t in theta]
        # Interior path (should be CW for Matplotlib Path hole convention if exterior is CCW)
        # Generate points CW by reversing theta or using reversed(theta)
        inner_int_vertices_cw = [(position[0] + inner_radius * np.cos(t), 
                                  position[1] + inner_radius * np.sin(t)) for t in reversed(theta)]
        super().__init__(vertices=outer_ext_vertices, 
                         interiors=[inner_int_vertices_cw] if inner_int_vertices_cw else [], 
                         material=material, color=color, optimize=optimize)
        self.points = points # Store for potential regeneration or other logic if needed
        self.position = position
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
    
    def shift(self, x, y):
        """Shift the ring by (x,y) and return self for method chaining."""
        self.position = (self.position[0] + x, self.position[1] + y)
        super().shift(x, y)
        return self
    
    def scale(self, s):
        """Scale the ring radii by s and return self for method chaining."""
        self.inner_radius *= s; self.outer_radius *= s
        # Regenerate vertices with new radii
        N = len(self.vertices) // 2  # Half the vertices for each circle
        theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        # Outer circle points (clockwise) and inner circle points (counterclockwise)
        outer_vertices = [(self.position[0] + self.outer_radius * np.cos(t), 
                          self.position[1] + self.outer_radius * np.sin(t)) for t in theta]
        inner_vertices = [(self.position[0] + self.inner_radius * np.cos(t), 
                          self.position[1] + self.inner_radius * np.sin(t)) for t in reversed(theta)]
        self.vertices = outer_vertices + inner_vertices
        return self
    
    def add_to_plot(self, ax, facecolor=None, edgecolor="black", alpha=None, linestyle=None):
        if facecolor is None: facecolor = self.color
        if alpha is None: alpha = 1
        if linestyle is None: linestyle = '-'
        # Use the generic Polygon.add_to_plot which now handles holes via self.vertices and self.interiors
        # Ring.__init__ now populates self.vertices (exterior) and self.interiors correctly.
        super().add_to_plot(ax, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linestyle=linestyle)

    def copy(self):
        # Ring's __init__ now correctly sets up vertices and interiors for Polygon base class
        return Ring(position=self.position, 
                    inner_radius=self.inner_radius, 
                    outer_radius=self.outer_radius, 
                    material=self.material, # Material can be shared
                    color=self.color, 
                    optimize=self.optimize,
                    points=self.points)

class CircularBend(Polygon):
    def __init__(self, position=(0,0), inner_radius=1, outer_radius=2, angle=90, rotation=0, material=None, 
                 facecolor=None, optimize=False, points=64):
        self.points = points
        theta = np.linspace(0, np.radians(angle), points)
        rotation_rad = np.radians(rotation)
        outer_vertices = [(position[0] + outer_radius * np.cos(t + rotation_rad),
                          position[1] + outer_radius * np.sin(t + rotation_rad)) for t in theta]
        inner_vertices = [(position[0] + inner_radius * np.cos(t + rotation_rad),
                          position[1] + inner_radius * np.sin(t + rotation_rad)) for t in reversed(theta)]
        vertices = outer_vertices + inner_vertices
        super().__init__(vertices=vertices, material=material, color=facecolor, optimize=optimize)
        self.position = position
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.angle = angle
        self.rotation = rotation
    
    def shift(self, x, y):
        """Shift the bend by (x,y) and return self for method chaining."""
        self.position = (self.position[0] + x, self.position[1] + y)
        super().shift(x, y)
        return self
    
    def rotate(self, angle, point=None):
        """Rotate the bend around its center or specified point."""
        self.rotation = (self.rotation + angle) % 360
        super().rotate(angle, point or self.position)
        return self
    
    def scale(self, s):
        """Scale the bend radii by s and return self for method chaining."""
        self.inner_radius *= s; self.outer_radius *= s
        N = len(self.vertices) // 2  # Half the vertices for each arc
        theta = np.linspace(0, np.radians(self.angle), N)
        rotation_rad = np.radians(self.rotation)
        outer_vertices = [(self.position[0] + self.outer_radius * np.cos(t + rotation_rad),
                          self.position[1] + self.outer_radius * np.sin(t + rotation_rad)) for t in theta]
        inner_vertices = [(self.position[0] + self.inner_radius * np.cos(t + rotation_rad),
                          self.position[1] + self.inner_radius * np.sin(t + rotation_rad)) for t in reversed(theta)]
        self.vertices = outer_vertices + inner_vertices
        return self
    
    def add_to_plot(self, ax, facecolor=None, edgecolor="black", alpha=None, linestyle=None):
        if facecolor is None: facecolor = self.color
        if alpha is None: alpha = 1
        if linestyle is None: linestyle = '-'
        # Convert angles to radians
        angle_rad = np.radians(self.angle)
        rotation_rad = np.radians(self.rotation)
        theta = np.linspace(rotation_rad, rotation_rad + angle_rad, self.points, endpoint=True)
        # Outer and inner arc points
        x_outer = self.position[0] + self.outer_radius * np.cos(theta)
        y_outer = self.position[1] + self.outer_radius * np.sin(theta)
        x_inner = self.position[0] + self.inner_radius * np.cos(theta)
        y_inner = self.position[1] + self.inner_radius * np.sin(theta)
        # Create a closed path by combining points and adding connecting lines
        vertices = np.vstack([
            [x_outer[0], y_outer[0]],
            *np.column_stack([x_outer[1:], y_outer[1:]]),
            [x_inner[-1], y_inner[-1]],
            *np.column_stack([x_inner[-2::-1], y_inner[-2::-1]]),
            [x_outer[0], y_outer[0]]
        ])
        # Define path codes for a single continuous path
        codes = [Path.MOVETO] + \
                [Path.LINETO] * (len(vertices) - 2) + \
                [Path.CLOSEPOLY]
        # Create the path and patch
        path = Path(vertices, codes)
        bend_patch = PathPatch(path, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor, linestyle=linestyle)
        ax.add_patch(bend_patch)
        
    def copy(self):
        return CircularBend(self.position, self.inner_radius, self.outer_radius, 
                            self.angle, self.rotation, self.material, self.color, self.optimize)

class Taper(Polygon):
    """Taper is a structure that tapers from a width to a height."""
    def __init__(self, position=(0,0), input_width=1, output_width=0.5, length=1, material=None, color=None, optimize=False):
        # Calculate vertices for the trapezoid shape
        x, y = position
        vertices = [(x, y - input_width/2),  # Bottom left
                    (x + length, y - output_width/2),  # Bottom right
                    (x + length, y + output_width/2),  # Top right
                    (x, y + input_width/2)] # Top left
        super().__init__(vertices=vertices, material=material, color=color)
        self.position = position
        self.input_width = input_width
        self.output_width = output_width
        self.length = length
        self.optimize = optimize

    def rotate(self, angle, point=None):
        """Rotate the taper around its center of mass or specified point."""
        # Use parent class rotation method
        super().rotate(angle, point)
        # Calculate new bounding box after rotation
        min_x = min(v[0] for v in self.vertices)
        min_y = min(v[1] for v in self.vertices)
        max_x = max(v[0] for v in self.vertices)
        max_y = max(v[1] for v in self.vertices)
        # Update position to left bottom corner and update length
        self.position = (min_x, min_y)
        self.length = max_x - min_x
        return self

    def copy(self):
        """Create a copy of this taper with the same attributes and vertices."""
        new_taper = Taper(self.position, self.input_width, self.output_width, 
                          self.length, self.material, self.color, self.optimize)
        # Ensure vertices are copied exactly as they are (important for rotated tapers)
        new_taper.vertices = [(x, y) for x, y in self.vertices]
        return new_taper

class PML:
    """Unified PML (Perfectly Matched Layer) class for absorbing boundary conditions."""
    def __init__(self, region_type, position, size, orientation, polynomial_order=2.0, sigma_factor=1.0, alpha_max=0.1):
        self.region_type = region_type  # "rect" or "corner"
        self.position = position
        self.orientation = orientation
        self.polynomial_order = polynomial_order  # Reduced to allow smoother transition
        self.sigma_factor = sigma_factor  # Reduced to allow waves to enter
        self.alpha_max = alpha_max  # Reduced frequency-shifting for smoother transition
        if region_type == "rect": self.width, self.height = size
        else: self.radius = size

    def add_to_plot(self, ax, facecolor='none', edgecolor="black", alpha=0.5, linestyle='--'):
        """Add the PML boundary to a plot with dashed lines."""
        if self.region_type == "rect":
            # Create a rectangle patch for rectangular PML regions
            rect_patch = MatplotlibRectangle(
                (self.position[0], self.position[1]),
                self.width, self.height,
                fill=False, 
                edgecolor=edgecolor,
                linestyle=linestyle,
                alpha=alpha
            )
            ax.add_patch(rect_patch)
        elif self.region_type == "corner":
            # Use a rectangle for corner PML regions as well
            # Position and size depend on orientation
            if self.orientation == "bottom-left":
                rect_patch = MatplotlibRectangle(
                    (self.position[0] - self.radius, self.position[1] - self.radius),
                    self.radius, self.radius,
                    fill=False,
                    edgecolor=edgecolor,
                    linestyle=linestyle,
                    alpha=alpha
                )
            elif self.orientation == "bottom-right":
                rect_patch = MatplotlibRectangle(
                    (self.position[0], self.position[1] - self.radius),
                    self.radius, self.radius,
                    fill=False,
                    edgecolor=edgecolor,
                    linestyle=linestyle,
                    alpha=alpha
                )
            elif self.orientation == "top-right":
                rect_patch = MatplotlibRectangle(
                    (self.position[0], self.position[1]),
                    self.radius, self.radius,
                    fill=False,
                    edgecolor=edgecolor,
                    linestyle=linestyle,
                    alpha=alpha
                )
            elif self.orientation == "top-left":
                rect_patch = MatplotlibRectangle(
                    (self.position[0] - self.radius, self.position[1]),
                    self.radius, self.radius,
                    fill=False,
                    edgecolor=edgecolor,
                    linestyle=linestyle,
                    alpha=alpha
                )
            ax.add_patch(rect_patch)

    def get_profile(self, normalized_distance):
        """Calculate PML absorption profile using gradual grading."""
        # Ensure distance is within [0,1]
        d = min(max(normalized_distance, 0.0), 1.0)
        # Create a smooth transition from 0 at the interface
        # Start with nearly zero conductivity at the interface and gradually increase
        if d < 0.05: sigma = 0.01 * (d/0.05)**2
        else: sigma = ((d - 0.05) / 0.95)**self.polynomial_order
        # Smooth frequency-shifting profile
        alpha = self.alpha_max * d**2  # Quadratic profile for smooth transition
        return sigma, alpha
    
    def get_conductivity(self, x, y, dx=None, dt=None, eps_avg=None):
        """Calculate PML conductivity at a point using smooth-transition PML."""
        # Calculate theoretical optimal conductivity based on impedance matching
        if dx is not None and eps_avg is not None:
            # Calculate impedance 
            eta = np.sqrt(MU_0 / (EPS_0 * eps_avg))
            # Optimal conductivity for minimal reflection at interface
            # Reduced from 1.2 to 0.8 for smoother transition
            sigma_max = 1.2 / (eta * dx)
            sigma_max *= self.sigma_factor  # Apply gentler factor
        else: sigma_max = 1.0  # Lower default conductivity
        
        # Get normalized distance based on region type and orientation
        if self.region_type == "rect":
            # Check if point is within rectangular PML region
            if not (self.position[0] <= x <= self.position[0] + self.width and
                    self.position[1] <= y <= self.position[1] + self.height):
                return 0.0
            # Calculate normalized distance from boundary based on orientation
            # Distance should be 0 at inner boundary and 1 at outer boundary
            if self.orientation == "left": distance = 1.0 - (x - self.position[0]) / self.width
            elif self.orientation == "right": distance = (x - self.position[0]) / self.width
            elif self.orientation == "top": distance = (y - self.position[1]) / self.height
            elif self.orientation == "bottom": distance = 1.0 - (y - self.position[1]) / self.height
            else: return 0.0
        
        else: # corner PML
            # Calculate distance from corner to point
            distance_from_corner = np.hypot(x - self.position[0], y - self.position[1])
            # Outside the PML region
            if distance_from_corner > self.radius: return 0.0
            # Check if in correct quadrant
            dx_from_corner = x - self.position[0]
            dy_from_corner = y - self.position[1]
            if self.orientation == "top-left" and (dx_from_corner > 0 or dy_from_corner < 0): return 0.0
            elif self.orientation == "top-right" and (dx_from_corner < 0 or dy_from_corner < 0): return 0.0
            elif self.orientation == "bottom-left" and (dx_from_corner > 0 or dy_from_corner > 0): return 0.0
            elif self.orientation == "bottom-right" and (dx_from_corner < 0 or dy_from_corner > 0): return 0.0
            # Normalize distance (0 at inner edge, 1 at corner)
            distance = distance_from_corner / self.radius
        
        # Get optimized profile values
        sigma_profile, alpha_profile = self.get_profile(distance)
        # Apply stretched-coordinate PML with gradual absorption
        conductivity = sigma_max * sigma_profile
        # The material-dependent scaling might have been causing excessive reflection
        # We'll use a gentler approach that smoothly transitions at the boundary
        if dt is not None:
            # Apply frequency-shifting with reduced effect near boundary
            frequency_factor = 1.0 / (1.0 + alpha_profile)
            conductivity *= frequency_factor
            
        return conductivity