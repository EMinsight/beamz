import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MatplotlibRectangle, PathPatch, Circle as MatplotlibCircle
from matplotlib.path import Path
import random
import numpy as np
from beamz.design.materials import Material
from beamz.const import µm, EPS_0, MU_0
from beamz.design.sources import ModeSource, GaussianSource
from beamz.design.monitors import Monitor
from beamz.design.helpers import get_si_scale_and_label
from beamz.helpers import display_header, display_status, display_design_summary, tree_view, console
import colorsys

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
        if auto_pml: self.init_boundaries(pml_size)
        display_status(f"Created design with size: {self.width:.2e} x {self.height:.2e} m")
        
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
                shapely_polygon = ShapelyPolygon(structure.vertices)
                if shapely_polygon.is_valid:
                    material_groups[material_key].append((structure, shapely_polygon))
                    structures_to_remove.append(structure)
                else:
                    display_status(f"Skipping invalid polygon: {structure}", "warning")
                    non_polygon_structures.append(structure)
            except Exception as e:
                display_status(f"Error converting structure to Shapely polygon: {e}", "warning")
                non_polygon_structures.append(structure)
        
        # Second pass: unify polygons within each material group
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
                    # Create a new polygon with the merged vertices
                    vertices = list(merged.exterior.coords[:-1])  # Exclude the last point which repeats the first
                    new_poly = Polygon(vertices=vertices, material=material)
                    new_structures.append(new_poly)
                    display_status(f"Unified {len(structure_group)} polygons with permittivity={material_key[0]}", "success")
                elif merged.geom_type == 'MultiPolygon':
                    # Create multiple polygons if the merger resulted in separate shapes
                    for geom in merged.geoms:
                        vertices = list(geom.exterior.coords[:-1])
                        new_poly = Polygon(vertices=vertices, material=material)
                        new_structures.append(new_poly)
                    display_status(f"Unified into {len(merged.geoms)} separate polygons with permittivity={material_key[0]}", "success")
                else:
                    # If the result is something unexpected, keep the original structures
                    display_status(f"Unexpected geometry type: {merged.geom_type}, keeping original structures", "warning")
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
        
        # Add the unified structures and non-polygon structures back
        self.structures.extend(new_structures)
        
        # Final report
        display_status(f"Polygon unification complete: {len(structures_to_remove)} structures merged into {len(new_structures)} unified shapes", "success")
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
        for structure in self.structures: structure.add_to_plot(ax)
        
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
    def __init__(self, vertices=None, material=None, color=None, optimize=False):
        self.vertices = vertices
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
        return self
    
    def scale(self, s):
        """Scale the polygon around its center of mass and return self for method chaining."""
        if self.vertices:
            # Calculate center of mass
            x_center = sum(v[0] for v in self.vertices) / len(self.vertices)
            y_center = sum(v[1] for v in self.vertices) / len(self.vertices)
            # Shift to origin, scale, then shift back
            self.vertices = [(x_center + (v[0] - x_center) * s,
                              y_center + (v[1] - y_center) * s)
                              for v in self.vertices]
        return self
    
    def rotate(self, angle, point=None):
        """Rotate the polygon around its center of mass or specified point.
        angle: Rotation angle in degrees
        point: Optional (x,y) point to rotate around. If None, rotates around center.
        """
        if self.vertices:
            angle_rad = np.radians(angle)
            if point is None:
                # Calculate center of mass
                x_center = sum(v[0] for v in self.vertices) / len(self.vertices)
                y_center = sum(v[1] for v in self.vertices) / len(self.vertices)
            else:
                x_center, y_center = point
            # Shift to origin, rotate, then shift back
            self.vertices = [
                (x_center + (v[0] - x_center) * np.cos(angle_rad) - (v[1] - y_center) * np.sin(angle_rad),
                 y_center + (v[0] - x_center) * np.sin(angle_rad) + (v[1] - y_center) * np.cos(angle_rad))
                for v in self.vertices
            ]
        return self

    def add_to_plot(self, ax, facecolor=None, edgecolor="black", alpha=None, linestyle=None):
        """Add the rectangle as a patch to the axis of a given figure."""
        if facecolor is None: facecolor = self.color
        if alpha is None: alpha = 1
        if linestyle is None: linestyle = '-'
        patch = plt.Polygon(self.vertices, facecolor=facecolor,
                alpha=alpha, edgecolor=edgecolor, linestyle=linestyle)
        ax.add_patch(patch)

    def copy(self):
        return Polygon(self.vertices, self.material)
        
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
        
    def _point_in_polygon(self, x, y, vertices=None):
        """Check if a point (x,y) is inside this polygon.
        
        Uses the ray-casting algorithm.
        """
        if vertices is None:
            vertices = self.vertices
            
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
        return Circle(self.position, self.radius, self.material, self.color, self.optimize)

class Ring(Polygon):
    def __init__(self, position=(0,0), inner_radius=1, outer_radius=2, material=None, color=None, optimize=False, points=256):
        theta = np.linspace(0, 2*np.pi, points, endpoint=False)
        outer_vertices = [(position[0] + outer_radius * np.cos(t), position[1] + outer_radius * np.sin(t)) for t in theta]
        inner_vertices = [(position[0] + inner_radius * np.cos(t), position[1] + inner_radius * np.sin(t)) for t in reversed(theta)]
        vertices = outer_vertices + inner_vertices
        super().__init__(vertices=vertices, material=material, color=color, optimize=optimize)
        self.points = points
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
        # Create points for the ring
        theta = np.linspace(0, 2 * np.pi, self.points, endpoint=True)
        # Outer circle points (counterclockwise) and inner circle points (clockwise)
        x_outer = self.position[0] + self.outer_radius * np.cos(theta)
        y_outer = self.position[1] + self.outer_radius * np.sin(theta)
        x_inner = self.position[0] + self.inner_radius * np.cos(theta[::-1])
        y_inner = self.position[1] + self.inner_radius * np.sin(theta[::-1])
        # Combine vertices
        vertices = np.vstack([np.column_stack([x_outer, y_outer]), np.column_stack([x_inner, y_inner])])
        # Define path codes
        codes = np.concatenate([[Path.MOVETO] + [Path.LINETO] * (self.points - 1),
                                [Path.MOVETO] + [Path.LINETO] * (self.points - 1)])
        # Create the path and patch
        path = Path(vertices, codes)
        ring_patch = PathPatch(path, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor, linestyle=linestyle)
        ax.add_patch(ring_patch)
    
    def copy(self):
        return Ring(self.position, self.inner_radius, self.outer_radius, self.material, self.color, self.optimize)

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
        
        if region_type == "rect":
            self.width, self.height = size
        else:  # corner
            self.radius = size
    
    def get_profile(self, normalized_distance):
        """Calculate PML absorption profile using gradual grading.
        
        Args:
            normalized_distance: Distance from PML inner boundary (0.0) to outer boundary (1.0)
            
        Returns:
            Tuple of (sigma, alpha) values for conductivity and frequency-shifting
        """
        # Ensure distance is within [0,1]
        d = min(max(normalized_distance, 0.0), 1.0)
        
        # Create a smooth transition from 0 at the interface
        # Start with nearly zero conductivity at the interface and gradually increase
        # This is crucial to prevent reflection at the boundary
        if d < 0.05:
            # Very gentle start at the boundary (nearly zero)
            sigma = 0.01 * (d/0.05)**2
        else:
            # Smooth polynomial grading for the rest
            sigma = ((d - 0.05) / 0.95)**self.polynomial_order
        
        # Smooth frequency-shifting profile
        alpha = self.alpha_max * d**2  # Quadratic profile for smooth transition
        
        return sigma, alpha
    
    def get_conductivity(self, x, y, dx=None, dt=None, eps_avg=None):
        """Calculate PML conductivity at a point using smooth-transition PML.
        
        Args:
            x, y: Position to evaluate
            dx: Grid cell size
            dt: Time step size
            eps_avg: Average permittivity
            
        Returns:
            Conductivity value at the point
        """
        # Calculate theoretical optimal conductivity based on impedance matching
        if dx is not None and eps_avg is not None:
            # Calculate impedance 
            eta = np.sqrt(MU_0 / (EPS_0 * eps_avg))
            # Optimal conductivity for minimal reflection at interface
            # Reduced from 1.2 to 0.8 for smoother transition
            sigma_max = 0.8 / (eta * dx)
            sigma_max *= self.sigma_factor  # Apply gentler factor
        else:
            sigma_max = 1.0  # Lower default conductivity
        
        # Get normalized distance based on region type and orientation
        if self.region_type == "rect":
            # Check if point is within rectangular PML region
            if not (self.position[0] <= x <= self.position[0] + self.width and
                    self.position[1] <= y <= self.position[1] + self.height):
                return 0.0
                
            # Calculate normalized distance from boundary based on orientation
            # Distance should be 0 at inner boundary and 1 at outer boundary
            if self.orientation == "left":
                # For left PML, x=position[0]+width is inner (0), x=position[0] is outer (1)
                distance = 1.0 - (x - self.position[0]) / self.width
            elif self.orientation == "right":
                # For right PML, x=position[0] is inner (0), x=position[0]+width is outer (1)
                distance = (x - self.position[0]) / self.width
            elif self.orientation == "top":
                # For top PML, y=position[1] is inner (0), y=position[1]+height is outer (1)
                distance = (y - self.position[1]) / self.height
            elif self.orientation == "bottom":
                # For bottom PML, y=position[1]+height is inner (0), y=position[1] is outer (1)
                distance = 1.0 - (y - self.position[1]) / self.height
            else:
                return 0.0
        
        else:  # corner PML
            # Calculate distance from corner to point
            distance_from_corner = np.hypot(x - self.position[0], y - self.position[1])
            # Outside the PML region
            if distance_from_corner > self.radius:
                return 0.0
                
            # Check if in correct quadrant
            dx_from_corner = x - self.position[0]
            dy_from_corner = y - self.position[1]
            
            if self.orientation == "top-left" and (dx_from_corner > 0 or dy_from_corner < 0):
                return 0.0
            elif self.orientation == "top-right" and (dx_from_corner < 0 or dy_from_corner < 0):
                return 0.0
            elif self.orientation == "bottom-left" and (dx_from_corner > 0 or dy_from_corner > 0):
                return 0.0
            elif self.orientation == "bottom-right" and (dx_from_corner < 0 or dy_from_corner > 0):
                return 0.0
                
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