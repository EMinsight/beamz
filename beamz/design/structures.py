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
        if auto_pml: self.init_boundaries(pml_size)

    def add(self, structure):
        """Add structures on top of the design."""
        if isinstance(structure, ModeSource):
            self.sources.append(structure)
            self.structures.append(structure)
        elif isinstance(structure, GaussianSource):
            self.sources.append(structure)
            self.structures.append(structure)
        elif isinstance(structure, Monitor):
            self.monitors.append(structure)
            self.structures.append(structure)
        else:
            self.structures.append(structure)
        if hasattr(structure, 'z') or hasattr(structure, 'depth'): self.is_3d = True

    def scatter(self, structure, n=1000, xyrange=(-5*µm, 5*µm), scale_range=(0.05, 1)):
        """Randomly distribute a given object over the design domain."""
        for _ in range(n):
            new_structure = structure.copy()
            new_structure.shift(random.uniform(xyrange[0], xyrange[1]), random.uniform(xyrange[0], xyrange[1]))
            new_structure.rotate(random.uniform(0, 360))
            new_structure.scale(random.uniform(scale_range[0], scale_range[1]))
            self.add(new_structure)

    def init_boundaries(self, pml_size=None):
        """Add boundary conditions to the design area (using PML)."""
        # Calculate PML size more intelligently if not specified
        if pml_size is None:
            # Find max permittivity in design for wavelength calculation
            max_permittivity = 1.0
            for structure in self.structures:
                if hasattr(structure, 'material') and hasattr(structure.material, 'permittivity'):
                    max_permittivity = max(max_permittivity, structure.material.permittivity)
            # Estimate minimum wavelength (assuming 1550nm free space wavelength typical for photonics)
            # This is a practical approximation for common photonic applications
            wavelength_estimate = 1.55e-6 / np.sqrt(max_permittivity)
            # Set PML size to be at least 1 wavelength and at most 20% of domain size
            min_size = wavelength_estimate
            max_size = min(self.width, self.height) * 0.2
            # Use a heuristic to set PML size based on domain dimensions and wavelength
            # At least 1 wavelength and at most 20% of the domain size
            pml_size = max(min_size, min(max_size, min(self.width, self.height) / 5))
            print(f"Auto-selected PML size: {pml_size:.2e} m (~{pml_size/wavelength_estimate:.1f} wavelengths)")
        # Create transparent material for PML outlines
        pml_material = Material(permittivity=1.0, permeability=1.0, conductivity=0.0)
        # Edges of the design domain - create functional PML boundaries
        self.boundaries.append(RectPML(position=(0, 0), width=pml_size, height=self.height, orientation="left"))
        self.boundaries.append(RectPML(position=(self.width - pml_size, 0), width=pml_size, height=self.height, orientation="right"))
        self.boundaries.append(RectPML(position=(0, self.height - pml_size), width=self.width, height=pml_size, orientation="top"))
        self.boundaries.append(RectPML(position=(0, 0), width=self.width, height=pml_size, orientation="bot"))
        # Corners of the design domain - create functional PML boundaries
        self.boundaries.append(CircularPML(position=(0, 0), radius=pml_size, orientation="top-left"))
        self.boundaries.append(CircularPML(position=(self.width, 0), radius=pml_size, orientation="top-right"))
        self.boundaries.append(CircularPML(position=(0, self.height), radius=pml_size, orientation="bottom-left"))
        self.boundaries.append(CircularPML(position=(self.width, self.height), radius=pml_size, orientation="bottom-right"))
        # Add visual representations of PML regions to the structures list
        # Left PML region
        left_pml = Rectangle(
            position=(0, 0),
            width=pml_size,
            height=self.height,
            material=pml_material,
            color='none',
            is_pml=True
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

    def show(self):
        """Display the design visually."""
        if not self.structures: print("No structures to display"); return
        # Determine appropriate SI unit and scale
        max_dim = max(self.width, self.height)
        scale, unit = get_si_scale_and_label(max_dim)
        print("Showing 2D design...")
        # Calculate figure size based on domain dimensions
        aspect_ratio = self.width / self.height
        base_size = 5
        if aspect_ratio > 1: figsize = (base_size * aspect_ratio, base_size)
        else: figsize = (base_size, base_size / aspect_ratio)
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
        # First check if we're in a PML boundary region
        pml_conductivity = 0.0
        # Calculate average permittivity in the domain for PML calculation
        if dx is not None:
            eps_values = []
            for structure in self.structures:
                if hasattr(structure, 'material') and hasattr(structure.material, 'permittivity'):
                    eps_values.append(structure.material.permittivity)
            eps_avg = np.mean(eps_values) if eps_values else 1.0
        else: eps_avg = None
        # Apply all PML boundaries
        for boundary in self.boundaries:
            if isinstance(boundary, RectPML) or isinstance(boundary, CircularPML):
                pml_conductivity += boundary.get_conductivity(x, y, dx=dx, dt=dt, eps_avg=eps_avg)
        # Get material values from structures
        for structure in reversed(self.structures):
            if isinstance(structure, Rectangle):
                if self._point_in_polygon(x, y, structure.vertices):
                    # Return with added PML conductivity
                    return [structure.material.permittivity, 
                            structure.material.permeability, 
                            structure.material.conductivity + pml_conductivity]
            elif isinstance(structure, Circle):
                if np.hypot(x - structure.position[0], y - structure.position[1]) <= structure.radius:
                    # Return with added PML conductivity
                    return [structure.material.permittivity, 
                            structure.material.permeability, 
                            structure.material.conductivity + pml_conductivity]
            elif isinstance(structure, Ring):
                distance = np.hypot(x - structure.position[0], y - structure.position[1])
                if structure.inner_radius <= distance <= structure.outer_radius:
                    # Return with added PML conductivity
                    return [structure.material.permittivity, 
                            structure.material.permeability, 
                            structure.material.conductivity + pml_conductivity]
            elif isinstance(structure, CircularBend):
                distance = np.hypot(x - structure.position[0], y - structure.position[1])
                if structure.inner_radius <= distance <= structure.outer_radius:
                    # Return with added PML conductivity
                    return [structure.material.permittivity, 
                            structure.material.permeability, 
                            structure.material.conductivity + pml_conductivity]
            elif isinstance(structure, Polygon):
                if self._point_in_polygon(x, y, structure.vertices):
                    # Return with added PML conductivity
                    return [structure.material.permittivity, 
                            structure.material.permeability, 
                            structure.material.conductivity + pml_conductivity]
        # Default with added PML conductivity
        return [1.0, 1.0, pml_conductivity]  # Default permittivity if no structure contains the point

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

class Polygon:
    def __init__(self, vertices=None, material=None, color=None, optimize=False):
        self.vertices = vertices
        self.material = material
        self.optimize = optimize
        self.color = color if color is not None else self.get_random_color_consistent()
    
    def get_random_color_consistent(self, saturation=0.5, value=0.5):
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
    def __init__(self, position=(0,0), inner_radius=1, outer_radius=2, material=None, color=None, optimize=False, points=64):
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

class RectPML:
    """Rectangular Perfectly Matched Layer (PML) for absorbing boundary conditions."""
    def __init__(self, position=(0,0), width=1, height=1, orientation="left", max_conductivity=None, polynomial_order=3):
        self.position = position
        self.width = width
        self.height = height
        self.orientation = orientation
        self.polynomial_order = polynomial_order
        # Default max conductivity or let it be calculated later
        self.max_conductivity = max_conductivity if max_conductivity is not None else 1.0
        
    def get_conductivity(self, x, y, dx=None, dt=None, eps_avg=None):
        """Calculate PML conductivity at a point.
        
        Args:
            x, y: Position to evaluate
            dx: Grid cell size (if None, use default max_conductivity)
            dt: Time step size (if None, use default max_conductivity)
            eps_avg: Average permittivity (if None, use default max_conductivity)
            
        Returns:
            Conductivity value at the point
        """
        # Check if point is within the PML region
        if not (self.position[0] <= x <= self.position[0] + self.width and
                self.position[1] <= y <= self.position[1] + self.height):
            return 0.0
        # Calculate max conductivity based on grid parameters if provided
        sigma_max = self.max_conductivity
        if dx is not None and eps_avg is not None:
            # Calculate impedance and max conductivity as in pml.py
            eta = np.sqrt(MU_0 / (EPS_0 * eps_avg))
            sigma_max = 0.5 / (eta * dx)
        # Calculate normalized distance from boundary based on orientation
        if self.orientation == "left": distance = (x - self.position[0]) / self.width
        elif self.orientation == "right": distance = 1.0 - (x - self.position[0]) / self.width
        elif self.orientation == "top": distance = 1.0 - (y - self.position[1]) / self.height
        elif self.orientation == "bottom": distance = (y - self.position[1]) / self.height
        else: return 0.0  # Default if orientation is not recognized
        # Ensure distance is within [0,1]
        distance = min(max(distance, 0.0), 1.0)
        # Calculate conductivity with polynomial scaling (default cubic as in pml.py)
        return sigma_max * (1.0 - distance) ** self.polynomial_order


class CircularPML:
    """Circular corner PML for better wave absorption at corners."""
    def __init__(self, position=(0,0), radius=1, orientation="top-left", max_conductivity=None, polynomial_order=3):
        self.position = position
        self.radius = radius
        self.orientation = orientation
        self.polynomial_order = polynomial_order
        # Default max conductivity or let it be calculated later
        self.max_conductivity = max_conductivity if max_conductivity is not None else 1.0
        
    def get_conductivity(self, x, y, dx=None, dt=None, eps_avg=None):
        """Calculate PML conductivity at a point.
        
        Args:
            x, y: Position to evaluate
            dx: Grid cell size (if None, use default max_conductivity)
            dt: Time step size (if None, use default max_conductivity)
            eps_avg: Average permittivity (if None, use default max_conductivity)
            
        Returns:
            Conductivity value at the point
        """
        # Calculate distance from corner to point (x,y)
        distance_from_corner = np.hypot(x - self.position[0], y - self.position[1])
        # Scale distance based on radius
        if distance_from_corner > self.radius: return 0.0  # Outside the PML region
        # For corners, we need to check if the point is in the correct quadrant
        # This prevents the corner PML from extending into the non-corner regions
        dx = x - self.position[0]
        dy = y - self.position[1]
        if self.orientation == "top-left" and (dx > 0 or dy > 0): return 0.0
        elif self.orientation == "top-right" and (dx < 0 or dy > 0): return 0.0
        elif self.orientation == "bottom-left" and (dx > 0 or dy < 0): return 0.0
        elif self.orientation == "bottom-right" and (dx < 0 or dy < 0): return 0.0
        # Calculate max conductivity based on grid parameters if provided
        sigma_max = self.max_conductivity
        if dx is not None and eps_avg is not None:
            # Calculate impedance and max conductivity as in pml.py
            eta = np.sqrt(MU_0 / (EPS_0 * eps_avg))
            sigma_max = 0.5 / (eta * dx)
        # Normalize distance to [0,1] range (1 at corner, 0 at edge of PML)
        normalized_distance = 1.0 - (distance_from_corner / self.radius)
        # Calculate conductivity with polynomial scaling (default cubic as in pml.py)
        return sigma_max * normalized_distance ** self.polynomial_order