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

class Design:
    # TODO: Implement 3D version generalization.
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
        self.is_3d = False
        if auto_pml:
            self.init_boundaries(pml_size)

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
        for i in range(n):
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
            is_pml=True  # Mark as PML region
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
        if not self.structures:
            print("No structures to display")
            return

        # Determine appropriate SI unit and scale
        max_dim = max(self.width, self.height)
        scale, unit = get_si_scale_and_label(max_dim)
            
        if self.is_3d:
            print("3D design not yet supported...")
        else:
            print("Showing 2D design...")
            # Calculate figure size based on domain dimensions
            aspect_ratio = self.width / self.height
            base_size = 5  # Slightly larger base size for single plot
            if aspect_ratio > 1: figsize = (base_size * aspect_ratio, base_size)
            else: figsize = (base_size, base_size / aspect_ratio)
            
            # Create a single figure for all structures
            fig, ax = plt.subplots(figsize=figsize)
            
            # Set equal aspect ratio explicitly
            ax.set_aspect('equal')
            
            # Now plot each structure
            for structure in self.structures:
                if isinstance(structure, Rectangle):
                    if structure.is_pml:
                        # PML regions get outlined with dashed lines
                        rect = MatplotlibRectangle(
                            (structure.position[0], structure.position[1]),
                            structure.width, structure.height,
                            facecolor='none', edgecolor='black', linestyle=':', alpha=1.0, linewidth=1.0, zorder=10)
                    else:
                        # Normal structures get solid fill
                        rect = MatplotlibRectangle(
                            (structure.position[0], structure.position[1]),
                            structure.width, structure.height,
                            facecolor=structure.color, edgecolor=self.border_color, alpha=1)
                    ax.add_patch(rect)
                elif isinstance(structure, Circle):
                    circle = plt.Circle(
                        (structure.position[0], structure.position[1]),
                        structure.radius,
                        facecolor=structure.color, edgecolor=self.border_color, alpha=1)
                    ax.add_patch(circle)
                elif isinstance(structure, Ring):
                    # Create points for the ring
                    N = 100  # Number of points for each circle
                    theta = np.linspace(0, 2 * np.pi, N, endpoint=True)
                    # Outer circle points (counterclockwise)
                    x_outer = structure.position[0] + structure.outer_radius * np.cos(theta)
                    y_outer = structure.position[1] + structure.outer_radius * np.sin(theta)
                    # Inner circle points (clockwise)
                    x_inner = structure.position[0] + structure.inner_radius * np.cos(theta[::-1])
                    y_inner = structure.position[1] + structure.inner_radius * np.sin(theta[::-1])
                    # Combine vertices
                    vertices = np.vstack([np.column_stack([x_outer, y_outer]),
                                        np.column_stack([x_inner, y_inner])])
                    # Define path codes
                    codes = np.concatenate([[Path.MOVETO] + [Path.LINETO] * (N - 1),
                                          [Path.MOVETO] + [Path.LINETO] * (N - 1)])
                    # Create the path and patch
                    path = Path(vertices, codes)
                    ring_patch = PathPatch(path, facecolor=structure.color, alpha=1, edgecolor=self.border_color)
                    ax.add_patch(ring_patch)
                elif isinstance(structure, CircularBend):
                    # Create points for the bend
                    N = 100  # Number of points for each arc
                    # Convert angles to radians
                    angle_rad = np.radians(structure.angle)
                    rotation_rad = np.radians(structure.rotation)
                    theta = np.linspace(rotation_rad, rotation_rad + angle_rad, N, endpoint=True)
                    # Outer arc points
                    x_outer = structure.position[0] + structure.outer_radius * np.cos(theta)
                    y_outer = structure.position[1] + structure.outer_radius * np.sin(theta)
                    # Inner arc points
                    x_inner = structure.position[0] + structure.inner_radius * np.cos(theta)
                    y_inner = structure.position[1] + structure.inner_radius * np.sin(theta)
                    # Create a closed path by combining points and adding connecting lines
                    vertices = np.vstack([
                        # Start at outer arc beginning
                        [x_outer[0], y_outer[0]],
                        # Draw outer arc
                        *np.column_stack([x_outer[1:], y_outer[1:]]),
                        # Connect to inner arc end
                        [x_inner[-1], y_inner[-1]],
                        # Draw inner arc backwards
                        *np.column_stack([x_inner[-2::-1], y_inner[-2::-1]]),
                        # Close the path by returning to start
                        [x_outer[0], y_outer[0]]
                    ])
                    # Define path codes for a single continuous path
                    codes = [Path.MOVETO] + \
                           [Path.LINETO] * (len(vertices) - 2) + \
                           [Path.CLOSEPOLY]
                    # Create the path and patch
                    path = Path(vertices, codes)
                    bend_patch = PathPatch(path, facecolor=structure.color, alpha=1, edgecolor=self.border_color)
                    ax.add_patch(bend_patch)
                elif isinstance(structure, Polygon):
                    polygon = plt.Polygon(structure.vertices, facecolor=structure.color, alpha=1, edgecolor=self.border_color)
                    ax.add_patch(polygon)
                elif isinstance(structure, ModeSource):
                    ax.plot((structure.start[0], structure.end[0]), (structure.start[1], structure.end[1]), '-', lw=4, color="crimson", label='Mode Source')
                    ax.plot((structure.start[0], structure.end[0]), (structure.start[1], structure.end[1]), '--', lw=2, color="black", label='Mode Source')
                elif isinstance(structure, Monitor):
                    ax.plot((structure.start[0], structure.end[0]), (structure.start[1], structure.end[1]), '-', lw=4, color="navy", label='Monitor')
                    ax.plot((structure.start[0], structure.end[0]), (structure.start[1], structure.end[1]), '--', lw=2, color="black", label='Monitor')
                elif isinstance(structure, GaussianSource):
                    pass

            # Set proper limits, title and labels
            ax.set_title('Design Layout')
            ax.set_xlabel(f'X ({unit})')
            ax.set_ylabel(f'Y ({unit})')
            # Set axis limits and ensure the full design is visible
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
        else:
            eps_avg = None
            
        # Apply all PML boundaries
        for boundary in self.boundaries:
            if isinstance(boundary, RectPML) or isinstance(boundary, CircularPML):
                pml_conductivity += boundary.get_conductivity(x, y, dx=dx, dt=dt, eps_avg=eps_avg)
        
        # Get material values from structures
        for structure in reversed(self.structures):
            if isinstance(structure, Rectangle):
                if (structure.position[0] <= x <= structure.position[0] + structure.width and
                    structure.position[1] <= y <= structure.position[1] + structure.height):
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


# ================================================ 2D structures
class Polygon:
    def __init__(self, vertices=None, material=None, color=None, optimize=False):
        self.vertices = vertices
        self.material = material
        self.optimize = optimize
        self.color = color if color is not None else self.get_random_color()

    def get_random_color(self):
        return '#{:06x}'.format(random.randint(0, 0xFFFFFF))
    
    def shift(self, x, y):
        """Shift the polygon by (x,y) and return self for method chaining."""
        if self.vertices:
            self.vertices = [(v[0] + x, v[1] + y) for v in self.vertices]
        return self
    
    def scale(self, s):
        """Scale the polygon around its center of mass and return self for method chaining."""
        if self.vertices:
            # Calculate center of mass
            x_center = sum(v[0] for v in self.vertices) / len(self.vertices)
            y_center = sum(v[1] for v in self.vertices) / len(self.vertices)
            # Shift to origin, scale, then shift back
            self.vertices = [
                (x_center + (v[0] - x_center) * s,
                 y_center + (v[1] - y_center) * s)
                for v in self.vertices
            ]
        return self
    
    def rotate(self, angle):
        """Rotate the polygon around its center of mass and return self for method chaining."""
        if self.vertices:
            # Calculate center of mass
            x_center = sum(v[0] for v in self.vertices) / len(self.vertices)
            y_center = sum(v[1] for v in self.vertices) / len(self.vertices)
            # Shift to origin, rotate, then shift back
            self.vertices = [
                (x_center + (v[0] - x_center) * np.cos(angle) - (v[1] - y_center) * np.sin(angle),
                 y_center + (v[0] - x_center) * np.sin(angle) + (v[1] - y_center) * np.cos(angle))
                for v in self.vertices
            ]
        return self

    def copy(self):
        return Polygon(self.vertices, self.material)

class Rectangle(Polygon):
    def __init__(self, position=(0,0), width=1, height=1, material=None, color=None, is_pml=False, optimize=False):
        self.position = position
        self.width = width
        self.height = height
        self.material = material
        self.is_pml = is_pml
        self.optimize = optimize
        self.color = color if color is not None else self.get_random_color()

    def get_random_color(self):
        return '#{:06x}'.format(random.randint(0, 0xFFFFFF))
    
    def shift(self, x, y):
        self.position = (self.position[0] + x, self.position[1] + y)
        return self
    
    def rotate(self, angle):
        """Rotate the rectangle around its center of mass and return self for method chaining."""
        pass

    def scale(self, s):
        """Scale the rectangle around its center of mass and return self for method chaining."""
        # Calculate center of mass
        x_center = self.position[0] + self.width/2
        y_center = self.position[1] + self.height/2
        # Get current position relative to center
        x_rel = self.width/2
        y_rel = self.height/2
        # Scale relative position and dimensions
        x_new = x_rel * s
        y_new = y_rel * s
        self.width *= s
        self.height *= s
        # Update position by adding center back
        self.position = (x_center + x_new, y_center + y_new)
        return self
    
    def copy(self):
        return Rectangle(self.position, self.width, self.height, self.material, self.color, self.is_pml)

class Circle:
    def __init__(self, position=(0,0), radius=1, material=None, color=None, optimize=False):
        self.position = position
        self.radius = radius
        self.material = material
        self.optimize = optimize
        self.color = color if color is not None else self.get_random_color()
    
    def get_random_color(self):
        return '#{:06x}'.format(random.randint(0, 0xFFFFFF))
    
    def shift(self, x, y):
        """Shift the circle by (x,y) and return self for method chaining."""
        self.position = (self.position[0] + x, self.position[1] + y)
        return self
    
    def rotate(self, angle):
        pass

    def scale(self, s):
        """Scale the circle radius by s and return self for method chaining."""
        self.radius *= s
        return self
    
    def copy(self):
        return Circle(self.position, self.radius, self.material)

class Ring:
    def __init__(self, position=(0,0), inner_radius=1, outer_radius=2, material=None, color=None, optimize=False):
        self.position = position
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.material = material
        self.optimize = optimize
        self.color = color if color is not None else self.get_random_color()

    def get_random_color(self):
        return '#{:06x}'.format(random.randint(0, 0xFFFFFF))
    
    # TODO
    def shift(self, x, y):
        pass

    # TODO
    def rotate(self, angle):
        pass

    # TODO
    def scale(self, s):
        pass
    
    def copy(self):
        return Ring(self.position, self.inner_radius, self.outer_radius, self.material)

class CircularBend:
    def __init__(self, position=(0,0), inner_radius=1, outer_radius=2, angle=90, rotation=0, material=None, color=None, optimize=False):
        self.position = position
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.angle = angle
        self.rotation = rotation
        self.material = material
        self.optimize = optimize
        self.color = color if color is not None else self.get_random_color()

    def get_random_color(self):
        return '#{:06x}'.format(random.randint(0, 0xFFFFFF))

    def shift(self, x, y):
        pass

    # TODO
    def rotate(self, angle):
        pass

    # TODO
    def scale(self, s):
        pass
    
    def copy(self):
        return CircularBend(self.position, self.inner_radius, self.outer_radius, self.angle, self.rotation, self.material)

class Taper(Polygon):
    """Taper is a structure that tapers from a width to a height."""
    def __init__(self, position=(0,0), input_width=1, output_width=0.5, length=1, material=None, color=None, optimize=False):
        # Calculate vertices for the trapezoid shape
        x, y = position
        vertices = [
            (x, y - input_width/2),  # Bottom left
            (x + length, y - output_width/2),  # Bottom right
            (x + length, y + output_width/2),  # Top right
            (x, y + input_width/2),  # Top left
        ]
        super().__init__(vertices=vertices, material=material, color=color)
        self.position = position
        self.input_width = input_width
        self.output_width = output_width
        self.length = length
        self.optimize = optimize
    def copy(self):
        return Taper(self.position, self.input_width, self.output_width, self.length, self.material)


# ================================================ 2D Boundaries

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
        if self.orientation == "left":
            distance = (x - self.position[0]) / self.width
        elif self.orientation == "right":
            distance = 1.0 - (x - self.position[0]) / self.width
        elif self.orientation == "top":
            distance = 1.0 - (y - self.position[1]) / self.height
        elif self.orientation == "bot":
            distance = (y - self.position[1]) / self.height
        else:
            return 0.0  # Default if orientation is not recognized
        
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


# ================================================ 3D structures