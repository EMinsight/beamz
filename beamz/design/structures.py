import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MatplotlibRectangle, PathPatch, Circle as MatplotlibCircle
from matplotlib.path import Path
import random
import numpy as np
from beamz.design.materials import Material
from beamz.const import µm
from beamz.design.sources import ModeSource
from beamz.design.monitors import ModeMonitor

def rgb_to_hex(r, g, b):
    """Convert RGB values to the hex color format used in get_random_color().
    Returns a string in the format '#{:06x}'."""
    return f'#{(r << 16) + (g << 8) + b:06x}'

def is_dark(color):
    """Check if a color is dark using relative luminance calculation. """
    # Remove '#' if present
    color = color.lstrip('#')
    # Convert hex to RGB components
    r = int(color[0:2], 16) / 255.0
    g = int(color[2:4], 16) / 255.0
    b = int(color[4:6], 16) / 255.0
    # Calculate relative luminance using sRGB coefficients
    # These coefficients account for human perception of brightness
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    # Return True if luminance is less than 0.5 (midpoint)
    return luminance < 0.5

def get_si_scale_and_label(value):
    """Convert a value to appropriate SI unit and return scale factor and label."""
    if value >= 1e-3: return 1e3, 'mm'
    elif value >= 1e-6: return 1e6, 'µm'
    elif value >= 1e-9: return 1e9, 'nm'
    else: return 1e12, 'pm'

class Design:
    # TODO: Implement 3D version generalization.
    def __init__(self, width=1, height=1, depth=None, material=None, color=None, border_color="black"):
        if material is None:
            material = Material(permittivity=1.0, permeability=1.0, conductivity=0.0)
        self.structures = [Rectangle(position=(0,0), width=width, height=height, material=material, color=color)]
        self.sources = []
        self.monitors = []
        self.time = 0
        self.is_3d = False
        self.width = width
        self.height = height
        self.depth = depth
        self.border_color = border_color

    def add(self, structure):
        """Add structures on top of the design."""
        if isinstance(structure, ModeSource):
            self.sources.append(structure)
            self.time = structure.signal.time if structure.signal > self.time else self.time
        elif isinstance(structure, ModeMonitor):
            self.monitors.append(structure)
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

    # TODO
    def borders(self, structure=None, all=None, top=None, right=None, bottom=None, left=None):
        """Add boundary conditions to the design area (currently only supports none or PML)."""
        if all is PML:
            # Apply the same PML to all borders
            self.add(PML(position=(0, 0), width=self.width, thickness=all.thickness))
            self.add(PML(position=(self.width - all.thickness, 0), width=all.thickness, height=self.height))
            self.add(PML(position=(0, self.height - all.thickness), width=self.width, height=all.thickness))
            self.add(PML(position=(0, 0), width=all.thickness, height=self.height))
        elif all is None:
            # Apply individual PMLs to specified borders
            if top is PML: self.add(PML(position=(0, 0), width=self.width, thickness=top.thickness))
            if right is PML: self.add(PML(position=(self.width - right.thickness, 0), width=right.thickness, height=self.height))
            if bottom is PML: self.add(PML(position=(0, self.height - bottom.thickness), width=self.width, height=bottom.thickness))
            if left is PML: self.add(PML(position=(0, 0), width=left.thickness, height=self.height))
        else:
            raise ValueError("PML must be specified in borders()...")

    def show(self):
        """Display the design visually."""
        if not self.structures:
            print("No structures to display")
            return

        # Determine appropriate SI unit and scale
        max_dim = max(self.width, self.height)
        scale, unit = get_si_scale_and_label(max_dim)
            
        if self.is_3d:
            print("Showing 3D design...")
            # Create 3 subplots for 3D visualization
            fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot each structure in all three views
            for structure in self.structures:
                if isinstance(structure, Rectangle):
                    # XY view
                    rect = MatplotlibRectangle(
                        (structure.position[0], structure.position[1]),
                        structure.width, structure.height,
                        facecolor=structure.color, edgecolor=self.border_color, alpha=1)
                    ax_xy.add_patch(rect)
                    # XZ view
                    rect = MatplotlibRectangle(
                        (structure.position[0], structure.position[2]),
                        structure.width, structure.depth,
                        facecolor=structure.color, edgecolor=self.border_color, alpha=1)
                    ax_xz.add_patch(rect)
                    # YZ view
                    rect = MatplotlibRectangle(
                        (structure.position[1], structure.position[2]),
                        structure.height, structure.depth,
                        facecolor=structure.color, edgecolor=self.border_color, alpha=1)
                    ax_yz.add_patch(rect)
            
            # Set labels and titles with SI units
            ax_xy.set_title('XY View')
            ax_xy.set_xlabel(f'X ({unit})')
            ax_xy.set_ylabel(f'Y ({unit})')
            ax_xz.set_title('XZ View')
            ax_xz.set_xlabel(f'X ({unit})')
            ax_xz.set_ylabel(f'Z ({unit})')
            ax_yz.set_title('YZ View')
            ax_yz.set_xlabel(f'Y ({unit})')
            ax_yz.set_ylabel(f'Z ({unit})')
            # Set axis limits
            ax_xy.set_xlim(0, self.width)
            ax_xy.set_ylim(0, self.height)
            ax_xz.set_xlim(0, self.width)
            ax_xz.set_ylim(0, self.depth)
            ax_yz.set_xlim(0, self.height)
            ax_yz.set_ylim(0, self.depth)
            # Update tick labels with scaled values
            for ax in [ax_xy, ax_xz, ax_yz]:
                ax.xaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
                ax.yaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')

        else:
            print("Showing 2D design...")
            # Calculate figure size based on domain dimensions
            aspect_ratio = self.width / self.height
            base_size = 3  # Base size for the smaller dimension
            if aspect_ratio > 1: figsize = (base_size * aspect_ratio, base_size)
            else: figsize = (base_size, base_size / aspect_ratio)

            # Create single plot for 2D visualization
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot each structure
            for structure in self.structures:
                if isinstance(structure, Rectangle):
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
                elif isinstance(structure, ModeMonitor):
                    ax.plot((structure.start[0], structure.end[0]), (structure.start[1], structure.end[1]), '-', lw=4, color="navy", label='Mode Monitor')
            
            ax.set_title('2D Design')
            ax.set_xlabel(f'X ({unit})')
            ax.set_ylabel(f'Y ({unit})')
            # Set axis limits
            ax.set_xlim(0, self.width)
            ax.set_ylim(0, self.height)
            # Update tick labels with scaled values
            ax.xaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
            ax.yaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
        
        # Set equal aspect ratio and adjust layout
        plt.tight_layout()
        plt.show()
        
    def __str__(self):
        return f"Design with {len(self.structures)} structures ({'3D' if self.is_3d else '2D'})"

    def get_material_value(self, x, y):
        """Return the material value at a given (x, y) coordinate, prioritizing the topmost structure."""
        for structure in reversed(self.structures):
            if isinstance(structure, Rectangle):
                #print("In Rectangle:", x, y, structure.position[0], structure.position[1], structure.width, structure.height)
                if (structure.position[0] <= x <= structure.position[0] + structure.width and
                    structure.position[1] <= y <= structure.position[1] + structure.height):
                    return structure.material.permittivity
            elif isinstance(structure, Circle):
                #print("In Circle:", x, y, structure.position[0], structure.position[1], structure.radius)
                if np.hypot(x - structure.position[0], y - structure.position[1]) <= structure.radius:
                    return structure.material.permittivity
            elif isinstance(structure, Ring):
                #print("In Ring:", x, y, structure.position[0], structure.position[1], structure.inner_radius, structure.outer_radius)
                distance = np.hypot(x - structure.position[0], y - structure.position[1])
                if structure.inner_radius <= distance <= structure.outer_radius:
                    return structure.material.permittivity
            elif isinstance(structure, Polygon):
                #print("In Polygon:", x, y, structure.vertices)
                if self._point_in_polygon(x, y, structure.vertices):
                    return structure.material.permittivity
        #print("In Default")
        return 1.0  # Default permittivity if no structure contains the point

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
class Rectangle:
    def __init__(self, position=(0,0), width=1, height=1, material=None, color=None):
        self.position = position
        self.width = width
        self.height = height
        self.material = material
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
        return Rectangle(self.position, self.width, self.height, self.material)

class Circle:
    def __init__(self, position=(0,0), radius=1, material=None, color=None):
        self.position = position
        self.radius = radius
        self.material = material
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
    def __init__(self, position=(0,0), inner_radius=1, outer_radius=2, material=None, color=None):
        self.position = position
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.material = material
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
    def __init__(self, position=(0,0), inner_radius=1, outer_radius=2, angle=90, rotation=0, material=None, color=None):
        self.position = position
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.angle = angle
        self.rotation = rotation
        self.material = material
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

class Polygon:
    def __init__(self, vertices=None, material=None, color=None):
        self.vertices = vertices
        self.material = material
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

class Taper(Polygon):
    """Taper is a structure that tapers from a width to a height."""
    def __init__(self, position=(0,0), input_width=1, output_width=0.5, length=1, material=None, color=None):
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

    def copy(self):
        return Taper(self.position, self.input_width, self.output_width, self.length, self.material)

# ================================================ 2D Boundaries
# TODO
class PML:
    """Perfectly Matched Layer (PML) is a boundary condition that absorbs waves at the edges of the simulation domain."""
    def __init__(self, position=(0,0), width=1*µm, thickness=20*µm, sigma_max=1.0, m=3.0):
        self.position = position
        self.width = width
        self.thickness = thickness
        self.sigma_max = sigma_max
        self.m = m
        
    def get_conductivity_profile(self, distance):
        """Calculate the conductivity profile at a given distance from the PML boundary."""
        if distance < 0 or distance > self.thickness: return 0.0
        return self.sigma_max * (distance / self.thickness) ** self.m
                
    def __str__(self):
        return f"PML(thickness={self.thickness}, sigma_max={self.sigma_max}, m={self.m})"

# Perfectly Conducting (PEC)

# Perfectly Reflecting (PBR)

# Periodic (PER)

# ================================================ 3D structures

