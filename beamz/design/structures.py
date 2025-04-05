import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MatplotlibRectangle, PathPatch, Circle as MatplotlibCircle
from matplotlib.path import Path
import random
import numpy as np
from beamz.design.materials import Material
from beamz.const import µm

def get_si_scale_and_label(value):
    """Convert a value to appropriate SI unit and return scale factor and label."""
    if value >= 1e-3:  # mm
        return 1e3, 'mm'
    elif value >= 1e-6:  # µm
        return 1e6, 'µm'
    elif value >= 1e-9:  # nm
        return 1e9, 'nm'
    else:  # pm
        return 1e12, 'pm'

class Design:
    def __init__(self, width=1, height=1, depth=None, material=None):
        if material is None:
            material = Material(permittivity=1.0, permeability=1.0, conductivity=0.0)
        self.structures = [Rectangle(position=(0,0), width=width, height=height, material=material)]
        self.is_3d = False
        self.width = width
        self.height = height
        self.depth = depth
        
    def add(self, structure):
        self.structures.append(structure)
        # Check if the new structure is 3D
        if hasattr(structure, 'z') or hasattr(structure, 'depth'):
            self.is_3d = True

    def scatter(self, structure, n=1000, xyrange=(-5*µm, 5*µm), scale_range=(0.05, 1)):
        for i in range(n):
            new_structure = structure.copy()
            new_structure.shift(random.uniform(xyrange[0], xyrange[1]), random.uniform(xyrange[0], xyrange[1]))
            new_structure.rotate(random.uniform(0, 360))
            new_structure.scale(random.uniform(scale_range[0], scale_range[1]))
            self.add(new_structure)

    def show(self):
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
                        facecolor=structure.color, alpha=1)
                    ax_xy.add_patch(rect)
                    # XZ view
                    rect = MatplotlibRectangle(
                        (structure.position[0], structure.position[2]),
                        structure.width, structure.depth,
                        facecolor=structure.color, alpha=1)
                    ax_xz.add_patch(rect)
                    # YZ view
                    rect = MatplotlibRectangle(
                        (structure.position[1], structure.position[2]),
                        structure.height, structure.depth,
                        facecolor=structure.color, alpha=1)
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
            # Create single plot for 2D visualization
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Plot each structure
            for structure in self.structures:
                if isinstance(structure, Rectangle):
                    rect = MatplotlibRectangle(
                        (structure.position[0], structure.position[1]),
                        structure.width, structure.height,
                        facecolor=structure.color, edgecolor='black', alpha=1)
                    ax.add_patch(rect)
                elif isinstance(structure, Circle):
                    circle = plt.Circle(
                        (structure.position[0], structure.position[1]),
                        structure.radius,
                        facecolor=structure.color, edgecolor='black', alpha=1)
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
                    ring_patch = PathPatch(path, facecolor=structure.color, alpha=1, edgecolor='black')
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
                    bend_patch = PathPatch(path, facecolor=structure.color, alpha=1, edgecolor='black')
                    ax.add_patch(bend_patch)
                elif isinstance(structure, Polygon):
                    polygon = plt.Polygon(structure.vertices, facecolor=structure.color, alpha=1, edgecolor='black')
                    ax.add_patch(polygon)
            
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


# ================================================ 2D structures
class Rectangle:
    def __init__(self, position=(0,0), width=1, height=1, material=None):
        self.position = position
        self.width = width
        self.height = height
        self.material = material
        self.color = self.get_random_color()

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
    def __init__(self, position=(0,0), radius=1, material=None):
        self.position = position
        self.radius = radius
        self.material = material
        self.color = self.get_random_color()
    
    def get_random_color(self):
        return '#{:06x}'.format(random.randint(0, 0xFFFFFF))
    
    def shift(self, x, y):
        """Shift the circle by (x,y) and return self for method chaining."""
        self.position = (self.position[0] + x, self.position[1] + y)
        return self

    def scale(self, s):
        """Scale the circle radius by s and return self for method chaining."""
        self.radius *= s
        return self
    
    def copy(self):
        return Circle(self.position, self.radius, self.material)

class Ring:
    def __init__(self, position=(0,0), inner_radius=1, outer_radius=2, material=None):
        self.position = position
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.material = material
        self.color = self.get_random_color()

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
    def __init__(self, position=(0,0), inner_radius=1, outer_radius=2, angle=90, rotation=0, material=None):
        self.position = position
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.angle = angle
        self.rotation = rotation
        self.material = material
        self.color = self.get_random_color()

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
    def __init__(self, vertices=None, material=None):
        self.vertices = vertices
        self.material = material
        self.color = self.get_random_color()

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


# ================================================ 3D structures

