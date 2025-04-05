import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MatplotlibRectangle, PathPatch, Circle as MatplotlibCircle
from matplotlib.path import Path
import random
import numpy as np

class Design:
    def __init__(self, width=1, height=1, depth=None):
        self.structures = []
        self.is_3d = False
        self.width = width
        self.height = height
        self.depth = depth
        
    def add(self, structure):
        self.structures.append(structure)
        # Check if the new structure is 3D
        if hasattr(structure, 'z') or hasattr(structure, 'depth'):
            self.is_3d = True
            
    def show(self):
        if not self.structures:
            print("No structures to display")
            return
            
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
                        facecolor=structure.color, alpha=0.5)
                    ax_xy.add_patch(rect)
                    # XZ view
                    rect = MatplotlibRectangle(
                        (structure.position[0], structure.position[2]),
                        structure.width, structure.depth,
                        facecolor=structure.color, alpha=0.5)
                    ax_xz.add_patch(rect)
                    # YZ view
                    rect = MatplotlibRectangle(
                        (structure.position[1], structure.position[2]),
                        structure.height, structure.depth,
                        facecolor=structure.color, alpha=0.5)
                    ax_yz.add_patch(rect)
            
            # Set labels and titles
            ax_xy.set_title('XY View')
            ax_xy.set_xlabel('X')
            ax_xy.set_ylabel('Y')
            ax_xz.set_title('XZ View')
            ax_xz.set_xlabel('X')
            ax_xz.set_ylabel('Z')
            ax_yz.set_title('YZ View')
            ax_yz.set_xlabel('Y')
            ax_yz.set_ylabel('Z')
            
        else:
            print("Showing 2D design...")
            # Create single plot for 2D visualization
            fig, ax = plt.subplots(figsize=(8, 8))
            # Set axis limits based on predefined dimensions
            ax.set_xlim(0, self.width)
            ax.set_ylim(0, self.height)
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Plot each structure
            for structure in self.structures:
                if isinstance(structure, Rectangle):
                    rect = MatplotlibRectangle(
                        (structure.position[0], structure.position[1]),
                        structure.width, structure.height,
                        facecolor=structure.color, alpha=1)
                    ax.add_patch(rect)
                elif isinstance(structure, Circle):
                    circle = plt.Circle(
                        (structure.position[0], structure.position[1]),
                        structure.radius,
                        facecolor=structure.color, alpha=1)
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
                    ring_patch = PathPatch(path, facecolor=structure.color, alpha=1, edgecolor='none')
                    ax.add_patch(ring_patch)
            
            ax.set_title('2D Design')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal')
        
        # Set equal aspect ratio and adjust layout
        plt.tight_layout()
        plt.show()
        
    def __str__(self):
        return f"Design with {len(self.structures)} structures ({'3D' if self.is_3d else '2D'})"




# ================================================ 2D structures
# rectangle
class Rectangle:
    def __init__(self, position=(0,0), width=1, height=1, material=None):
        self.position = position
        self.width = width
        self.height = height
        self.material = material
        self.color = self.get_random_color()

    def get_random_color(self):
        return '#{:06x}'.format(random.randint(0, 0xFFFFFF))

# taper

class Circle:
    def __init__(self, position=(0,0), radius=1, material=None):
        self.position = position
        self.radius = radius
        self.material = material
        self.color = self.get_random_color()
    
    def get_random_color(self):
        return '#{:06x}'.format(random.randint(0, 0xFFFFFF))

class Ring:
    def __init__(self, position=(0,0), inner_radius=1, outer_radius=2, material=None):
        self.position = position
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.material = material
        self.color = self.get_random_color()

    def get_random_color(self):
        return '#{:06x}'.format(random.randint(0, 0xFFFFFF))

# bend



# ================================================ 3D structures

