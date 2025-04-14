from typing import Dict, Optional
import numpy as np
from beamz.const import *
from beamz.simulation.meshing import RegularGrid
from beamz.design.sources import ModeSource
from beamz.design.structures import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MatplotlibRectangle, Circle as MatplotlibCircle, PathPatch
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation

class FDTD:
    """
    FDTD simulation class.

    Args:
        design: Design object containing the structures, sources, and monitors to simulate and measure
        grid: RegularGrid object used to discretize the design
        device: str, "cpu" (using numpy backend) or "gpu" (using jax backend)
    """
    def __init__(self, design, time, mesh: str = "regular", resolution: float = 0.02*µm):
        # Initialize the design and mesh
        self.design = design
        self.resolution = resolution
        self.mesh = RegularGrid(design=self.design, resolution=self.resolution) if mesh == "regular" else None
        self.dx = self.mesh.dx
        self.dy = self.mesh.dy
        self.epsilon_r = self.mesh.permittivity
        self.mu_r = self.mesh.permeability
        self.sigma = self.mesh.conductivity
        # Initialize the fields
        self.nx, self.ny = self.mesh.shape
        self.Ez = np.zeros((self.nx, self.ny))
        self.Hx = np.zeros((self.nx, self.ny-1))
        self.Hy = np.zeros((self.nx-1, self.ny))
        # Initialize the time
        self.time = time
        self.dt = self.time[1] - self.time[0]
        self.num_steps = len(self.time)
        # Initialize the sources
        self.sources = self.design.sources
        # Initialize the results
        self.results = {"Ez": [], "Hx": [], "Hy": [], "t": []}
        # Initialize animation attributes
        self.fig = None
        self.ax = None
        self.anim = None
        self.im = None
        print(f"Initialized FDTD with grid size: {self.nx}x{self.ny}")

    def update_h_fields(self):
        """Update magnetic field components with PML"""
        self.Hx[:, :] = self.Hx[:, :] - (self.dt/(MU_0*self.dy)) * \
                        (self.Ez[:, 1:] - self.Ez[:, :-1])
        self.Hy[:, :] = self.Hy[:, :] + (self.dt/(MU_0*self.dx)) * \
                        (self.Ez[1:, :] - self.Ez[:-1, :])
    
    def update_e_field(self):
        """Update electric field component with PML"""
        # First update the main field without PML
        self.Ez[1:-1, 1:-1] = self.Ez[1:-1, 1:-1] + \
            (self.dt/(EPS_0*self.epsilon_r[1:-1, 1:-1])) * \
            ((self.Hy[1:, 1:-1] - self.Hy[:-1, 1:-1])/self.dx - \
             (self.Hx[1:-1, 1:] - self.Hx[1:-1, :-1])/self.dy)
        # Then apply PML only at the boundaries where sigma > 0
        mask = self.sigma[1:-1, 1:-1] > 0
        if np.any(mask):
            sigma_x = self.sigma[1:-1, 1:-1][mask]
            sigma_y = self.sigma[1:-1, 1:-1][mask]
            # Update coefficients for PML regions only
            cx = np.exp(-sigma_x * self.dt / EPS_0)
            cy = np.exp(-sigma_y * self.dt / EPS_0)
            # Apply PML absorption only at boundaries
            self.Ez[1:-1, 1:-1][mask] *= (cx + cy) / 2

    def simulate_step(self):
        """Perform one FDTD step"""
        self.update_h_fields()
        self.update_e_field()

    def plot_field(self, field: str = "Ez", t: float = None) -> None:
        """Plot a field at a given time with proper scaling and units."""
        # Handle the case where we're plotting current state (not from results)
        if len(self.results['t']) == 0:
            current_field = getattr(self, field)  # Get current field state
            current_t = self.t
            print(f"Plotting current field state at t = {current_t:.2e} s")
        else:
            if t is None: t = self.results['t'][-1]
            # Find closest time step
            t_idx = np.argmin(np.abs(np.array(self.results['t']) - t))
            current_field = self.results[field][t_idx]
            current_t = self.results['t'][t_idx]
            print(f"Plotting saved field at t = {current_t:.2e} s (index {t_idx})")
        
        #print(f"Field range: min = {np.min(current_field):.2e}, max = {np.max(current_field):.2e}")
        
        # Determine appropriate SI unit and scale for spatial dimensions
        max_dim = max(self.design.width, self.design.height)
        if max_dim >= 1e-3: scale, unit = 1e3, 'mm'
        elif max_dim >= 1e-6: scale, unit = 1e6, 'µm'
        elif max_dim >= 1e-9: scale, unit = 1e9, 'nm'
        else: scale, unit = 1e12, 'pm'
        # Calculate figure size based on grid dimensions
        grid_height, grid_width = current_field.shape
        aspect_ratio = grid_width / grid_height
        base_size = 6  # Base size for the smaller dimension
        if aspect_ratio > 1: figsize = (base_size * aspect_ratio, base_size)
        else: figsize = (base_size, base_size / aspect_ratio)
        # Create the figure
        plt.figure(figsize=figsize)
        plt.imshow(current_field, origin='lower', 
                  extent=(0, self.design.width, 0, self.design.height),
                  cmap='RdBu', aspect='equal', interpolation='bicubic')
        plt.colorbar(label=f'{field} Field Amplitude')
        plt.title(f'{field} Field at t = {current_t:.2e} s')
        plt.xlabel(f'X ({unit})')
        plt.ylabel(f'Y ({unit})')
        # Update tick labels with scaled values
        plt.gca().xaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
        plt.gca().yaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')

        # Create an overlay of the design outlines
        print(f"Design structures: {self.design.structures}")
        for structure in self.design.structures:
            print(f"Processing structure: {type(structure).__name__}")
            if isinstance(structure, Rectangle):
                print(f"Adding Rectangle at {structure.position} with width={structure.width}, height={structure.height}")
                rect = MatplotlibRectangle(
                    (structure.position[0], structure.position[1]),
                    structure.width, structure.height,
                    facecolor='none', edgecolor='black', alpha=0.5, linestyle='--')
                plt.gca().add_patch(rect)
            elif isinstance(structure, Circle):
                print(f"Adding Circle at {structure.position} with radius={structure.radius}")
                circle = MatplotlibCircle(
                    (structure.position[0], structure.position[1]),
                    structure.radius,
                    facecolor='none', edgecolor='black', alpha=0.5, linestyle='--')
                plt.gca().add_patch(circle)
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
                ring_patch = PathPatch(path, facecolor='none', edgecolor='black', alpha=0.5, linestyle='--')
                plt.gca().add_patch(ring_patch)
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
                    [x_outer[0], y_outer[0]],
                    *np.column_stack([x_outer[1:], y_outer[1:]]),
                    [x_inner[-1], y_inner[-1]],
                    *np.column_stack([x_inner[-2::-1], y_inner[-2::-1]]),
                    [x_outer[0], y_outer[0]]
                ])
                # Define path codes for a single continuous path
                codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 2) + [Path.CLOSEPOLY]
                # Create the path and patch
                path = Path(vertices, codes)
                bend_patch = PathPatch(path, facecolor='none', edgecolor='black', alpha=0.5, linestyle='--')
                plt.gca().add_patch(bend_patch)
            elif isinstance(structure, Polygon):
                polygon = plt.Polygon(structure.vertices, facecolor='none', edgecolor='black', alpha=0.5, linestyle='--')
                plt.gca().add_patch(polygon)
            elif isinstance(structure, ModeSource):
                plt.plot((structure.start[0], structure.end[0]), (structure.start[1], structure.end[1]), '-', lw=4, color="crimson", alpha=0.5, label='Mode Source')
                plt.plot((structure.start[0], structure.end[0]), (structure.start[1], structure.end[1]), '--', lw=2, color="black", alpha=0.5)
            elif isinstance(structure, ModeMonitor):
                plt.plot((structure.start[0], structure.end[0]), (structure.start[1], structure.end[1]), '-', lw=4, color="navy", alpha=0.5, label='Mode Monitor')

        plt.tight_layout()
        plt.show()

    def animate_live(self, field: str = "Ez", t: float = None, axis_scale=[-1,1]):
        """Animate the field in real time using matplotlib animation.
        
        Args:
            field (str): Field component to animate ('Ez', 'Hx', or 'Hy')
            t (float): Current simulation time
        """
        # If animation already exists, just update the data
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            current_field = getattr(self, field)
            self.im.set_array(current_field)
            self.ax.set_title(f't = {self.t:.2e} s')
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            return

        # Create new figure and axis if none exists
        aspect_ratio = self.ny / self.nx
        base_size = 6
        if aspect_ratio > 1:
            figsize = (base_size, base_size * aspect_ratio)
        else:
            figsize = (base_size / aspect_ratio, base_size)
            
        self.fig, self.ax = plt.subplots(figsize=figsize)

        # Get current field data
        current_field = getattr(self, field)
        
        # Create initial plot with proper scaling
        self.im = self.ax.imshow(current_field, origin='lower',
                                extent=(0, self.design.width, 0, self.design.height),
                                cmap='RdBu', aspect='equal', interpolation='bicubic', vmin=axis_scale[0], vmax=axis_scale[1])
        
        # Add colorbar
        colorbar = plt.colorbar(self.im, orientation='vertical', aspect=30, extend='both')
        colorbar.set_label(f'{field} Field Amplitude')
        
        # Add design structure outlines
        for structure in self.design.structures:
            if isinstance(structure, Rectangle):
                rect = MatplotlibRectangle(
                    (structure.position[0], structure.position[1]),
                    structure.width, structure.height,
                    facecolor='none', edgecolor='black', alpha=0.5, linestyle='--')
                self.ax.add_patch(rect)
            elif isinstance(structure, Circle):
                circle = MatplotlibCircle(
                    (structure.position[0], structure.position[1]),
                    structure.radius,
                    facecolor='none', edgecolor='black', alpha=0.5, linestyle='--')
                self.ax.add_patch(circle)
            elif isinstance(structure, Ring):
                # Create points for the ring
                N = 100
                theta = np.linspace(0, 2 * np.pi, N, endpoint=True)
                x_outer = structure.position[0] + structure.outer_radius * np.cos(theta)
                y_outer = structure.position[1] + structure.outer_radius * np.sin(theta)
                x_inner = structure.position[0] + structure.inner_radius * np.cos(theta[::-1])
                y_inner = structure.position[1] + structure.inner_radius * np.sin(theta[::-1])
                vertices = np.vstack([np.column_stack([x_outer, y_outer]),
                                   np.column_stack([x_inner, y_inner])])
                codes = np.concatenate([[Path.MOVETO] + [Path.LINETO] * (N - 1),
                                     [Path.MOVETO] + [Path.LINETO] * (N - 1)])
                path = Path(vertices, codes)
                ring_patch = PathPatch(path, facecolor='none', edgecolor='black', alpha=0.5, linestyle='--')
                self.ax.add_patch(ring_patch)
            elif isinstance(structure, ModeSource):
                self.ax.plot((structure.start[0], structure.end[0]), 
                           (structure.start[1], structure.end[1]), 
                           '-', lw=4, color="crimson", alpha=0.5)
                self.ax.plot((structure.start[0], structure.end[0]), 
                           (structure.start[1], structure.end[1]), 
                           '--', lw=2, color="black", alpha=0.5)

        # Set axis labels with proper scaling
        max_dim = max(self.design.width, self.design.height)
        if max_dim >= 1e-3: scale, unit = 1e3, 'mm'
        elif max_dim >= 1e-6: scale, unit = 1e6, 'µm'
        elif max_dim >= 1e-9: scale, unit = 1e9, 'nm'
        else: scale, unit = 1e12, 'pm'
        
        # Add axis labels with proper scaling
        plt.xlabel(f'X ({unit})')
        plt.ylabel(f'Y ({unit})')
        self.ax.xaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
        self.ax.yaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)  # Small pause to ensure window is shown

    def run(self, steps: Optional[int] = None, save=True, live=True, axis_scale=[-1,1]) -> Dict:
        """Run the simulation."""
        # Initialize simulation state
        self.t = 0
        self._total_steps = self.num_steps
        self._save_results = save

        # Calculate Courant number to check stability
        c = 1/np.sqrt(EPS_0 * MU_0)  # Speed of light
        courant = c * self.dt / min(self.dx, self.dy)
        if courant > 1/np.sqrt(2):
            print(f"Warning: Simulation may be unstable! Courant number = {courant:.3f} > {1/np.sqrt(2):.3f}")
            print(f"Consider reducing dt or increasing dx/dy")

        # Run simulation
        for step in range(self.num_steps):
            # Update fields
            self.simulate_step()
            
            # Apply sources
            for source in self.sources:
                if isinstance(source, ModeSource):
                    # Get the mode profile for the first mode
                    mode_profile = source.mode_profiles[0]
                    # Get the time modulation for this step
                    modulation = source.signal[step]
                    # Apply the mode profile to all points
                    for point in mode_profile:
                        amplitude, x_raw, y_raw = point
                        # Convert the position to the nearest grid point using correct resolutions
                        x = int(round(x_raw / self.dx))
                        y = int(round(y_raw / self.dy))
                        # Skip points outside the grid
                        if x < 0 or x >= self.nx or y < 0 or y >= self.ny:
                            continue
                        # Add the source contribution to the field (don't overwrite!)
                        self.Ez[y, x] += amplitude * modulation 
                        # Apply direction to the source
                        if source.direction == "+x": self.Ez[y, x-1] = 0
                        elif source.direction == "-x": self.Ez[y, x+1] = 0
                        elif source.direction == "+y": self.Ez[y-1, x] = 0
                        elif source.direction == "-y": self.Ez[y+1, x] = 0
                else:
                    # Handle other source types here if needed
                    pass
                    
            # Save results if requested
            if save:
                self.results['Ez'].append(self.Ez.copy())
                self.results['Hx'].append(self.Hx.copy())
                self.results['Hy'].append(self.Hy.copy())
                self.results['t'].append(self.t)
            
            # Update time
            self.t += self.dt
            
            # Show progress
            if live:  # Update every 5 steps for smoother animation
                self.animate_live(field="Ez", t=self.t, axis_scale=axis_scale)

        # Clean up animation
        if live and self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.im = None

        #if save:
        #    self.save_data()
        #    self.save_figures()
        #    self.save_video()

        return self.results