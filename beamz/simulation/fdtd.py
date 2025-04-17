from typing import Dict, Optional
import numpy as np
from beamz.const import *
from beamz.simulation.meshing import RegularGrid
from beamz.design.sources import ModeSource, LineSource
from beamz.design.structures import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MatplotlibRectangle, Circle as MatplotlibCircle, PathPatch
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation

class FDTD:
    """FDTD simulation class."""
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

    def update_h_fields(self):
        """Update magnetic field components with conductivity (including PML)."""
        # For PML to work properly, we need magnetic conductivity
        # We'll derive it from electric conductivity with impedance matching
        
        # Calculate magnetic conductivity (sigma_m) from electric conductivity (sigma)
        # For impedance matching: sigma_m = sigma * (mu_0/epsilon_0)
        sigma_m_x = self.sigma[:, :-1] * MU_0 / EPS_0
        sigma_m_y = self.sigma[:-1, :] * MU_0 / EPS_0
        
        # Calculate curl of E for H-field updates
        curl_e_x = (self.Ez[:, 1:] - self.Ez[:, :-1]) / self.dy
        curl_e_y = (self.Ez[1:, :] - self.Ez[:-1, :]) / self.dx
        
        # Update Hx with semi-implicit scheme for magnetic conductivity
        denom_x = 1.0 + sigma_m_x * self.dt / (2.0 * MU_0)
        factor_x = (1.0 - sigma_m_x * self.dt / (2.0 * MU_0)) / denom_x
        source_x = (self.dt / MU_0) / denom_x
        self.Hx = factor_x * self.Hx - source_x * curl_e_x
        
        # Update Hy with semi-implicit scheme for magnetic conductivity
        denom_y = 1.0 + sigma_m_y * self.dt / (2.0 * MU_0)
        factor_y = (1.0 - sigma_m_y * self.dt / (2.0 * MU_0)) / denom_y
        source_y = (self.dt / MU_0) / denom_y
        self.Hy = factor_y * self.Hy + source_y * curl_e_y
    
    def update_e_field(self):
        """Update electric field component with conductivity (including PML)."""
        # Calculate curl of H
        curl_h_x = np.zeros_like(self.Ez)
        curl_h_y = np.zeros_like(self.Ez)
        # Interior points calculation
        curl_h_x[1:-1, 1:-1] = (self.Hx[1:-1, 1:] - self.Hx[1:-1, :-1]) / self.dy
        curl_h_y[1:-1, 1:-1] = (self.Hy[1:, 1:-1] - self.Hy[:-1, 1:-1]) / self.dx
        
        # For better numerical stability, use semi-implicit scheme for conductivity
        # First calculate the denominator
        denom = 1.0 + self.sigma[1:-1, 1:-1] * self.dt / (2.0 * EPS_0 * self.epsilon_r[1:-1, 1:-1])
        # Then the numerator factors
        factor1 = (1.0 - self.sigma[1:-1, 1:-1] * self.dt / (2.0 * EPS_0 * self.epsilon_r[1:-1, 1:-1])) / denom
        factor2 = (self.dt / (EPS_0 * self.epsilon_r[1:-1, 1:-1])) / denom
        
        # Update Ez field with FDTD, conductivity term handles PML regions
        self.Ez[1:-1, 1:-1] = factor1 * self.Ez[1:-1, 1:-1] + factor2 * (-curl_h_x[1:-1, 1:-1] + curl_h_y[1:-1, 1:-1])

    def simulate_step(self):
        """Perform one FDTD step with stability checks."""
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
        for structure in self.design.structures:
            print(f"Processing structure: {type(structure).__name__}")
            if isinstance(structure, Rectangle):
                if structure.is_pml:
                    rect = MatplotlibRectangle(
                        (structure.position[0], structure.position[1]),
                        structure.width, structure.height,
                        facecolor='none', edgecolor='black', alpha=1, linestyle=':')
                else:
                    rect = MatplotlibRectangle(
                        (structure.position[0], structure.position[1]),
                        structure.width, structure.height,
                        facecolor=structure.color, edgecolor=self.border_color, alpha=1)
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
            elif isinstance(structure, LineSource):
                plt.plot((structure.start[0], structure.end[0]), (structure.start[1], structure.end[1]), '-', lw=4, color="crimson", alpha=0.5, label='Line Source')
                plt.plot((structure.start[0], structure.end[0]), (structure.start[1], structure.end[1]), '--', lw=2, color="black", alpha=0.5)
            elif isinstance(structure, ModeMonitor):
                plt.plot((structure.start[0], structure.end[0]), (structure.start[1], structure.end[1]), '-', lw=4, color="navy", alpha=0.5, label='Mode Monitor')

        plt.tight_layout()
        plt.show()

    def animate_live(self, field: str = "Ez", axis_scale=[-1,1]):
        """Animate the field in real time using matplotlib animation."""
        # If animation already exists, just update the data
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            current_field = getattr(self, field)
            self.im.set_array(current_field)
            self.ax.set_title(f't = {self.t:.2e} s')
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            return

        # Get current field data
        current_field = getattr(self, field)
        
        # Calculate figure size based on grid dimensions
        grid_height, grid_width = current_field.shape
        aspect_ratio = grid_width / grid_height
        base_size = 5  # Base size for the smaller dimension
        if aspect_ratio > 1: figsize = (base_size * aspect_ratio * 1.2, base_size)
        else: figsize = (base_size * 1.2, base_size / aspect_ratio)
            
        self.fig, self.ax = plt.subplots(figsize=figsize)
        
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
                if structure.is_pml:
                    rect = MatplotlibRectangle(
                        (structure.position[0], structure.position[1]),
                        structure.width, structure.height,
                        facecolor='none', edgecolor='black', alpha=1.0, linestyle=':')
                else:
                    rect = MatplotlibRectangle(
                        (structure.position[0], structure.position[1]),
                        structure.width, structure.height,
                        facecolor="none", edgecolor="black", alpha=1, linestyle="--")
                self.ax.add_patch(rect)
            elif isinstance(structure, Circle):
                circle = MatplotlibCircle(
                    (structure.position[0], structure.position[1]),
                    structure.radius,
                    facecolor='none', edgecolor='black', alpha=1, linestyle='--')
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
                ring_patch = PathPatch(path, facecolor='none', edgecolor='black', alpha=1, linestyle='--')
                self.ax.add_patch(ring_patch)
            elif isinstance(structure, ModeSource):
                self.ax.plot((structure.start[0], structure.end[0]), 
                           (structure.start[1], structure.end[1]), 
                           '-', lw=4, color="crimson", alpha=1)
                self.ax.plot((structure.start[0], structure.end[0]), 
                           (structure.start[1], structure.end[1]), 
                           '--', lw=2, color="black", alpha=1)

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

    def run(self, steps: Optional[int] = None, save=True, live=True, axis_scale=[-1,1], save_animation=False, 
            animation_filename='fdtd_animation.mp4', clean_visualization=True) -> Dict:
        """Run the simulation.
        
        Args:
            steps: Number of steps to run. If None, run until the end of the time array.
            save: Whether to save field data at each step.
            live: Whether to show live animation of the simulation.
            axis_scale: Color scale limits for the field visualization.
            save_animation: Whether to save an animation of the simulation as an mp4 file.
            animation_filename: Filename for the saved animation (must end in .mp4).
        
        Returns:
            Dictionary containing the simulation results.
        """
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
                elif isinstance(source, LineSource):
                    # Get the time modulation for this step
                    modulation = source.signal[step]
                    # Calculate the line's angle and direction
                    dx = source.end[0] - source.start[0]
                    dy = source.end[1] - source.start[1]
                    angle = np.arctan2(dy, dx)
                    
                    # Apply the source to the grid
                    if source.distribution is None:
                        # Calculate grid indices for start and end points
                        x_start = int(round(source.start[0] / self.dx))
                        y_start = int(round(source.start[1] / self.dy))
                        x_end = int(round(source.end[0] / self.dx))
                        y_end = int(round(source.end[1] / self.dy))

                        # Use Bresenham's line algorithm to get all grid points along the line
                        dx = abs(x_end - x_start)
                        dy = abs(y_end - y_start)
                        x, y = x_start, y_start
                        n = 1 + dx + dy
                        x_inc = 1 if x_end > x_start else -1
                        y_inc = 1 if y_end > y_start else -1
                        error = dx - dy
                        dx *= 2
                        dy *= 2
                        
                        while n > 0:
                            # Only apply if within grid bounds
                            if 0 <= x < self.nx and 0 <= y < self.ny:
                                self.Ez[y, x] += modulation
                                # Apply direction to the source
                                if source.direction == "+x":
                                    if angle > -np.pi/4 and angle < np.pi/4:
                                        self.Ez[y, x-2] = 0
                                elif source.direction == "-x":
                                    if angle > 3*np.pi/4 or angle < -3*np.pi/4:
                                        self.Ez[y, x+2] = 0
                                elif source.direction == "+y":
                                    if angle > np.pi/4 and angle < 3*np.pi/4:
                                        self.Ez[y-2, x] = 0
                                elif source.direction == "-y":
                                    if angle > -3*np.pi/4 and angle < -np.pi/4:
                                        self.Ez[y+2, x] = 0
                            if error > 0:
                                x += x_inc
                                error -= dy
                            else:
                                y += y_inc
                                error += dx
                            n -= 1
                    else:
                        # Apply the source to all points in the distribution
                        for point in source.distribution:
                            amplitude, x_raw, y_raw = point
                            # Convert the position to the nearest grid point
                            x = int(round(x_raw / self.dx))
                            y = int(round(y_raw / self.dy))
                            # Only apply if within grid bounds
                            if 0 <= x < self.nx and 0 <= y < self.ny:
                                self.Ez[y, x] += amplitude * modulation
                                # Apply direction to the source
                                if source.direction == "+x":
                                    if angle > -np.pi/4 and angle < np.pi/4:
                                        self.Ez[y, x-1] = 0
                                elif source.direction == "-x":
                                    if angle > 3*np.pi/4 or angle < -3*np.pi/4:
                                        self.Ez[y, x+1] = 0
                                elif source.direction == "+y":
                                    if angle > np.pi/4 and angle < 3*np.pi/4:
                                        self.Ez[y-1, x] = 0
                                elif source.direction == "-y":
                                    if angle > -3*np.pi/4 and angle < -np.pi/4:
                                        self.Ez[y+1, x] = 0
            
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
                self.animate_live(field="Ez", axis_scale=axis_scale)

        # Clean up animation
        if live and self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.im = None
        # Save animation if requested
        if save_animation and save:
            self.save_animation(field="Ez", axis_scale=axis_scale, filename=animation_filename, clean_visualization=clean_visualization)
        return self.results
        
    def save_animation(self, field: str = "Ez", axis_scale=[-1, 1], filename='fdtd_animation.mp4', fps=60, frame_skip=4, clean_visualization=False):
        """Save an animation of the simulation results as an mp4 file."""
        if len(self.results[field]) == 0:
            print("No field data to animate. Make sure to run the simulation with save=True.")
            return
            
        # Create list of frame indices to use (applying frame_skip)
        total_frames = len(self.results[field])
        frame_indices = range(0, total_frames, frame_skip)
        
        # Calculate figure size based on grid dimensions
        grid_height, grid_width = self.results[field][0].shape
        aspect_ratio = grid_width / grid_height
        base_size = 5  # Base size for the smaller dimension
        if aspect_ratio > 1: figsize = (base_size * aspect_ratio * 1.2, base_size)
        else: figsize = (base_size * 1.2, base_size / aspect_ratio)
        
        # Set up figure and axes based on visualization style
        if clean_visualization:
            # Create figure with absolutely no padding or borders
            # Use tight_layout and adjust figure parameters to eliminate all whitespace
            if aspect_ratio > 1: figsize = (base_size * aspect_ratio, base_size)
            else: figsize = (base_size, base_size / aspect_ratio)
            fig = plt.figure(figsize=figsize, frameon=False)
            # Use the entire figure space with no margins
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()
            # Ensure no padding or margins
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            # Explicitly turn off all spines, ticks, and labels
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        else:
            # Create standard figure with normal padding
            fig, ax = plt.subplots(figsize=figsize)
            # Get units for axis labels
            max_dim = max(self.design.width, self.design.height)
            if max_dim >= 1e-3: scale, unit = 1e3, 'mm'
            elif max_dim >= 1e-6: scale, unit = 1e6, 'µm'
            elif max_dim >= 1e-9: scale, unit = 1e9, 'nm'
            else: scale, unit = 1e12, 'pm'
        
        # Create initial plot
        im = ax.imshow(self.results[field][0], origin='lower',
                      extent=(0, self.design.width, 0, self.design.height),
                      cmap='RdBu', aspect='equal', interpolation='bicubic', 
                      vmin=axis_scale[0], vmax=axis_scale[1])
        
        # Add colorbar if not using clean visualization
        if not clean_visualization:
            colorbar = plt.colorbar(im, orientation='vertical', aspect=30, extend='both')
            colorbar.set_label(f'{field} Field Amplitude')
        
        # Add design structure outlines
        for structure in self.design.structures:
            if isinstance(structure, Rectangle):
                if structure.is_pml:
                    rect = MatplotlibRectangle(
                        (structure.position[0], structure.position[1]),
                        structure.width, structure.height,
                        facecolor='none', edgecolor='black', alpha=1.0, linestyle=':')
                else:
                    rect = MatplotlibRectangle(
                        (structure.position[0], structure.position[1]),
                        structure.width, structure.height,
                        facecolor="none", edgecolor="black", alpha=1)
                ax.add_patch(rect)
            elif isinstance(structure, Circle):
                circle = MatplotlibCircle(
                    (structure.position[0], structure.position[1]),
                    structure.radius,
                    facecolor='none', edgecolor='black', alpha=1, linestyle='--')
                ax.add_patch(circle)
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
                ring_patch = PathPatch(path, facecolor='none', edgecolor='black', alpha=1, linestyle='--')
                ax.add_patch(ring_patch)
            elif isinstance(structure, ModeSource) or isinstance(structure, LineSource):
                ax.plot((structure.start[0], structure.end[0]), 
                      (structure.start[1], structure.end[1]), 
                      '-', lw=4, color="crimson", alpha=0.5)
                ax.plot((structure.start[0], structure.end[0]), 
                      (structure.start[1], structure.end[1]), 
                      '--', lw=2, color="black", alpha=1)
        
        # Configure standard plot elements if not using clean visualization
        if not clean_visualization:
            # Add axis labels with proper scaling
            plt.xlabel(f'X ({unit})')
            plt.ylabel(f'Y ({unit})')
            ax.xaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
            ax.yaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
            
            # Title with time information
            title = ax.set_title(f't = {self.results["t"][0]:.2e} s')
        else:
            title = None
        
        # Animation update function
        def update(frame_idx):
            frame = frame_indices[frame_idx]
            im.set_array(self.results[field][frame])
            if not clean_visualization:
                title.set_text(f't = {self.results["t"][frame]:.2e} s')
                return [im, title]
            return [im]
        
        # Create animation with only the selected frames
        frames = len(frame_indices)
        ani = FuncAnimation(fig, update, frames=frames, blit=True)
        
        # Save animation
        try:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps)
            if clean_visualization:
                # Use only supported parameters
                ani.save(filename, writer=writer, dpi=300)
            else:
                ani.save(filename, writer=writer, dpi=100)
            print(f"Animation saved to {filename} (using {frames} of {total_frames} frames)")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Make sure FFmpeg is installed on your system.")
        
        # Close the figure
        plt.close(fig)