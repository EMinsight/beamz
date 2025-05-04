from typing import Dict, Optional
import numpy as np
from beamz.const import *
from beamz.simulation.meshing import RegularGrid
from beamz.design.sources import ModeSource, GaussianSource
from beamz.design.structures import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MatplotlibRectangle, Circle as MatplotlibCircle, PathPatch
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation
from beamz.simulation.backends import get_backend

class FDTD:
    """FDTD simulation class."""
    def __init__(self, design, time, mesh: str = "regular", resolution: float = 0.02*µm, backend="numpy", backend_options=None):
        # Initialize the design and mesh
        self.design = design
        self.resolution = resolution
        self.mesh = RegularGrid(design=self.design, resolution=self.resolution) if mesh == "regular" else None
        self.dx = self.mesh.dx
        self.dy = self.mesh.dy
        self.epsilon_r = self.mesh.permittivity
        self.mu_r = self.mesh.permeability
        self.sigma = self.mesh.conductivity
        
        # Initialize the backend
        backend_options = backend_options or {}
        self.backend = get_backend(name=backend, **backend_options)
        
        # Initialize the fields with the backend
        self.nx, self.ny = self.mesh.shape
        self.Ez = self.backend.zeros((self.nx, self.ny))
        self.Hx = self.backend.zeros((self.nx, self.ny-1))
        self.Hy = self.backend.zeros((self.nx-1, self.ny))
        
        # Convert material properties to backend arrays
        self.epsilon_r = self.backend.from_numpy(self.epsilon_r)
        self.mu_r = self.backend.from_numpy(self.mu_r)
        self.sigma = self.backend.from_numpy(self.sigma)
        
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
        self.Hx, self.Hy = self.backend.update_h_fields(
            self.Hx, self.Hy, self.Ez, self.sigma, 
            self.dx, self.dy, self.dt, MU_0, EPS_0
        )
    
    def update_e_field(self):
        """Update electric field component with conductivity (including PML)."""
        self.Ez = self.backend.update_e_field(
            self.Ez, self.Hx, self.Hy, self.sigma, self.epsilon_r,
            self.dx, self.dy, self.dt, EPS_0
        )

    def simulate_step(self):
        """Perform one FDTD step with stability checks."""
        self.update_h_fields()
        self.update_e_field()

    # TODO: Implement this
    def show(self, field: str = "Ez", t=None, cmap="RdBu", axis_scale=[-1,1]):
        """Show a given field or power at a given time or integrated over time."""
        pass

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
                N = 100
                # Only draw the circular bend within the specified angle
                angle_rad = np.radians(structure.angle)
                rotation_rad = np.radians(structure.rotation)
                
                # Calculate start and end angles based on rotation
                start_angle = rotation_rad
                end_angle = rotation_rad + angle_rad
                
                # Create theta values only for the specified angle range
                theta = np.linspace(start_angle, end_angle, N, endpoint=True)
                
                # Generate outer and inner curve points
                x_outer = structure.position[0] + structure.outer_radius * np.cos(theta)
                y_outer = structure.position[1] + structure.outer_radius * np.sin(theta)
                x_inner = structure.position[0] + structure.inner_radius * np.cos(theta)
                y_inner = structure.position[1] + structure.inner_radius * np.sin(theta)
                
                # Create vertex list for the bend shape
                vertices = np.vstack([
                    # Start at the first outer point
                    [x_outer[0], y_outer[0]],
                    # Add all the outer curve points
                    *np.column_stack([x_outer[1:], y_outer[1:]]),
                    # Add the last inner point
                    [x_inner[-1], y_inner[-1]],
                    # Add all inner points in reverse
                    *np.column_stack([x_inner[-2::-1], y_inner[-2::-1]]),
                    # Close the polygon
                    [x_outer[0], y_outer[0]]
                ])
                
                codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 2) + [Path.CLOSEPOLY]
                path = Path(vertices, codes)
                bend_patch = PathPatch(path, facecolor='none', edgecolor='black', alpha=1, linestyle='--')
                plt.gca().add_patch(bend_patch)
            elif isinstance(structure, Polygon):
                polygon = plt.Polygon(structure.vertices, facecolor='none', edgecolor='black', alpha=0.5, linestyle='--')
                plt.gca().add_patch(polygon)
            elif isinstance(structure, ModeSource):
                plt.plot((structure.start[0], structure.end[0]), (structure.start[1], structure.end[1]), '-', lw=4, color="crimson", alpha=0.5, label='Mode Source')
                plt.plot((structure.start[0], structure.end[0]), (structure.start[1], structure.end[1]), '--', lw=2, color="black", alpha=0.5)
            elif isinstance(structure, GaussianSource):
                source_circle = MatplotlibCircle(
                    (structure.position[0], structure.position[1]),
                    structure.width,  # Use width as radius
                    facecolor='none', edgecolor='orange', alpha=0.8, linestyle='dotted', label='Gaussian Source')
                plt.gca().add_patch(source_circle)
            #elif isinstance(structure, ModeMonitor):
            #    plt.plot((structure.start[0], structure.end[0]), (structure.start[1], structure.end[1]), '-', lw=4, color="navy", alpha=0.5, label='Mode Monitor')

        plt.tight_layout()
        plt.show()

    def animate_live(self, field_data=None, field="Ez", axis_scale=[-1,1]):
        """Animate the field in real time using matplotlib animation."""
        if field_data is None:
            field_data = self.backend.to_numpy(getattr(self, field))
        
        # If animation already exists, just update the data
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            current_field = field_data
            self.im.set_array(current_field)
            self.ax.set_title(f't = {self.t:.2e} s')
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            return

        # Get current field data
        current_field = field_data
        
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
            elif isinstance(structure, CircularBend):
                # Create points for the bend
                N = 100
                # Only draw the circular bend within the specified angle
                angle_rad = np.radians(structure.angle)
                rotation_rad = np.radians(structure.rotation)
                
                # Calculate start and end angles based on rotation
                start_angle = rotation_rad
                end_angle = rotation_rad + angle_rad
                
                # Create theta values only for the specified angle range
                theta = np.linspace(start_angle, end_angle, N, endpoint=True)
                
                # Generate outer and inner curve points
                x_outer = structure.position[0] + structure.outer_radius * np.cos(theta)
                y_outer = structure.position[1] + structure.outer_radius * np.sin(theta)
                x_inner = structure.position[0] + structure.inner_radius * np.cos(theta)
                y_inner = structure.position[1] + structure.inner_radius * np.sin(theta)
                
                # Create vertex list for the bend shape
                vertices = np.vstack([
                    # Start at the first outer point
                    [x_outer[0], y_outer[0]],
                    # Add all the outer curve points
                    *np.column_stack([x_outer[1:], y_outer[1:]]),
                    # Add the last inner point
                    [x_inner[-1], y_inner[-1]],
                    # Add all inner points in reverse
                    *np.column_stack([x_inner[-2::-1], y_inner[-2::-1]]),
                    # Close the polygon
                    [x_outer[0], y_outer[0]]
                ])
                
                codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 2) + [Path.CLOSEPOLY]
                path = Path(vertices, codes)
                bend_patch = PathPatch(path, facecolor='none', edgecolor='black', alpha=1, linestyle='--')
                self.ax.add_patch(bend_patch)
                
            elif isinstance(structure, ModeSource):
                self.ax.plot((structure.start[0], structure.end[0]), 
                           (structure.start[1], structure.end[1]), 
                           '-', lw=4, color="crimson", alpha=1)
                self.ax.plot((structure.start[0], structure.end[0]), 
                           (structure.start[1], structure.end[1]), 
                           '--', lw=2, color="black", alpha=1)
            elif isinstance(structure, GaussianSource):
                source_circle = MatplotlibCircle(
                    (structure.position[0], structure.position[1]),
                    structure.width,
                    facecolor='none', edgecolor='black', alpha=1, linestyle='dotted')
                self.ax.add_patch(source_circle)

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

        # Determine optimal save frequency based on backend type
        is_gpu_backend = hasattr(self.backend, 'device') and self.backend.device.type in ['cuda', 'mps']
        
        # For GPU backends, avoid excessive CPU-GPU transfers by batching the result saves
        if is_gpu_backend and save:
            save_freq = max(10, self.num_steps // 100)  # Save approximately every 1% of steps or min of 10 steps
            print(f"GPU backend detected: Optimizing result storage (saving every {save_freq} steps)")
        else:
            save_freq = 1  # Save every step for CPU backends
            
        # For live visualization with GPU backends, don't update too frequently
        if is_gpu_backend and live:
            live_update_freq = max(5, self.num_steps // 50)  # Update visualization approximately every 2% of steps
            print(f"GPU backend detected: Optimizing visualization (updating every {live_update_freq} steps)")
        else:
            live_update_freq = 5  # Update visualization every 5 steps for CPU backends

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
                elif isinstance(source, GaussianSource):
                    modulation = source.signal[step]
                    center_x_phys, center_y_phys = source.position
                    width_phys = source.width  # Assuming width is standard deviation (sigma)
                    
                    # Convert physical units to grid coordinates
                    center_x_grid = center_x_phys / self.dx
                    center_y_grid = center_y_phys / self.dy
                    # sigma in grid units
                    width_x_grid = width_phys / self.dx 
                    width_y_grid = width_phys / self.dy 

                    # Define the grid range to apply the source (e.g., +/- 3 sigma)
                    # Use max(1, ...) to ensure at least one grid cell width if sigma_grid is very small
                    wx_grid_cells = max(1, int(round(3 * width_x_grid)))
                    wy_grid_cells = max(1, int(round(3 * width_y_grid)))
                    
                    # Calculate bounding box indices, clamped to grid boundaries
                    x_center_idx = int(round(center_x_grid))
                    y_center_idx = int(round(center_y_grid))
                    
                    x_start = max(0, x_center_idx - wx_grid_cells)
                    x_end = min(self.nx, x_center_idx + wx_grid_cells + 1)
                    y_start = max(0, y_center_idx - wy_grid_cells)
                    y_end = min(self.ny, y_center_idx + wy_grid_cells + 1)

                    # Create meshgrid for the affected area
                    y_indices = np.arange(y_start, y_end)
                    x_indices = np.arange(x_start, x_end)
                    y_grid, x_grid = np.meshgrid(y_indices, x_indices, indexing='ij')
                    
                    # Calculate squared distances from the center in grid units
                    dist_x_sq = (x_grid - center_x_grid)**2
                    dist_y_sq = (y_grid - center_y_grid)**2
                    
                    # Calculate Gaussian amplitude (handle zero width appropriately)
                    # Use small epsilon to avoid division by zero if width_grid is exactly 0
                    epsilon = 1e-9
                    sigma_x_sq = width_x_grid**2 + epsilon
                    sigma_y_sq = width_y_grid**2 + epsilon

                    exponent = -(dist_x_sq / (2 * sigma_x_sq) + dist_y_sq / (2 * sigma_y_sq))
                    gaussian_amp = np.exp(exponent)
                    
                    # Convert to backend array type if not already
                    gaussian_amp = self.backend.from_numpy(gaussian_amp)
                            
                    # Add the source contribution to the Ez field slice
                    self.Ez[y_start:y_end, x_start:x_end] += gaussian_amp * modulation
            
            # Save results if requested and at the right frequency
            if save and (step % save_freq == 0 or step == self.num_steps - 1):
                # Convert arrays to numpy for saving
                self.results['Ez'].append(self.backend.to_numpy(self.backend.copy(self.Ez)))
                self.results['Hx'].append(self.backend.to_numpy(self.backend.copy(self.Hx)))
                self.results['Hy'].append(self.backend.to_numpy(self.backend.copy(self.Hy)))
                self.results['t'].append(self.t)
                
            # Update time
            self.t += self.dt
            
            # Show progress
            if live and (step % live_update_freq == 0 or step == self.num_steps - 1):
                # Convert to numpy for visualization
                Ez_np = self.backend.to_numpy(self.Ez)
                self.animate_live(field_data=Ez_np, field="Ez", axis_scale=axis_scale)

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
            elif isinstance(structure, ModeSource):
                ax.plot((structure.start[0], structure.end[0]), 
                      (structure.start[1], structure.end[1]), 
                      '-', lw=4, color="crimson", alpha=0.5)
                ax.plot((structure.start[0], structure.end[0]), 
                      (structure.start[1], structure.end[1]), 
                      '--', lw=2, color="black", alpha=1)
            elif isinstance(structure, GaussianSource):
                source_circle = MatplotlibCircle(
                    (structure.position[0], structure.position[1]),
                    structure.width,
                    facecolor='none', edgecolor='white', alpha=1, linestyle='dotted')
                ax.add_patch(source_circle)
        
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
        
    def plot_power(self, cmap: str = "hot", log_scale: bool = False, vmin: float = None, vmax: float = None, db_colorbar: bool = False):
        """Plot the time-integrated power distribution from the E and H fields.
        
        Args:
            cmap: Colormap to use for the plot (default: 'hot')
            log_scale: Whether to plot power in logarithmic scale (dB) (default: False)
            vmin: Minimum value for colorbar scaling (optional)
            vmax: Maximum value for colorbar scaling (optional)
            db_colorbar: Whether to display colorbar in dB scale even with linear data (default: False)
            
        Returns:
            None
        """
        if len(self.results['Ez']) == 0 or len(self.results['Hx']) == 0 or len(self.results['Hy']) == 0:
            print("No field data to calculate power. Make sure to run the simulation with save=True.")
            return
        
        # Calculate power from Poynting vector components
        # For 2D TE mode with Ez, Hx, Hy, the time-averaged Poynting vector has components:
        # Sx = -Ez * Hy, Sy = Ez * Hx
        power = np.zeros((self.nx, self.ny))
        
        # Calculate power integrated over time
        for t_idx in range(len(self.results['t'])):
            Ez = self.results['Ez'][t_idx]
            
            # Need to interpolate H fields to same grid as Ez
            Hx_interp = np.zeros_like(Ez)
            Hy_interp = np.zeros_like(Ez)
            
            # Interpolate Hx (centered at (i, j+1/2)) to (i, j)
            Hx_interp[:, 1:-1] = 0.5 * (self.results['Hx'][t_idx][:, :-1] + self.results['Hx'][t_idx][:, 1:])
            Hx_interp[:, 0] = self.results['Hx'][t_idx][:, 0]  # Edge case
            Hx_interp[:, -1] = self.results['Hx'][t_idx][:, -1]  # Edge case
            
            # Interpolate Hy (centered at (i+1/2, j)) to (i, j)
            Hy_interp[1:-1, :] = 0.5 * (self.results['Hy'][t_idx][:-1, :] + self.results['Hy'][t_idx][1:, :])
            Hy_interp[0, :] = self.results['Hy'][t_idx][0, :]  # Edge case
            Hy_interp[-1, :] = self.results['Hy'][t_idx][-1, :]  # Edge case
            
            # Calculate Poynting vector components
            Sx = -Ez * Hy_interp
            Sy = Ez * Hx_interp
            
            # Add magnitude of Poynting vector to total power
            power += np.sqrt(Sx**2 + Sy**2)
        
        # Normalize by number of time steps
        power /= len(self.results['t'])
        
        # Find maximum power for normalization (used in both log_scale and db_colorbar)
        max_power = np.max(power)
        
        # Convert to dB if log_scale is True
        if log_scale:
            # Add small epsilon to avoid log(0)
            epsilon = np.finfo(float).eps
            # Scale by maximum value to make max = 0dB
            power_normalized = power / max_power
            power_db = 10 * np.log10(power_normalized + epsilon)
            plot_data = power_db
            power_label = 'Power (dB)'
            
            # Set default dB range from 0 to -40 dB if not specified
            if vmin is None:
                vmin = -40
            if vmax is None:
                vmax = 0
        else:
            plot_data = power
            if db_colorbar:
                power_label = 'Power (dB)'
            else:
                power_label = 'Power (W/m²)'
        
        # Calculate figure size based on grid dimensions
        aspect_ratio = self.ny / self.nx
        base_size = 6  # Base size for the smaller dimension
        if aspect_ratio > 1:
            figsize = (base_size, base_size * aspect_ratio)
        else:
            figsize = (base_size / aspect_ratio, base_size)
        
        # Create the figure
        plt.figure(figsize=figsize)
        im = plt.imshow(plot_data, origin='lower',
                       extent=(0, self.design.width, 0, self.design.height),
                       cmap=cmap, aspect='equal', interpolation='bicubic',
                       vmin=vmin, vmax=vmax)
        
        # Create colorbar with proper formatting
        if db_colorbar and not log_scale:
            # Create a custom formatter to display linear values in dB
            from matplotlib.ticker import FuncFormatter
            epsilon = np.finfo(float).eps
            
            def db_formatter(x, pos):
                # Convert linear value to dB relative to max_power
                return f"{10 * np.log10((x/max_power) + epsilon):.1f}"
            
            formatter = FuncFormatter(db_formatter)
            colorbar = plt.colorbar(im, format=formatter, label=power_label)
            
            # Set default range for dB colorbar from 0 to -40 dB
            if vmin is None and vmax is None:
                # We're modifying just the formatter, not the actual data range
                colorbar.set_ticks(np.linspace(max_power, max_power * 10**(-40/10), 5))
                colorbar.set_ticklabels(['0', '-10', '-20', '-30', '-40'])
        else:
            colorbar = plt.colorbar(im, label=power_label)
        
        plt.title('Time-integrated Power Distribution')
        
        # Set proper unit labels based on design dimensions
        max_dim = max(self.design.width, self.design.height)
        if max_dim >= 1e-3: scale, unit = 1e3, 'mm'
        elif max_dim >= 1e-6: scale, unit = 1e6, 'µm'
        elif max_dim >= 1e-9: scale, unit = 1e9, 'nm'
        else: scale, unit = 1e12, 'pm'
        
        plt.xlabel(f'X ({unit})')
        plt.ylabel(f'Y ({unit})')
        plt.gca().xaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
        plt.gca().yaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
        
        # Create an overlay of the design outlines
        for structure in self.design.structures:
            if isinstance(structure, Rectangle):
                if structure.is_pml:
                    rect = MatplotlibRectangle(
                        (structure.position[0], structure.position[1]),
                        structure.width, structure.height,
                        facecolor='none', edgecolor='white', alpha=1, linestyle=':')
                else:
                    rect = MatplotlibRectangle(
                        (structure.position[0], structure.position[1]),
                        structure.width, structure.height,
                        facecolor='none', edgecolor='white', alpha=0.5)
                plt.gca().add_patch(rect)
            elif isinstance(structure, Circle):
                circle = MatplotlibCircle(
                    (structure.position[0], structure.position[1]),
                    structure.radius,
                    facecolor='none', edgecolor='white', alpha=0.5, linestyle='--')
                plt.gca().add_patch(circle)
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
                ring_patch = PathPatch(path, facecolor='none', edgecolor='white', alpha=0.5, linestyle='--')
                plt.gca().add_patch(ring_patch)
            elif isinstance(structure, CircularBend):
                # Create points for the bend
                N = 100
                angle_rad = np.radians(structure.angle)
                rotation_rad = np.radians(structure.rotation)
                start_angle = rotation_rad
                end_angle = rotation_rad + angle_rad
                theta = np.linspace(start_angle, end_angle, N, endpoint=True)
                
                # Generate outer and inner curve points
                x_outer = structure.position[0] + structure.outer_radius * np.cos(theta)
                y_outer = structure.position[1] + structure.outer_radius * np.sin(theta)
                x_inner = structure.position[0] + structure.inner_radius * np.cos(theta)
                y_inner = structure.position[1] + structure.inner_radius * np.sin(theta)
                
                # Create vertex list for the bend shape
                vertices = np.vstack([
                    [x_outer[0], y_outer[0]],
                    *np.column_stack([x_outer[1:], y_outer[1:]]),
                    [x_inner[-1], y_inner[-1]],
                    *np.column_stack([x_inner[-2::-1], y_inner[-2::-1]]),
                    [x_outer[0], y_outer[0]]
                ])
                
                codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 2) + [Path.CLOSEPOLY]
                path = Path(vertices, codes)
                bend_patch = PathPatch(path, facecolor='none', edgecolor='white', alpha=0.5, linestyle='--')
                plt.gca().add_patch(bend_patch)
            elif isinstance(structure, ModeSource):
                plt.plot((structure.start[0], structure.end[0]), 
                         (structure.start[1], structure.end[1]), 
                         '-', lw=4, color="crimson", alpha=0.5)
                plt.plot((structure.start[0], structure.end[0]), 
                         (structure.start[1], structure.end[1]), 
                         '--', lw=2, color="white", alpha=0.5)
            elif isinstance(structure, GaussianSource):
                source_circle = MatplotlibCircle(
                    (structure.position[0], structure.position[1]),
                    structure.width,
                    facecolor='none', edgecolor='orange', alpha=0.8, linestyle='dotted')
                plt.gca().add_patch(source_circle)
            elif isinstance(structure, Monitor):
                plt.plot((structure.start[0], structure.end[0]), 
                          (structure.start[1], structure.end[1]), 
                         '-', lw=4, color="navy", alpha=0.5)
        
        plt.tight_layout()
        plt.show()