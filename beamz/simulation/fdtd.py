from typing import Dict, Optional
import numpy as np
from beamz.const import *
from beamz.simulation.meshing import RegularGrid, RegularGrid3D, create_mesh
from beamz.design.sources import ModeSource, GaussianSource
from beamz.design.structures import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MatplotlibRectangle, Circle as MatplotlibCircle, PathPatch
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation
from beamz.simulation.backends import get_backend
from beamz.helpers import (
    display_header, display_status, create_rich_progress, 
    display_parameters, display_results, display_simulation_status, 
    display_time_elapsed
)
import datetime
from beamz.helpers import get_si_scale_and_label

class FDTD:
    """FDTD simulation class supporting both 2D and 3D electromagnetic simulations.
    
    Automatically detects whether the design is 2D or 3D and uses appropriate Maxwell equations:
    - 2D: TE-polarized (Ez, Hx, Hy fields)
    - 3D: Full Maxwell equations (Ex, Ey, Ez, Hx, Hy, Hz fields)
    """
    def __init__(self, design, time, mesh: str = "regular", resolution: float = 0.02*µm, backend="numpy", backend_options=None):
        # Initialize the design and detect dimensionality
        self.design = design
        self.resolution = resolution
        self.is_3d = design.is_3d and design.depth > 0
        
        # Initialize appropriate mesh based on dimensionality
        if mesh == "regular":
            if self.is_3d:
                self.mesh = RegularGrid3D(design=self.design, resolution_xy=self.resolution)
                display_status("Using 3D FDTD with full Maxwell equations", "info")
            else:
                self.mesh = RegularGrid(design=self.design, resolution=self.resolution)
                display_status("Using 2D FDTD with TE-polarized Maxwell equations", "info")
        else: self.mesh = None
        
        # Set grid resolutions
        self.dx = self.mesh.dx if hasattr(self.mesh, 'dx') else self.mesh.resolution_xy
        self.dy = self.mesh.dy if hasattr(self.mesh, 'dy') else self.mesh.resolution_xy
        if self.is_3d: self.dz = self.mesh.dz if hasattr(self.mesh, 'dz') else self.mesh.resolution_z
        
        # Set grid dimensions from mesh
        if self.is_3d:
            # For 3D mesh: shape is (nz, ny, nx)
            self.nz, self.ny, self.nx = self.mesh.permittivity.shape
        else:
            # For 2D mesh: shape is (ny, nx)
            self.ny, self.nx = self.mesh.permittivity.shape
        # Get material properties
        self.epsilon_r = self.mesh.permittivity
        self.mu_r = self.mesh.permeability
        self.sigma = self.mesh.conductivity
        
        # Initialize the backend
        backend_options = backend_options or {}
        self.backend = get_backend(name=backend, **backend_options)
        
        # Initialize fields based on dimensionality
        self._init_fields()
        
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
        
        # Initialize the results based on dimensionality
        if self.is_3d: self.results = {"Ex": [], "Ey": [], "Ez": [], "Hx": [], "Hy": [], "Hz": [], "t": []}
        else: self.results = {"Ez": [], "Hx": [], "Hy": [], "t": []}
            
        # Initialize animation attributes
        self.fig = None
        self.ax = None
        self.anim = None
        self.im = None
        
        # Initialize monitor data storage
        self.monitor_data = {}
        
        # Initialize power accumulation
        self.power_accumulated = None
        self.power_accumulation_count = 0
        
        # Initialize simulation start time
        self.start_time = None

    def _init_fields(self):
        """Initialize field arrays based on dimensionality."""
        if self.is_3d: 
            try:
                # Yee grid staggering in 3D (arrays ordered as [z, y, x])
                # Electric fields live on edges
                self.Ex = self.backend.zeros((self.nz,     self.ny,     self.nx-1), dtype=np.complex128)
                self.Ey = self.backend.zeros((self.nz,     self.ny-1,   self.nx    ), dtype=np.complex128)
                self.Ez = self.backend.zeros((self.nz-1,   self.ny,     self.nx    ), dtype=np.complex128)
                # Magnetic fields live on faces (Yee grid)
                # Hx: (nz-1, ny-1, nx), Hy: (nz-1, ny, nx-1), Hz: (nz, ny-1, nx-1)
                self.Hx = self.backend.zeros((self.nz-1,   self.ny-1,   self.nx    ), dtype=np.complex128)
                self.Hy = self.backend.zeros((self.nz-1,   self.ny,     self.nx-1  ), dtype=np.complex128)
                self.Hz = self.backend.zeros((self.nz,     self.ny-1,   self.nx-1  ), dtype=np.complex128)
            except TypeError:
                # Fallback if dtype not supported
                self.Ex = self.backend.zeros((self.nz,     self.ny,     self.nx-1))
                self.Ey = self.backend.zeros((self.nz,     self.ny-1,   self.nx    ))
                self.Ez = self.backend.zeros((self.nz-1,   self.ny,     self.nx    ))
                self.Hx = self.backend.zeros((self.nz-1,   self.ny-1,   self.nx    ))
                self.Hy = self.backend.zeros((self.nz-1,   self.ny,     self.nx-1  ))
                self.Hz = self.backend.zeros((self.nz,     self.ny-1,   self.nx-1  ))
                self.is_complex_backend = False
            else: self.is_complex_backend = True
        else: 
            try:
                # Try with dtype parameter if supported - use (ny, nx) format to match backend
                self.Ez = self.backend.zeros((self.ny, self.nx), dtype=np.complex128)
            except TypeError:
                # Fallback if dtype not supported - create real array and handle complex values in code
                self.Ez = self.backend.zeros((self.ny, self.nx))
                self.is_complex_backend = False
            else: self.is_complex_backend = True
            # Staggered magnetic fields: Hx has same rows as Ez and one fewer column; Hy has one fewer row and same columns
            self.Hx = self.backend.zeros((self.ny, self.nx-1))
            self.Hy = self.backend.zeros((self.ny-1, self.nx))


    def simulate_step(self):
        """Perform one FDTD step with stability checks."""
        if self.is_3d: self._update_3d_fields()
        else:
            # 2D TE-polarized Maxwell equations update
            self.Hx, self.Hy = self.backend.update_h_fields(
                self.Hx, self.Hy, self.Ez, self.sigma, 
                self.dx, self.dy, self.dt, MU_0, EPS_0)
            self.Ez = self.backend.update_e_field(
                self.Ez, self.Hx, self.Hy, self.sigma, self.epsilon_r,
                self.dx, self.dy, self.dt, EPS_0)

    def _update_3d_fields(self):
        """Full 3D Yee update for all 6 field components with PML via sigma arrays."""
        dt = self.dt; dx = self.dx; dy = self.dy; dz = self.dz
        mu0 = MU_0; eps0 = EPS_0

        # Convenience references
        Ex = self.Ex; Ey = self.Ey; Ez = self.Ez
        Hx = self.Hx; Hy = self.Hy; Hz = self.Hz
        eps_r = self.epsilon_r; sigma = self.sigma

        # --- Update H fields (n -> n+1/2) ---
        # Magnetic conductivities at staggered positions
        sigma_m_hx = (sigma[:-1, :-1, :] * mu0 / eps0) if sigma.ndim == 3 else sigma * 0
        sigma_m_hy = (sigma[:-1, :, :-1] * mu0 / eps0) if sigma.ndim == 3 else sigma * 0
        sigma_m_hz = (sigma[:, :-1, :-1] * mu0 / eps0) if sigma.ndim == 3 else sigma * 0

        # Curls of E at H locations
        # Hx shape: (nz-1, ny-1, nx)
        dEz_dy = (Ez[:, 1:, :] - Ez[:, :-1, :]) / dy          # (nz-1, ny-1, nx)
        dEy_dz = (Ey[1:, :, :] - Ey[:-1, :, :]) / dz          # (nz-1, ny-1, nx)
        curlE_x = dEz_dy - dEy_dz

        # Hy shape: (nz-1, ny, nx-1)
        dEx_dz = (Ex[1:, :, :] - Ex[:-1, :, :]) / dz          # (nz-1, ny, nx-1)
        dEz_dx = (Ez[:, :, 1:] - Ez[:, :, :-1]) / dx          # (nz-1, ny, nx-1)
        curlE_y = dEx_dz - dEz_dx

        # Hz shape: (nz, ny-1, nx-1)
        dEy_dx = (Ey[:, :, 1:] - Ey[:, :, :-1]) / dx          # (nz, ny-1, nx-1)
        dEx_dy = (Ex[:, 1:, :] - Ex[:, :-1, :]) / dy          # (nz, ny-1, nx-1)
        curlE_z = dEy_dx - dEx_dy

        # Semi-implicit PML factors for H
        denom_hx = 1.0 + sigma_m_hx * dt / (2.0 * mu0)
        factor_hx = (1.0 - sigma_m_hx * dt / (2.0 * mu0)) / denom_hx
        source_hx = (dt / mu0) / denom_hx

        denom_hy = 1.0 + sigma_m_hy * dt / (2.0 * mu0)
        factor_hy = (1.0 - sigma_m_hy * dt / (2.0 * mu0)) / denom_hy
        source_hy = (dt / mu0) / denom_hy

        denom_hz = 1.0 + sigma_m_hz * dt / (2.0 * mu0)
        factor_hz = (1.0 - sigma_m_hz * dt / (2.0 * mu0)) / denom_hz
        source_hz = (dt / mu0) / denom_hz

        Hx[:] = factor_hx * Hx - source_hx * curlE_x
        Hy[:] = factor_hy * Hy - source_hy * curlE_y
        Hz[:] = factor_hz * Hz - source_hz * curlE_z

        # --- Update E fields (n+1/2 -> n+1) ---
        # Electric conductivities and permittivities at E locations (use interior slices)
        # Ex interior: [1:-1, 1:-1, :]
        if eps_r.ndim == 3:
            eps_ex = eps_r[1:-1, 1:-1, :-1]
            sig_ex = sigma[1:-1, 1:-1, :-1]
        else:
            eps_ex = eps_r
            sig_ex = sigma

        # Ey interior: [1:-1, :, 1:-1]
        if eps_r.ndim == 3:
            eps_ey = eps_r[1:-1, :-1, 1:-1]
            sig_ey = sigma[1:-1, :-1, 1:-1]
        else:
            eps_ey = eps_r
            sig_ey = sigma

        # Ez interior: [:, 1:-1, 1:-1]
        if eps_r.ndim == 3:
            eps_ez = eps_r[:-1, 1:-1, 1:-1]
            sig_ez = sigma[:-1, 1:-1, 1:-1]
        else:
            eps_ez = eps_r
            sig_ez = sigma

        # Curls of H at E locations (interior updates)
        # Ex interior indices
        dHz_dy_ex = (Hz[:, 1:, :] - Hz[:, :-1, :]) / dy              # (nz, ny-2, nx-1)
        dHy_dz_ex = (Hy[1:, :, :] - Hy[:-1, :, :]) / dz              # (nz-2, ny, nx-1)
        # Align to (nz-2, ny-2, nx-1)
        curlH_x = dHz_dy_ex[1:-1, :, :] - dHy_dz_ex[:, 1:-1, :]

        # Ey interior indices
        dHx_dz_ey = (Hx[1:, :, :] - Hx[:-1, :, :]) / dz              # (nz-2, ny-1, nx)
        dHz_dx_ey = (Hz[:, :, 1:] - Hz[:, :, :-1]) / dx              # (nz, ny-1, nx-1)
        # Align to (nz-2, ny-1, nx-2)
        curlH_y = dHx_dz_ey[:, :, 1:-1] - dHz_dx_ey[1:-1, :, :]

        # Ez interior indices
        dHy_dx_ez = (Hy[:, :, 1:] - Hy[:, :, :-1]) / dx              # (nz-1, ny, nx-2)
        dHx_dy_ez = (Hx[:, 1:, :] - Hx[:, :-1, :]) / dy              # (nz-1, ny-2, nx)
        # Align to (nz-1, ny-2, nx-2)
        curlH_z = dHy_dx_ez[:, 1:-1, :] - dHx_dy_ez[:, :, 1:-1]

        # Semi-implicit PML factors for E
        denom_ex = 1.0 + sig_ex * dt / (2.0 * eps0 * eps_ex)
        factor_ex = (1.0 - sig_ex * dt / (2.0 * eps0 * eps_ex)) / denom_ex
        source_ex = (dt / (eps0 * eps_ex)) / denom_ex

        denom_ey = 1.0 + sig_ey * dt / (2.0 * eps0 * eps_ey)
        factor_ey = (1.0 - sig_ey * dt / (2.0 * eps0 * eps_ey)) / denom_ey
        source_ey = (dt / (eps0 * eps_ey)) / denom_ey

        denom_ez = 1.0 + sig_ez * dt / (2.0 * eps0 * eps_ez)
        factor_ez = (1.0 - sig_ez * dt / (2.0 * eps0 * eps_ez)) / denom_ez
        source_ez = (dt / (eps0 * eps_ez)) / denom_ez

        # Apply updates to interior regions
        Ex[1:-1, 1:-1, :] = factor_ex * Ex[1:-1, 1:-1, :] + source_ex * (curlH_x)
        Ey[1:-1, :, 1:-1] = factor_ey * Ey[1:-1, :, 1:-1] + source_ey * (curlH_y)
        Ez[:, 1:-1, 1:-1] = factor_ez * Ez[:, 1:-1, 1:-1] + source_ez * (curlH_z)

    def initialize_simulation(self, save=True, live=True, axis_scale=[-1,1], save_animation=False, 
                             animation_filename='fdtd_animation.mp4', clean_visualization=True, 
                             save_fields=None, decimate_save=1, accumulate_power=False,
                             save_memory_mode=False):
        """Initialize the simulation before running steps.
        
        Args:
            save: Whether to save field data at each step.
            live: Whether to show live animation of the simulation.
            axis_scale: Color scale limits for the field visualization.
            save_animation: Whether to save an animation of the simulation as an mp4 file.
            animation_filename: Filename for the saved animation (must end in .mp4).
            save_fields: List of fields to save (None = auto-select based on dimensionality)
            decimate_save: Save only every nth time step (1 = save all, 10 = save every 10th step)
            accumulate_power: Instead of saving all fields, accumulate power and save that
            save_memory_mode: If True, avoid storing all field data and only keep monitors/power
        """
        # Set default save_fields based on dimensionality
        if save_fields is None:
            if self.is_3d:
                save_fields = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
            else:
                save_fields = ['Ez', 'Hx', 'Hy']
        
        # Record start time
        self.start_time = datetime.datetime.now()
        # Initialize simulation state
        self.t = 0
        self.current_step = 0
        self._total_steps = self.num_steps
        self._save_results = save
        self._save_fields = save_fields
        self._decimate_save = decimate_save
        self._live = live
        self._axis_scale = axis_scale
        
        # Save mode flags as class attributes for monitor access
        self.save_memory_mode = save_memory_mode
        self.accumulate_power = accumulate_power
        
        # Display simulation header and parameters
        sim_params = {
            "Domain size": f"{self.design.width:.2e} x {self.design.height:.2e} m",
            "Resolution": f"{self.resolution:.2e} m",
            "Time steps": self.num_steps,
            "Time delta": f"{self.dt:.2e} s",
            "Total time": f"{self.time[-1]:.2e} s",
            "Backend": self.backend.__class__.__name__,
            "Save fields": ", ".join(save_fields),
            "Memory-saving mode": "Enabled" if save_memory_mode else "Disabled",
            "Accumulate power": "Enabled" if accumulate_power else "Disabled",
            "Live animation": "Enabled" if live else "Disabled"
        }
        display_parameters(sim_params, "Simulation Parameters")
        
        # Check stability using the helper function
        from beamz.helpers import check_fdtd_stability
        n_max = np.sqrt(np.max(self.epsilon_r))
        is_stable, courant, safe_limit = check_fdtd_stability(dt=self.dt, dx=self.dx, dy=self.dy, n_max=n_max, safety_factor=1.0)
        if not is_stable:
            display_status(f"Simulation may be unstable! Courant number = {courant:.3f} > {safe_limit:.3f}", "warning")
            display_status("Consider reducing dt or increasing dx/dy", "warning")
        else: 
            display_status(f"Stability check passed (Courant number = {courant:.3f} / {safe_limit:.3f})", "success")
        
        # Set up power accumulation if requested
        if accumulate_power:
            self.power_accumulated = np.zeros((self.ny, self.nx))
            self.power_accumulation_count = 0
            
        # Determine optimal save frequency based on backend type
        is_gpu_backend = hasattr(self.backend, 'device') and getattr(self.backend, 'device', '') in ['cuda', 'mps', 'gpu']
        # For GPU backends, avoid excessive CPU-GPU transfers by batching the result saves
        if is_gpu_backend and save:
            save_freq = max(10, self.num_steps // 100)  # Save approximately every 1% of steps or min of 10 steps
            display_status(f"GPU backend detected: Optimizing result storage (saving every {save_freq} steps)", "info")
        else: 
            save_freq = 1  # Save every step for CPU backends
            
        # Apply additional decimation based on user setting
        self._effective_save_freq = save_freq * decimate_save
        
        # If in save_memory_mode, clear any existing results to start fresh
        if save_memory_mode: 
            for field in self.results: 
                if field != 't': 
                    self.results[field] = []
            display_status("Memory-saving mode active: Only storing monitor data and/or power accumulation", "info")

    def step(self) -> bool:
        """Perform one simulation step. Returns True if simulation should continue, False if complete."""
        if self.current_step >= self.num_steps:
            return False
            
        # Update fields
        self.simulate_step()
        
        # Apply sources
        self._apply_sources()
        
        # Record monitor data
        self._record_monitor_data(self.current_step)
        
        # Accumulate power if requested
        self._accumulate_power()
        
        # Save results if requested and at the right frequency
        self._save_step_results()
        
        # Show live animation if requested
        self._update_live_animation()

        # Update time & step counter
        self.t += self.dt
        self.current_step += 1
        
        return True

    def finalize_simulation(self):
        """Clean up and finalize the simulation."""
        # Clean up animation
        if self._live and self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.im = None
                
        # Display completion information
        display_status("Simulation complete!", "success")
        display_time_elapsed(self.start_time)
        
        # Calculate final power average if accumulating
        if self.accumulate_power and self.power_accumulation_count > 0:
            self.power_accumulated /= self.power_accumulation_count
            
        # Display memory usage estimate
        memory_usage = self.estimate_memory_usage(time_steps=self.num_steps, save_fields=self._save_fields)
        display_status(f"Estimated memory usage: {memory_usage['Full simulation']['Total memory (MB)']:.2f} MB", "info")
        
        return self.results

    def run(self, steps: Optional[int] = None, save=True, live=True, axis_scale=[-1,1], save_animation=False, 
            animation_filename='fdtd_animation.mp4', clean_visualization=True, 
            save_fields=None, decimate_save=1, accumulate_power=False,
            save_memory_mode=False) -> Dict:
        """Run the complete simulation using the new step-by-step approach.
        
        Args:
            steps: Number of steps to run. If None, run until the end of the time array.
            save: Whether to save field data at each step.
            live: Whether to show live animation of the simulation.
            axis_scale: Color scale limits for the field visualization.
            save_animation: Whether to save an animation of the simulation as an mp4 file.
            animation_filename: Filename for the saved animation (must end in .mp4).
            save_fields: List of fields to save (None = auto-select based on dimensionality)
            decimate_save: Save only every nth time step (1 = save all, 10 = save every 10th step)
            accumulate_power: Instead of saving all fields, accumulate power and save that
            save_memory_mode: If True, avoid storing all field data and only keep monitors/power
        
        Returns:
            Dictionary containing the simulation results.
        """
        # Initialize the simulation
        self.initialize_simulation(save=save, live=live, axis_scale=axis_scale, 
                                  save_animation=save_animation, 
                                  animation_filename=animation_filename,
                                  clean_visualization=clean_visualization,
                                  save_fields=save_fields, decimate_save=decimate_save,
                                  accumulate_power=accumulate_power, 
                                  save_memory_mode=save_memory_mode)
        
        # Run the simulation with progress tracking
        with create_rich_progress() as progress:
            task = progress.add_task("Running simulation...", total=self.num_steps)
            
            while self.step():
                progress.update(task, advance=1)
        
        # Finalize and return results
        return self.finalize_simulation()

    def _apply_sources(self):
        """Apply all sources for the current time step."""
        for source in self.sources:
            if isinstance(source, ModeSource):
                # Get the mode profile for the first mode
                mode_profile = source.mode_profiles[0]
                # Get the time modulation for this step
                modulation = source.signal[self.current_step]
                # Apply the mode profile to all points
                for point in mode_profile:
                    if len(point) == 4:  # 3D mode profile: [amplitude, x, y, z]
                        amplitude, x_raw, y_raw, z_raw = point
                    else:  # 2D mode profile: [amplitude, x, y]
                        amplitude, x_raw, y_raw = point
                        z_raw = 0
                        
                    # Convert the position to the nearest grid point using correct resolutions
                    x = int(round(x_raw / self.dx))
                    y = int(round(y_raw / self.dy))
                    
                    if self.is_3d:
                        z = int(round(z_raw / self.dz))
                        # Skip points outside the 3D grid
                        if (x < 0 or x >= self.nx or y < 0 or y >= self.ny or 
                            z < 0 or z >= self.nz): continue
                        # Apply to center z-slice or specified z-coordinate
                        z_target = min(z, self.Ez.shape[0] - 1) if z < self.Ez.shape[0] else self.Ez.shape[0] // 2
                    else:
                        # Skip points outside the 2D grid
                        if x < 0 or x >= self.nx or y < 0 or y >= self.ny: continue
                        z_target = None
                    
                    # Add the source contribution to the field (handling complex values)
                    if hasattr(self, 'is_complex_backend') and not self.is_complex_backend:
                        # For backends that don't support complex numbers, we need to extract real part
                        if isinstance(amplitude * modulation, complex):
                            source_value = np.real(amplitude * modulation)
                        else: source_value = amplitude * modulation
                    else: source_value = amplitude * modulation

                    # Apply source to appropriate field component and location
                    if self.is_3d:
                        self.Ez[z_target, y, x] += source_value
                        # Apply direction constraints for 3D
                        if source.direction == "+x" and x > 0: self.Ez[z_target, y, x-1] = 0
                        elif source.direction == "-x" and x < self.nx-1: self.Ez[z_target, y, x+1] = 0
                        elif source.direction == "+y" and y > 0: self.Ez[z_target, y-1, x] = 0
                        elif source.direction == "-y" and y < self.ny-1: self.Ez[z_target, y+1, x] = 0
                        elif source.direction == "+z" and z_target > 0: self.Ez[z_target-1, y, x] = 0
                        elif source.direction == "-z" and z_target < self.Ez.shape[0]-1: self.Ez[z_target+1, y, x] = 0
                    else:
                        self.Ez[y, x] += source_value
                        # Apply direction constraints for 2D
                        if source.direction == "+x" and x > 0: self.Ez[y, x-1] = 0
                        elif source.direction == "-x" and x < self.nx-1: self.Ez[y, x+1] = 0
                        elif source.direction == "+y" and y > 0: self.Ez[y-1, x] = 0
                        elif source.direction == "-y" and y < self.ny-1: self.Ez[y+1, x] = 0
                        
            elif isinstance(source, GaussianSource):
                modulation = source.signal[self.current_step]
                # Get 3D position (sources now always have 3D positions)
                center_x_phys, center_y_phys, center_z_phys = source.position
                width_phys = source.width  # Assuming width is standard deviation (sigma)
                
                # Convert physical units to grid coordinates
                center_x_grid = center_x_phys / self.dx
                center_y_grid = center_y_phys / self.dy
                
                if self.is_3d:
                    center_z_grid = center_z_phys / self.dz
                    # sigma in grid units
                    width_x_grid = width_phys / self.dx 
                    width_y_grid = width_phys / self.dy 
                    width_z_grid = width_phys / self.dz
                    
                    # Define the grid range to apply the source (e.g., +/- 3 sigma)
                    wx_grid_cells = max(1, int(round(3 * width_x_grid)))
                    wy_grid_cells = max(1, int(round(3 * width_y_grid)))
                    wz_grid_cells = max(1, int(round(3 * width_z_grid)))
                    
                    # Calculate bounding box indices, clamped to grid boundaries
                    x_center_idx = int(round(center_x_grid))
                    y_center_idx = int(round(center_y_grid))
                    z_center_idx = int(round(center_z_grid))
                    
                    x_start = max(0, x_center_idx - wx_grid_cells)
                    x_end = min(self.nx, x_center_idx + wx_grid_cells + 1)
                    y_start = max(0, y_center_idx - wy_grid_cells)
                    y_end = min(self.ny, y_center_idx + wy_grid_cells + 1)
                    z_start = max(0, z_center_idx - wz_grid_cells)
                    z_end = min(self.nz, z_center_idx + wz_grid_cells + 1)
                    
                    # Ensure z_end doesn't exceed Ez array bounds
                    z_end = min(z_end, self.Ez.shape[0])
                    
                    # Create meshgrid for the affected 3D area
                    z_indices = np.arange(z_start, z_end)
                    y_indices = np.arange(y_start, y_end)
                    x_indices = np.arange(x_start, x_end)
                    z_grid, y_grid, x_grid = np.meshgrid(z_indices, y_indices, x_indices, indexing='ij')
                    
                    # Calculate squared distances from the center in grid units
                    dist_x_sq = (x_grid - center_x_grid)**2
                    dist_y_sq = (y_grid - center_y_grid)**2
                    dist_z_sq = (z_grid - center_z_grid)**2
                    
                    # Calculate Gaussian amplitude for 3D (no artificial normalization to preserve amplitude)
                    epsilon = 1e-9
                    sigma_x_sq = width_x_grid**2 + epsilon
                    sigma_y_sq = width_y_grid**2 + epsilon
                    sigma_z_sq = width_z_grid**2 + epsilon
                    exponent = -(dist_x_sq / (2 * sigma_x_sq) + dist_y_sq / (2 * sigma_y_sq) + dist_z_sq / (2 * sigma_z_sq))
                    gaussian_amp = np.exp(exponent)
                    
                    gaussian_amp = self.backend.from_numpy(gaussian_amp)
                    # Add the source contribution to the Ez field (inject at nearest Yee Ez plane)
                    # Ez has shape (nz-1, ny, nx); clamp z to valid index
                    z_ez_idx = max(0, min(self.Ez.shape[0]-1, z_start))
                    self.Ez[z_ez_idx:z_ez_idx + (z_end - z_start), y_start:y_end, x_start:x_end] += gaussian_amp[:(z_end - z_start), :, :] * modulation
                    
                else:
                    # 2D Gaussian source (original implementation)
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
                    gaussian_amp = np.exp(exponent) / 4  # 2D normalization
                    
                    gaussian_amp = self.backend.from_numpy(gaussian_amp)
                    # Add the source contribution to the Ez field slice
                    self.Ez[y_start:y_end, x_start:x_end] += gaussian_amp * modulation

    def _accumulate_power(self):
        """Accumulate power for this time step if requested."""
        if not self.accumulate_power:
            return
            
        if self.is_3d:
            # 3D power calculation with proper field interpolation
            Ex_np = self.backend.to_numpy(self.Ex)
            Ey_np = self.backend.to_numpy(self.Ey)
            Ez_np = self.backend.to_numpy(self.Ez)
            Hx_np = self.backend.to_numpy(self.Hx)
            Hy_np = self.backend.to_numpy(self.Hy)
            Hz_np = self.backend.to_numpy(self.Hz)
            
            # For simplified power calculation, use the center region where all fields overlap
            # This avoids complex interpolation while still giving meaningful power distribution
            min_z = min(Ex_np.shape[0], Ey_np.shape[0], Ez_np.shape[0], Hx_np.shape[0], Hy_np.shape[0], Hz_np.shape[0])
            min_y = min(Ex_np.shape[1], Ey_np.shape[1], Ez_np.shape[1], Hx_np.shape[1], Hy_np.shape[1], Hz_np.shape[1])
            min_x = min(Ex_np.shape[2], Ey_np.shape[2], Ez_np.shape[2], Hx_np.shape[2], Hy_np.shape[2], Hz_np.shape[2])
            
            # Extract overlapping regions
            Ex_center = Ex_np[:min_z, :min_y, :min_x]
            Ey_center = Ey_np[:min_z, :min_y, :min_x]
            Ez_center = Ez_np[:min_z, :min_y, :min_x]
            Hx_center = Hx_np[:min_z, :min_y, :min_x]
            Hy_center = Hy_np[:min_z, :min_y, :min_x]
            Hz_center = Hz_np[:min_z, :min_y, :min_x]
            
            # Calculate Poynting vector S = E × H (take real part for power)
            Sx = np.real(Ey_center * np.conj(Hz_center) - Ez_center * np.conj(Hy_center))
            Sy = np.real(Ez_center * np.conj(Hx_center) - Ex_center * np.conj(Hz_center))
            Sz = np.real(Ex_center * np.conj(Hy_center) - Ey_center * np.conj(Hx_center))
            
            # Calculate power magnitude |S|
            power_mag = np.sqrt(Sx**2 + Sy**2 + Sz**2)
            
            # Initialize or accumulate power
            if self.power_accumulated is None:
                self.power_accumulated = power_mag.copy()
            else:
                # Ensure shapes match before adding (handle 2D vs 3D transitions)
                if self.power_accumulated.shape != power_mag.shape:
                    self.power_accumulated = power_mag.copy()
                    self.power_accumulation_count = 0
                self.power_accumulated += power_mag
            self.power_accumulation_count += 1
        else:
            # 2D power calculation
            Ez_np = self.backend.to_numpy(self.Ez)
            Hx_np = self.backend.to_numpy(self.Hx)
            Hy_np = self.backend.to_numpy(self.Hy)
            
            # Check if any field is complex
            is_complex = np.iscomplexobj(Ez_np) or np.iscomplexobj(Hx_np) or np.iscomplexobj(Hy_np)
            
            # Handle complex Ez data
            if np.iscomplexobj(Ez_np):
                Ez_real = np.real(Ez_np)
                Ez_imag = np.imag(Ez_np)
            else:
                Ez_real = Ez_np
                Ez_imag = np.zeros_like(Ez_np)
            
            # Extend magnetic fields to match Ez dimensions
            # Create arrays with matching dtype to avoid warnings
            if is_complex:
                Hx_full = np.zeros_like(Ez_np, dtype=np.complex128)
                Hy_full = np.zeros_like(Ez_np, dtype=np.complex128)
            else:
                Hx_full = np.zeros_like(Ez_real)
                Hy_full = np.zeros_like(Ez_real)
                
            # Properly handle filling the arrays for 2D
            if np.iscomplexobj(Hx_np):
                Hx_full[:, :-1] = Hx_np
            else:
                Hx_full[:, :-1] = Hx_np
                
            if np.iscomplexobj(Hy_np):
                Hy_full[:-1, :] = Hy_np
            else:
                Hy_full[:-1, :] = Hy_np
            
            # For complex fields, extract real/imag parts for power calculation
            if is_complex:
                Hx_real = np.real(Hx_full)
                Hx_imag = np.imag(Hx_full)
                Hy_real = np.real(Hy_full)
                Hy_imag = np.imag(Hy_full)
                
                # Calculate Poynting vector components for real and imaginary parts
                # Using complete formula for complex Poynting vector:
                # S = (1/2) Re[E × H*] where H* is complex conjugate of H
                Sx = -Ez_real * Hy_real - Ez_imag * Hy_imag
                Sy = Ez_real * Hx_real + Ez_imag * Hx_imag
            else:
                # Real-only fields, simple calculation
                Sx = -Ez_real * Hy_full
                Sy = Ez_real * Hx_full
            
            # Calculate power magnitude (|S|²)
            power_mag = Sx**2 + Sy**2
            
            # Initialize or accumulate power
            if self.power_accumulated is None:
                self.power_accumulated = power_mag.copy()
            else:
                # Ensure shapes match before adding (handle 2D vs 3D transitions)
                if self.power_accumulated.shape != power_mag.shape:
                    self.power_accumulated = power_mag.copy()
                    self.power_accumulation_count = 0
                self.power_accumulated += power_mag
            self.power_accumulation_count += 1

    def _save_step_results(self):
        """Save results for this time step if requested and at the right frequency."""
        if (self._save_results and not self.save_memory_mode and 
            (self.current_step % self._effective_save_freq == 0 or self.current_step == self.num_steps - 1)):
            # Convert arrays to numpy for saving (use magnitude for complex to increase visibility)
            self.results['t'].append(self.t)
            for field in self._save_fields:
                arr = getattr(self, field)
                arr_np = self.backend.to_numpy(self.backend.copy(arr))
                if np.iscomplexobj(arr_np):
                    arr_np = np.abs(arr_np)
                self.results[field].append(arr_np)

    def plot_field(self, field: str = "Ez", t: float = None, z_slice: int = None) -> None:
        """Plot a field at a given time with proper scaling and units.
        
        Args:
            field: Field component to plot ('Ez', 'Ex', 'Ey', 'Hx', 'Hy', 'Hz')
            t: Time to plot (if None, uses current field)
            z_slice: For 3D simulations, which z-slice to plot (if None, uses center)
        """
        if len(self.results['t']) == 0:
            current_field = getattr(self, field)
            current_t = self.t
        else:
            t_idx = np.argmin(np.abs(np.array(self.results['t']) - t))
            current_field = self.results[field][t_idx]
            current_t = self.results['t'][t_idx]
            
        # Convert to NumPy array if it's a backend array
        if hasattr(current_field, 'device'):
            current_field = self.backend.to_numpy(current_field)
        
        # Handle 3D fields by taking a 2D slice
        if self.is_3d and len(current_field.shape) == 3:
            if z_slice is None:
                z_slice = current_field.shape[0] // 2  # Use center slice
            current_field = current_field[z_slice, :, :]
            slice_info = f" (z-slice {z_slice})"
        else:
            slice_info = ""
            
        # Handle complex data - we'll display the real part for visualization
        if np.iscomplexobj(current_field):
            current_field = np.real(current_field)
            field_label = f'Re({field}){slice_info}'
        else:
            field_label = field + slice_info
            
        # Determine appropriate SI unit and scale for spatial dimensions
        scale, unit = get_si_scale_and_label(max(self.design.width, self.design.height))
        # Calculate figure size based on grid dimensions
        grid_height, grid_width = current_field.shape
        aspect_ratio = grid_width / grid_height
        base_size = 6  # Base size for the smaller dimension
        if aspect_ratio > 1: figsize = (base_size * aspect_ratio, base_size)
        else: figsize = (base_size, base_size / aspect_ratio)
        # Create the figure with the added structure outlines
        plt.figure(figsize=figsize)
        plt.imshow(current_field, origin='lower', 
                  extent=(0, self.design.width, 0, self.design.height),
                  cmap='RdBu', aspect='equal', interpolation='bicubic')
        plt.colorbar(label=f'{field_label} Field Amplitude')
        plt.title(f'{field_label} Field at t = {current_t:.2e} s')
        plt.xlabel(f'X ({unit})'); plt.ylabel(f'Y ({unit})')
        plt.gca().xaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
        plt.gca().yaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
        for structure in self.design.structures:
            # Use dashed lines for PML regions
            if hasattr(structure, 'is_pml') and structure.is_pml:
                structure.add_to_plot(plt.gca(), edgecolor="black", linestyle='--', facecolor='none', alpha=0.5)
            else:
                structure.add_to_plot(plt.gca(), facecolor="none", edgecolor="black", linestyle="-")
        plt.tight_layout()
        plt.show()

    def animate_live(self, field_data=None, field="Ez", axis_scale=[-1,1], z_slice=None):
        """Animate the field in real time using matplotlib animation.
        
        Args:
            field_data: Field data to animate (if None, gets from current field)
            field: Field component to animate
            axis_scale: Color scale limits
            z_slice: For 3D simulations, which z-slice to animate (if None, uses center)
        """
        if field_data is None: 
            field_data = self.backend.to_numpy(getattr(self, field))
        
        # Handle 3D fields by taking a 2D slice
        if self.is_3d and len(field_data.shape) == 3:
            if z_slice is None:
                z_slice = field_data.shape[0] // 2  # Use center slice
            field_data = field_data[z_slice, :, :]
            slice_info = f" (z-slice {z_slice})"
        else:
            slice_info = ""
        
        # Handle complex data - we'll display the real part for visualization
        if np.iscomplexobj(field_data):
            field_data = np.real(field_data)
            
        # If animation already exists, just update the data
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            current_field = field_data
            self.im.set_array(current_field)
            self.ax.set_title(f't = {self.t:.2e} s{slice_info}')
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
        # Create the figure and axis
        self.fig, self.ax = plt.subplots(figsize=figsize)
        # Create initial plot with proper scaling
        self.im = self.ax.imshow(current_field, origin='lower',
                                extent=(0, self.design.width, 0, self.design.height),
                                cmap='RdBu', aspect='equal', interpolation='bicubic', vmin=axis_scale[0], vmax=axis_scale[1])
        # Add colorbar
        colorbar = plt.colorbar(self.im, orientation='vertical', aspect=30, extend='both')
        colorbar.set_label(f'{field}{slice_info} Field Amplitude')
        # Add design structure outlines
        for structure in self.design.structures:
            # Use dashed lines for PML regions
            if hasattr(structure, 'is_pml') and structure.is_pml:
                structure.add_to_plot(self.ax, edgecolor="black", linestyle='--', facecolor='none', alpha=0.5)
            else:
                structure.add_to_plot(self.ax, facecolor="none")
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

    def _update_live_animation(self):
        """Update live animation if requested."""
        if self._live and (self.current_step % 2 == 0 or self.current_step == self.num_steps - 1):
            # Prefer showing Ez magnitude for better visibility, normalize color scale
            field = "Ez" if not self.is_3d else "Ez"
            data = getattr(self, field)
            Ez_np = self.backend.to_numpy(data)
            self.animate_live(field_data=Ez_np, field=field, axis_scale=self._axis_scale)

    def _record_monitor_data(self, step):
        """Record field data at monitor locations"""
        # Skip if no monitors
        if not self.design.monitors: return
            
        # Convert field data to numpy for monitors
        if self.is_3d:
            # 3D simulation
            Ex_np = self.backend.to_numpy(self.Ex)
            Ey_np = self.backend.to_numpy(self.Ey)
            Ez_np = self.backend.to_numpy(self.Ez)
            Hx_np = self.backend.to_numpy(self.Hx)
            Hy_np = self.backend.to_numpy(self.Hy)
            Hz_np = self.backend.to_numpy(self.Hz)
            # Record data for each monitor
            for monitor in self.design.monitors:
                if hasattr(monitor, 'record_fields') and callable(monitor.record_fields):
                    monitor.record_fields(
                        Ex_np, Ey_np, Ez_np, Hx_np, Hy_np, Hz_np,
                        self.t, self.dx, self.dy, self.dz, step=step)
        else:
            # 2D simulation
            Ez_np = self.backend.to_numpy(self.Ez)
            Hx_np = self.backend.to_numpy(self.Hx)
            Hy_np = self.backend.to_numpy(self.Hy)
            # Record data for each monitor
            for monitor in self.design.monitors:
                if hasattr(monitor, 'record_fields') and callable(monitor.record_fields):
                    # Force monitor to use 2D mode for 2D simulations
                    if hasattr(monitor, 'is_3d'):
                        original_is_3d = monitor.is_3d
                        monitor.is_3d = False
                    monitor.record_fields(
                        Ez_np, Hx_np, Hy_np,
                        self.t, self.dx, self.dy, step=step)
                    # Restore original is_3d setting
                    if hasattr(monitor, 'is_3d'):
                        monitor.is_3d = original_is_3d

    def save_animation(self, field: str = "Ez", axis_scale=[-1, 1], filename='fdtd_animation.mp4', 
                       fps=60, frame_skip=4, clean_visualization=False):
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
            fig, ax = plt.subplots(figsize=figsize)
            max_dim = max(self.design.width, self.design.height)
            scale, unit = get_si_scale_and_label(max_dim)

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
            # Use dashed lines for PML regions
            if hasattr(structure, 'is_pml') and structure.is_pml:
                structure.add_to_plot(ax, edgecolor="black", linestyle='--', facecolor='none', alpha=0.5)
            else: structure.add_to_plot(ax, facecolor="none", edgecolor="black")
        # Configure standard plot elements if not using clean visualization
        if not clean_visualization:
            # Add axis labels with proper scaling
            plt.xlabel(f'X ({unit})')
            plt.ylabel(f'Y ({unit})')
            ax.xaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
            ax.yaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
            # Title with time information
            title = ax.set_title(f't = {self.results["t"][0]:.2e} s')
        else: title = None
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
            if clean_visualization: ani.save(filename, writer=writer, dpi=300)
            else: ani.save(filename, writer=writer, dpi=100)
            print(f"Animation saved to {filename} (using {frames} of {total_frames} frames)")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Make sure FFmpeg is installed on your system.")
        # Close the figure
        plt.close(fig)
        
    def plot_power(self, cmap: str = "hot", vmin: float = None, vmax: float = None, db_colorbar: bool = False):
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
        # Check if we have accumulated power
        if self.power_accumulated is not None:
            # Use pre-accumulated power
            power = self.power_accumulated
            print("Using accumulated power data")
        elif len(self.results['Ez']) > 0 and len(self.results['Hx']) > 0 and len(self.results['Hy']) > 0:
            # Calculate power from saved field data
            print("Calculating power from saved field data")
            # Initialize power array
            power = np.zeros((self.nx, self.ny))
            # Calculate average power over all time steps
            for t_idx in range(len(self.results['t'])):
                Ez = self.results['Ez'][t_idx]
                Hx_raw = self.results['Hx'][t_idx]
                Hy_raw = self.results['Hy'][t_idx]
                # Check if any field is complex
                is_complex = np.iscomplexobj(Ez) or np.iscomplexobj(Hx_raw) or np.iscomplexobj(Hy_raw)
                # Handle complex Ez data
                if np.iscomplexobj(Ez):
                    Ez_real = np.real(Ez)
                    Ez_imag = np.imag(Ez)
                else:
                    Ez_real = Ez
                    Ez_imag = np.zeros_like(Ez)
                # Extend magnetic fields to match Ez dimensions
                if is_complex:
                    Hx = np.zeros_like(Ez, dtype=np.complex128)
                    Hy = np.zeros_like(Ez, dtype=np.complex128)
                else:
                    Hx = np.zeros_like(Ez_real)
                    Hy = np.zeros_like(Ez_real)
                # Properly handle filling the arrays
                if np.iscomplexobj(Hx_raw):
                    Hx[:, :-1] = Hx_raw
                else:
                    Hx[:, :-1] = Hx_raw
                if np.iscomplexobj(Hy_raw):
                    Hy[:-1, :] = Hy_raw
                else:
                    Hy[:-1, :] = Hy_raw
                # For complex fields, use proper calculation
                if is_complex:
                    Hx_real = np.real(Hx)
                    Hx_imag = np.imag(Hx)
                    Hy_real = np.real(Hy)
                    Hy_imag = np.imag(Hy)
                    # Calculate Poynting vector components for real and imaginary parts
                    # Using formula for complex Poynting vector: S = (1/2) Re[E × H*]
                    Sx = -Ez_real * Hy_real - Ez_imag * Hy_imag
                    Sy = Ez_real * Hx_real + Ez_imag * Hx_imag
                else:
                    # Real-only fields
                    Sx = -Ez_real * Hy
                    Sy = Ez_real * Hx
                # Calculate power magnitude (|S|²)
                power_mag = Sx**2 + Sy**2
                # Accumulate power
                power += power_mag
            # Average power over time steps
            power /= len(self.results['t'])
        else:
            print("No field data to calculate power. Make sure to run the simulation with save=True or accumulate_power=True.")
            return
        # Get SI scale and label
        scale, unit = get_si_scale_and_label(max(self.design.width, self.design.height))
        # Configure plot size
        aspect_ratio = power.shape[1] / power.shape[0]
        base_size = 8  # Base size for the smaller dimension
        if aspect_ratio > 1: figsize = (base_size * aspect_ratio, base_size)
        else: figsize = (base_size, base_size / aspect_ratio)
        # Apply logarithmic scaling if requested
        max_power = np.max(power)
        if max_power <= 0:
            print("Warning: Maximum power is zero or negative. Cannot plot logarithmic scale.")
            log_scale = False

        # Create the figure and axis
        self.fig, self.ax = plt.subplots(figsize=figsize)
        # Create initial plot with proper scaling
        self.im = self.ax.imshow(power, origin='lower',
                                extent=(0, self.design.width, 0, self.design.height),
                                cmap=cmap, aspect='equal', interpolation='bicubic', vmin=vmin, vmax=vmax)
        # Add colorbar
        colorbar = plt.colorbar(self.im, orientation='vertical', aspect=30, extend='both')
        # Convert to dB scale for colorbar if requested
        if db_colorbar:
            # Define a formatter function to convert linear values to dB
            def db_formatter(x, pos):
                if x <= 0: return "-∞ dB"  # Return negative infinity for zero/negative values
                ratio = max(x / max_power, 1e-10)  # Ensure minimum value is 1e-10 (-100 dB)
                db_val = 10 * np.log10(ratio)
                return f"{db_val:.1f} dB"
            colorbar.formatter = plt.FuncFormatter(db_formatter)
            # Set specific dB ticks
            colorbar.update_ticks() 
            colorbar.set_label('Relative Power (dB)')
        else: colorbar.set_label('Power (a.u.)')
        # Add design structure outlines
        for structure in self.design.structures:
            # Use dashed lines for PML regions
            if hasattr(structure, 'is_pml') and structure.is_pml:
                structure.add_to_plot(self.ax, edgecolor="white", linestyle='--', facecolor='none', alpha=0.5)
            else:
                structure.add_to_plot(self.ax, facecolor="none", edgecolor="white")
        # Add axis labels with proper scaling
        plt.xlabel(f'X ({unit})')
        plt.ylabel(f'Y ({unit})')
        self.ax.xaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
        self.ax.yaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
        # Add plot elements
        plt.title('Time-Averaged Power Distribution')
        plt.tight_layout()
        plt.show()

    def estimate_memory_usage(self, time_steps=None, save_fields=None):
        """Estimate memory usage of the simulation with current settings.
        
        Args:
            time_steps: Number of time steps to estimate for (defaults to self.num_steps)
            save_fields: List of fields to save (defaults to appropriate fields for dimensionality)
            
        Returns:
            Dictionary with memory usage estimates in MB
        """
        if time_steps is None: time_steps = self.num_steps
        if save_fields is None:
            if self.is_3d: save_fields = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
            else: save_fields = ['Ez', 'Hx', 'Hy']
        # Calculate size of arrays
        bytes_per_value = np.float64(0).nbytes  # Usually 8 bytes for float64
        # Calculate individual field sizes based on dimensionality
        field_sizes = {}
        if self.is_3d:
            # 3D field array sizes
            field_sizes['Ex'] = (self.nz * self.ny * (self.nx-1)) * bytes_per_value
            field_sizes['Ey'] = (self.nz * (self.ny-1) * self.nx) * bytes_per_value
            field_sizes['Ez'] = ((self.nz-1) * self.ny * self.nx) * bytes_per_value
            field_sizes['Hx'] = ((self.nz-1) * (self.ny-1) * self.nx) * bytes_per_value
            field_sizes['Hy'] = ((self.nz-1) * self.ny * (self.nx-1)) * bytes_per_value
            field_sizes['Hz'] = (self.nz * (self.ny-1) * (self.nx-1)) * bytes_per_value
        else:
            # 2D field array sizes
            field_sizes['Ez'] = self.nx * self.ny * bytes_per_value
            field_sizes['Hx'] = self.nx * (self.ny-1) * bytes_per_value
            field_sizes['Hy'] = (self.nx-1) * self.ny * bytes_per_value
        
        t_size = time_steps * bytes_per_value
        
        # Calculate total memory for all time steps
        total_size = t_size
        single_step_size = 0
        for field in save_fields:
            if field in field_sizes:
                field_size = field_sizes[field]
                total_size += field_size * time_steps
                single_step_size += field_size
        
        # Convert to more readable units
        kb = 1024
        mb = kb * 1024
        gb = mb * 1024
        
        result = {
            'Single timestep': {
                **{field: field_sizes.get(field, 0) / mb for field in save_fields},
                'Total': single_step_size / mb
            },
            'Full simulation': {
                'Total memory (MB)': total_size / mb,
                'Total memory (GB)': total_size / gb,
                'Time steps': time_steps,
                'Grid size': f"{self.nx} x {self.ny}" + (f" x {self.nz}" if self.is_3d else ""),
                'Fields saved': ', '.join(save_fields),
                'Dimensionality': '3D' if self.is_3d else '2D'
            }
        }
        
        # Display results
        display_status(f"Estimated memory usage: {total_size / mb:.2f} MB", "info")
        return result