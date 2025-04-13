from typing import Dict, Optional
import numpy as np
from beamz.const import *
from beamz.simulation.meshing import RegularGrid
from beamz.design.sources import ModeSource

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
        import matplotlib.pyplot as plt
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
        
        print(f"Field range: min = {np.min(current_field):.2e}, max = {np.max(current_field):.2e}")
        
        # Determine appropriate SI unit and scale for spatial dimensions
        max_dim = max(self.design.width, self.design.height)
        if max_dim >= 1e-3: scale, unit = 1e3, 'mm'
        elif max_dim >= 1e-6: scale, unit = 1e6, 'µm'
        elif max_dim >= 1e-9: scale, unit = 1e9, 'nm'
        else: scale, unit = 1e12, 'pm'
        # Calculate figure size based on grid dimensions
        grid_height, grid_width = current_field.shape
        aspect_ratio = grid_width / grid_height
        base_size = 2.5  # Base size for the smaller dimension
        if aspect_ratio > 1: figsize = (base_size * aspect_ratio, base_size)
        else: figsize = (base_size, base_size / aspect_ratio)
        # Create the figure
        plt.figure(figsize=figsize)
        plt.imshow(current_field, origin='lower', 
                  extent=(0, self.design.width, 0, self.design.height),
                  cmap='RdBu', aspect='equal')
        plt.colorbar(label=f'{field} Field Amplitude')
        plt.title(f'{field} Field at t = {current_t:.2e} s')
        plt.xlabel(f'X ({unit})')
        plt.ylabel(f'Y ({unit})')
        # Update tick labels with scaled values
        plt.gca().xaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
        plt.gca().yaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
        plt.tight_layout()
        plt.show()

    def run(self, steps: Optional[int] = None, save=True, animate_live=True) -> Dict:
        """Run the simulation."""
        # Initialize simulation state
        self.t = 0
        self._total_steps = self.num_steps
        self._save_results = save
        print(f"Starting simulation with {self.num_steps} steps")
        print(f"Initial field range: min = {np.min(self.Ez):.2e}, max = {np.max(self.Ez):.2e}")

        # Calculate Courant number to check stability
        c = 1/np.sqrt(EPS_0 * MU_0)  # Speed of light
        courant = c * self.dt / min(self.dx, self.dy)
        if courant > 1/np.sqrt(2):
            print(f"Warning: Simulation may be unstable! Courant number = {courant:.3f} > {1/np.sqrt(2):.3f}")
            print(f"Consider reducing dt or increasing dx/dy")

        # If not animating, run normally
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
                        self.Ez[y, x] += amplitude * modulation  # Note: numpy uses [row, col] = [y, x]
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
            
            # Show progress and check for divergence
            if step % 10 == 0:
                max_field = np.max(np.abs(self.Ez))
                print(f"Step {step}/{self.num_steps}")
                print(f"Current field range: min = {np.min(self.Ez):.2e}, max = {np.max(self.Ez):.2e}")
                
                # Check for divergence
                if max_field > 1e10:  # Arbitrary threshold, adjust as needed
                    print("Warning: Field values are diverging! Stopping simulation.")
                    break
                    
                self.plot_field(field="Ez", t=self.t)

        print(f"Simulation completed. Final field range: min = {np.min(self.Ez):.2e}, max = {np.max(self.Ez):.2e}")
        return self.results