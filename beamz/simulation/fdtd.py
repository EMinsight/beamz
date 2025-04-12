from typing import Dict, Optional
import numpy as np
from beamz.const import *
from beamz.simulation.meshing import RegularGrid

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
        self.Ez = np.zeros(self.mesh.shape)
        self.Hx = np.zeros((self.mesh.shape[0], self.mesh.shape[1]-1))
        self.Hy = np.zeros((self.mesh.shape[0]-1, self.mesh.shape[1]))
        # Initialize the time
        self.time = time
        self.dt = time[1] - time[0]
        self.num_steps = len(time)
        # Initialize the sources
        self.sources = self.design.sources

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

    def run(self, steps: Optional[int] = None, save=True, animate_live=True) -> Dict:
        """Run the simulation."""
        if steps is None:
            steps = self.num_steps
        # Initialize simulation state
        self.t = 0
        self._total_steps = steps
        self._save_results = save
            
        # If not animating, run normally
        for step in range(steps):
            # Update fields
            self.simulate_step()
            # Apply sources
            for source in self.sources:
                x, y = source.position
                if 0 <= x < self.nx and 0 <= y < self.ny:
                    amplitude = source.signal.get_amplitude(self.t)
                    self.Ez[x, y] += amplitude
            # Update time
            self.t += self.dt
            # Show progress
            if step % 100 == 0:
                print(f"Step {step}/{steps}")

        return self.results
    
    def plot_field(self, field: str = "Ez", t: float = None) -> None:
        """Plot a field at a given time."""
        import matplotlib.pyplot as plt
        if t is None: t = self.results['t'][-1]
        # Find closest time step
        t_idx = np.argmin(np.abs(np.array(self.results['t']) - t)) 
        plt.figure(figsize=(10, 8))
        plt.imshow(self.results[field][t_idx], cmap='RdBu')
        plt.colorbar(label=field)
        plt.title(f"{field} field at t = {self.results['t'][t_idx]:.2e} s")
        plt.show()