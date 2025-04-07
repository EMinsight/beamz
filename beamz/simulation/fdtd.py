from typing import Dict, List, Tuple, Optional
import numpy as np
from beamz.const import *
from beamz.simulation.meshing import RegularGrid

class Simulation:
    def __init__(self, name: str = None, type: str = "2D", size: Tuple[int, ...] = (100, 100), 
                 grid: RegularGrid = None, boundaries: List[Dict] = None,
                 structures: List[Dict] = None, sources: List[Dict] = None,
                 monitors: List[Dict] = None, device: str = "cpu"):
        self.name = name
        self.type = type
        self.size = size
        self.grid = grid
        self.cell_size = self.grid.cell_size
        self.structures = structures or []
        self.boundaries = boundaries or []
        self.sources = sources or []
        self.monitors = monitors or []
        self.device = device
        # Physical constants
        self.c0 = LIGHT_SPEED  # Speed of light in vacuum
        self.epsilon_0 = VAC_PERMITTIVITY  # Vacuum permittivity
        self.mu_0 = VAC_PERMEABILITY  # Vacuum permeability
        # Grid parameters
        self.nx, self.ny = size
        self.dx = self.dy = self.cell_size
        # Initialize fields
        self.Ez = np.zeros((self.nx, self.ny))
        self.Hx = np.zeros((self.nx, self.ny-1))
        self.Hy = np.zeros((self.nx-1, self.ny))
        # Time parameters
        self.t = 0
        self.dt = self.cell_size / (self.c0 * np.sqrt(2))  # CFL condition
        self.time = 0
        self.num_steps = int(self.time / self.dt)  # Initialize num_steps
        
    def update_h_fields(self):
        """Update magnetic field components with PML"""
        self.Hx[:, :] = self.Hx[:, :] - (self.dt/(self.mu_0*self.dy)) * \
                        (self.Ez[:, 1:] - self.Ez[:, :-1])
        self.Hy[:, :] = self.Hy[:, :] + (self.dt/(self.mu_0*self.dx)) * \
                        (self.Ez[1:, :] - self.Ez[:-1, :])
    
    def update_e_field(self):
        """Update electric field component with PML"""
        # First update the main field without PML
        self.Ez[1:-1, 1:-1] = self.Ez[1:-1, 1:-1] + \
            (self.dt/(self.epsilon_0*self.epsilon_r[1:-1, 1:-1])) * \
            ((self.Hy[1:, 1:-1] - self.Hy[:-1, 1:-1])/self.dx - \
             (self.Hx[1:-1, 1:] - self.Hx[1:-1, :-1])/self.dy)
        # Then apply PML only at the boundaries where sigma > 0
        mask = self.sigma[1:-1, 1:-1] > 0
        if np.any(mask):
            sigma_x = self.sigma[1:-1, 1:-1][mask]
            sigma_y = self.sigma[1:-1, 1:-1][mask]
            # Update coefficients for PML regions only
            cx = np.exp(-sigma_x * self.dt / self.epsilon_0)
            cy = np.exp(-sigma_y * self.dt / self.epsilon_0)
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