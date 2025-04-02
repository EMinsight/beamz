from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import numpy as np

class StandardGrid:
    """A standard uniform grid for FDTD simulations."""
    
    def __init__(self, cell_size: float = 1.0):
        """Initialize a standard grid.
        
        Args:
            cell_size (float): Size of each grid cell
        """
        self.cell_size = cell_size

class Simulation:
    def __init__(self, name: str, type: str = "2D", size: Tuple[int, ...] = (100, 100), 
                 grid: StandardGrid = None, structures: List[Dict] = None,
                 sources: List[Dict] = None, monitors: List[Dict] = None,
                 device: str = "cpu"):
        """Initialize a simulation.
        
        Args:
            name (str): Name of the simulation
            type (str): Simulation type ("2D" or "3D")
            size (tuple): Grid size (nx, ny) or (nx, ny, nz)
            grid (StandardGrid): Grid configuration
            structures (List[Dict]): List of structures in the simulation
            sources (List[Dict]): List of sources
            monitors (List[Dict]): List of monitors
            device (str): Device to run simulation on ("cpu" or "cuda")
        """
        self.name = name
        self.type = type
        self.size = size
        self.grid = grid or StandardGrid()
        self.cell_size = self.grid.cell_size
        self.structures = structures or []
        self.sources = sources or []
        self.monitors = monitors or []
        self.device = device
        
        # Physical constants
        self.c0 = 3e8  # Speed of light in vacuum
        self.epsilon_0 = 8.85e-12  # Vacuum permittivity
        self.mu_0 = 1.256e-6  # Vacuum permeability
        
        # Grid parameters
        self.nx, self.ny = size
        self.dx = self.dy = self.cell_size
        
        # Initialize fields
        self.Ez = np.zeros((self.nx, self.ny))
        self.Hx = np.zeros((self.nx, self.ny-1))
        self.Hy = np.zeros((self.nx-1, self.ny))
        
        # Material properties (default: vacuum)
        self.epsilon_r = np.ones((self.nx, self.ny))
        
        # Add conductivity array for PML
        self.sigma = np.zeros((self.nx, self.ny))
        
        # Add PML field components
        self.Ezx = np.zeros((self.nx, self.ny))
        self.Ezy = np.zeros((self.nx, self.ny))
        
        # Time parameters
        self.t = 0
        self.dt = self.cell_size / (self.c0 * np.sqrt(2))  # CFL condition
        self.time = 0
        self.num_steps = int(self.time / self.dt)  # Initialize num_steps
        
        # Results storage
        self.results = {
            'Ez': [],
            'Hx': [],
            'Hy': [],
            't': []
        }
        
        # Initialize simulation components
        self.materials: Dict[str, Dict] = {}
        self.boundaries: List[Dict] = []
        
        # Version
        self.__version__ = "0.1.0"
        
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
        
    def set_pml(self, sigma):
        """Set the PML conductivity profile"""
        self.sigma = sigma
    
    def summary(self):
        """Print a summary of the simulation parameters"""
        pass

    def run(self, steps: Optional[int] = None, save=True, animate_live=True) -> Dict:
        """Run the simulation.
        
        Args:
            steps (int, optional): Number of steps to run. If None, runs for full duration.
            save (bool): Whether to save field data
            animate_live (bool): Whether to show live animation
            
        Returns:
            Dict: Simulation results
        """
        if steps is None:
            steps = self.num_steps
            
        for step in range(steps):
            # Update fields
            self.update_h_fields()
            self.update_e_field()
            
            # Apply sources
            for source in self.sources:
                x, y = source.position
                if 0 <= x < self.nx and 0 <= y < self.ny:
                    # Get wave amplitude at current time
                    amplitude = source.signal.get_amplitude(self.t)
                    # Apply source with direction
                    self.Ez[x, y] += amplitude
            
            # Save results if requested
            if save and step % 10 == 0:  # Save every 10th step
                self.results['Ez'].append(self.Ez.copy())
                self.results['Hx'].append(self.Hx.copy())
                self.results['Hy'].append(self.Hy.copy())
                self.results['t'].append(self.t)
            
            # Update time
            self.t += self.dt
            
            # Show progress
            if step % 100 == 0:
                print(f"Step {step}/{steps}")
        
        return self.results
    
    def plot_field(self, field: str = "Ez", t: float = None) -> None:
        """Plot a field at a given time.
        
        Args:
            field (str): Field to plot ("Ez", "Hx", or "Hy")
            t (float, optional): Time to plot. If None, plots last saved time.
        """
        import matplotlib.pyplot as plt
        
        if t is None:
            t = self.results['t'][-1]
            
        # Find closest time step
        t_idx = np.argmin(np.abs(np.array(self.results['t']) - t))
        
        plt.figure(figsize=(10, 8))
        plt.imshow(self.results[field][t_idx], cmap='RdBu')
        plt.colorbar(label=field)
        plt.title(f"{field} field at t = {self.results['t'][t_idx]:.2e} s")
        plt.show()

    def to_dict(self) -> Dict:
        """Convert simulation configuration to a dictionary."""
        return {
            'type': self.type,
            'size': self.size,
            'cell_size': self.cell_size,
            'dt': self.dt,
            'time': self.time,
            'num_steps': self.num_steps,
            'materials': self.materials,
            'sources': self.sources,
            'boundaries': self.boundaries,
            'timestamp': datetime.now().isoformat(),
            'version': self.__version__
        }
    
    def save_config(self, filepath: str) -> None:
        """Save simulation configuration to a JSON file."""
        config = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
    
    @classmethod
    def load_config(cls, filepath: str) -> 'Simulation':
        """Load simulation configuration from a JSON file."""
        pass
