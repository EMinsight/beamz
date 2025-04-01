"""
Simulation module for BeamZ.
"""

from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import numpy as np

class Simulation:
    """Base simulation class."""
    
    __version__ = "0.1.0"
    
    def __init__(self, type: str = "2D", size: Tuple[int, ...] = (100, 100), 
                 cell_size: float = 0.1, dt: float = 0.1, time: float = 1.0, device="cpu"):
        """Initialize a simulation.
        
        Args:
            type (str): Simulation type ("2D" or "3D")
            size (tuple): Grid size (nx, ny) or (nx, ny, nz)
            cell_size (float): Size of each grid cell
            dt (float): Time step
            time (float): Total simulation time
        """
        self.type = type
        self.size = size
        self.cell_size = cell_size
        self.dt = dt
        self.time = time
        self.num_steps = int(time / dt)
        self.device = device
        
        # Physical constants
        self.c0 = 3e8  # Speed of light in vacuum
        self.epsilon_0 = 8.85e-12  # Vacuum permittivity
        self.mu_0 = 1.256e-6  # Vacuum permeability
        
        # Grid parameters
        self.nx, self.ny = size
        self.dx, self.dy = cell_size, cell_size
        
        # Initialize fields
        self.Ez = np.zeros((self.nx, self.ny))
        self.Hx = np.zeros((self.nx, self.ny-1))
        self.Hy = np.zeros((self.nx-1, self.ny))
        
        # Material properties (default: vacuum)
        self.epsilon_r = np.ones((self.nx, self.ny))
        
        # Source parameters
        self.t = 0
        self.source_x = self.nx // 2
        self.source_y = self.ny // 2
        
        # Add conductivity array for PML
        self.sigma = np.zeros((self.nx, self.ny))
        
        # Add PML field components
        self.Ezx = np.zeros((self.nx, self.ny))
        self.Ezy = np.zeros((self.nx, self.ny))
        
        # Initialize simulation components
        self.materials: Dict[str, Dict] = {}
        self.sources: List[Dict] = []
        self.boundaries: List[Dict] = []
    
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
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        # Create simulation with basic parameters
        sim = cls(
            type=config['type'],
            size=config['size'],
            cell_size=config['cell_size'],
            dt=config['dt'],
            time=config['time']
        )
        
        # Add materials
        for name, props in config['materials'].items():
            sim.add_material(name, **props)
        
        # Add sources
        for source in config['sources']:
            sim.add_source(**source)
        
        # Add boundaries
        for boundary in config['boundaries']:
            sim.add_boundary(**boundary)
        
        return sim
    
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

    def run(self, steps: Optional[int] = None) -> None:
        """Run the simulation.
        
        Args:
            steps (int, optional): Number of steps to run. If None, runs for full duration.
        """
        if steps is None:
            steps = self.num_steps
        # Implementation will be in subclasses
        raise NotImplementedError("Run method must be implemented in subclasses")
