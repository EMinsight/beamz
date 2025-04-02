from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import numpy as np
import h5py
import os
from .sources import PointSource, Wave

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
        
        # Create results directory if it doesn't exist
        self.results_dir = "simulation_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
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

    def save_results(self, filename: str = None) -> str:
        """Save simulation results to an HDF5 file.
        
        Args:
            filename (str, optional): Name of the file to save to. If None, generates a name.
            
        Returns:
            str: Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{timestamp}.h5"
        
        filepath = os.path.join(self.results_dir, filename)
        
        with h5py.File(filepath, 'w') as f:
            # Save metadata
            meta = f.create_group('metadata')
            meta.attrs['name'] = self.name
            meta.attrs['type'] = self.type
            meta.attrs['size'] = self.size
            meta.attrs['cell_size'] = self.cell_size
            meta.attrs['dt'] = self.dt
            meta.attrs['time'] = self.time
            meta.attrs['num_steps'] = self.num_steps
            meta.attrs['version'] = self.__version__
            meta.attrs['timestamp'] = datetime.now().isoformat()
            
            # Save grid parameters
            grid = f.create_group('grid')
            grid.attrs['nx'] = self.nx
            grid.attrs['ny'] = self.ny
            grid.attrs['dx'] = self.dx
            grid.attrs['dy'] = self.dy
            
            # Save physical constants
            constants = f.create_group('constants')
            constants.attrs['c0'] = self.c0
            constants.attrs['epsilon_0'] = self.epsilon_0
            constants.attrs['mu_0'] = self.mu_0
            
            # Save material properties
            materials = f.create_group('materials')
            for name, props in self.materials.items():
                mat = materials.create_group(name)
                for key, value in props.items():
                    mat.attrs[key] = value
            
            # Save field data
            fields = f.create_group('fields')
            for field_name, field_data in self.results.items():
                if field_name != 't':  # Time is stored separately
                    fields.create_dataset(field_name, data=np.array(field_data))
            
            # Save time points
            f.create_dataset('time', data=np.array(self.results['t']))
            
            # Save sources
            sources = f.create_group('sources')
            for i, source in enumerate(self.sources):
                src = sources.create_group(f'source_{i}')
                src.attrs['position'] = source.position
                src.attrs['direction'] = source.signal.direction
                src.attrs['amplitude'] = source.signal.amplitude
                src.attrs['frequency'] = source.signal.frequency
                src.attrs['ramp_up_time'] = source.signal.ramp_up_time
                src.attrs['ramp_down_time'] = source.signal.ramp_down_time
        
        print(f"Results saved to {filepath}")
        return filepath

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
        
        # Save results to file if requested
        if save:
            self.save_results()
        
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

    @classmethod
    def load_results(cls, filepath: str) -> 'Simulation':
        """Load simulation results from an HDF5 file.
        
        Args:
            filepath (str): Path to the HDF5 file
            
        Returns:
            Simulation: A new simulation instance with loaded results
        """
        with h5py.File(filepath, 'r') as f:
            # Load metadata
            meta = f['metadata']
            sim = cls(
                name=meta.attrs['name'],
                type=meta.attrs['type'],
                size=meta.attrs['size'],
                grid=StandardGrid(cell_size=meta.attrs['cell_size'])
            )
            
            # Load grid parameters
            grid = f['grid']
            sim.nx = grid.attrs['nx']
            sim.ny = grid.attrs['ny']
            sim.dx = grid.attrs['dx']
            sim.dy = grid.attrs['dy']
            
            # Load physical constants
            constants = f['constants']
            sim.c0 = constants.attrs['c0']
            sim.epsilon_0 = constants.attrs['epsilon_0']
            sim.mu_0 = constants.attrs['mu_0']
            
            # Load material properties
            materials = f['materials']
            for name in materials.keys():
                mat = materials[name]
                sim.materials[name] = {
                    key: mat.attrs[key] for key in mat.attrs.keys()
                }
            
            # Load field data
            fields = f['fields']
            for field_name in fields.keys():
                sim.results[field_name] = fields[field_name][:]
            
            # Load time points
            sim.results['t'] = f['time'][:]
            
            # Load sources
            sources = f['sources']
            for i in range(len(sources)):
                src = sources[f'source_{i}']
                source = PointSource(
                    position=src.attrs['position'],
                    signal=Wave(
                        direction=src.attrs['direction'],
                        amplitude=src.attrs['amplitude'],
                        frequency=src.attrs['frequency'],
                        ramp_up_time=src.attrs['ramp_up_time'],
                        ramp_down_time=src.attrs['ramp_down_time']
                    )
                )
                sim.sources.append(source)
        
        return sim

    def setup_pml(self, thickness: int = 10, sigma_max: float = 1.0, m: float = 3.0):
        """Set up a Perfectly Matched Layer (PML) at the boundaries.
        
        Args:
            thickness (int): Number of cells in the PML region
            sigma_max (float): Maximum conductivity value in the PML
            m (float): Polynomial grading order (typically 3-4)
        """
        # Initialize PML conductivity array
        self.sigma = np.zeros((self.nx, self.ny))
        
        # Create PML profile
        for i in range(thickness):
            # Calculate normalized distance from boundary (0 to 1)
            x = (thickness - i) / thickness
            
            # Calculate conductivity using polynomial grading
            sigma = sigma_max * (x ** m)
            
            # Apply to all four boundaries
            # Left and right boundaries
            self.sigma[i, :] = sigma
            self.sigma[-(i+1), :] = sigma
            
            # Top and bottom boundaries
            self.sigma[:, i] = sigma
            self.sigma[:, -(i+1)] = sigma
            
            # Corners (use maximum of both directions)
            self.sigma[i, i] = max(self.sigma[i, i], sigma)
            self.sigma[i, -(i+1)] = max(self.sigma[i, -(i+1)], sigma)
            self.sigma[-(i+1), i] = max(self.sigma[-(i+1), i], sigma)
            self.sigma[-(i+1), -(i+1)] = max(self.sigma[-(i+1), -(i+1)], sigma)
