from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import numpy as np
import h5py
import os
from .sources import PointSource, Wave
from beamz.simulation.const import *

class StandardGrid:
    def __init__(self, cell_size: float = 1.0):
        self.cell_size = cell_size

class Simulation:
    def __init__(self,
                 name: str = None,
                 type: str = "2D",
                 size: Tuple[int, ...] = (100, 100), 
                 grid: StandardGrid = None,
                 boundaries: List[Dict] = None,
                 structures: List[Dict] = None,
                 sources: List[Dict] = None,
                 monitors: List[Dict] = None,
                 time: float = None,
                 dt: float = None,
                 device: str = "cpu",
                 animate_live: bool = False):
        """Initialize a simulation."""
        self.name = name
        self.type = type
        self.size = size
        self.grid = grid or StandardGrid()
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
        # Material properties (default: vacuum)
        self.epsilon_r = np.ones((self.nx, self.ny))
        # Add conductivity array (e.g. for PML)
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
        """Save simulation results to an HDF5 file."""
        # Define the filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{timestamp}.h5"
        # Define the filepath
        filepath = os.path.join(self.results_dir, filename)
        # Create the file
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
                src.attrs['amplitude'] = source.signal.amplitude
                src.attrs['frequency'] = source.signal.frequency
                src.attrs['ramp_up_time'] = source.signal.ramp_up_time
                src.attrs['ramp_down_time'] = source.signal.ramp_down_time
            # Save boundaries configuration
            if self.boundaries is not None:
                boundaries = f.create_group('boundaries')
                if self.boundaries.top is not None:
                    top = boundaries.create_group('top')
                    top.attrs['thickness'] = self.boundaries.top.thickness
                    top.attrs['sigma_max'] = self.boundaries.top.sigma_max
                    top.attrs['m'] = self.boundaries.top.m
                if self.boundaries.bottom is not None:
                    bottom = boundaries.create_group('bottom')
                    bottom.attrs['thickness'] = self.boundaries.bottom.thickness
                    bottom.attrs['sigma_max'] = self.boundaries.bottom.sigma_max
                    bottom.attrs['m'] = self.boundaries.bottom.m
                if self.boundaries.left is not None:
                    left = boundaries.create_group('left')
                    left.attrs['thickness'] = self.boundaries.left.thickness
                    left.attrs['sigma_max'] = self.boundaries.left.sigma_max
                    left.attrs['m'] = self.boundaries.left.m
                if self.boundaries.right is not None:
                    right = boundaries.create_group('right')
                    right.attrs['thickness'] = self.boundaries.right.thickness
                    right.attrs['sigma_max'] = self.boundaries.right.sigma_max
                    right.attrs['m'] = self.boundaries.right.m
        print(f"Results saved to {filepath}")
        return filepath

    def _animate_step(self, frame):
        """Perform one simulation step and update the animation."""
        # Update fields
        self.update_h_fields()
        self.update_e_field()
        # Apply sources
        for source in self.sources:
            x, y = source.position
            if 0 <= x < self.nx and 0 <= y < self.ny:
                amplitude = source.signal.get_amplitude(self.t)
                self.Ez[x, y] += amplitude
        # Save results if requested
        if self._save_results and frame % 10 == 0:
            self.results['Ez'].append(self.Ez.copy())
            self.results['Hx'].append(self.Hx.copy())
            self.results['Hy'].append(self.Hy.copy())
            self.results['t'].append(self.t)
        # Update time
        self.t += self.dt
        # Update the image
        self._animation_im.set_array(self.Ez)
        self._animation_im.set_clim(self.Ez.min(), self.Ez.max())
        # Print progress
        if frame % 100 == 0:
            print(f"Step {frame}/{self._total_steps}")
        return [self._animation_im]

    def run(self, steps: Optional[int] = None, save=True, animate_live=True) -> Dict:
        """Run the simulation."""
        if steps is None:
            steps = self.num_steps
        # Initialize simulation state
        self.t = 0
        self._total_steps = steps
        self._save_results = save
        # Initialize results storage
        if save:
            self.results = {
                'Ez': [],
                'Hx': [],
                'Hy': [],
                't': []
            }
            
        if animate_live:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            print("Animating live...")
            # Ensure Ez is initialized
            assert self.Ez is not None and self.Ez.size > 0, "Error: self.Ez must be initialized"
            # Create figure and axes
            fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
            # Create initial plot
            self._animation_im = ax.imshow(self.Ez, cmap='RdBu', aspect='equal',
                                         interpolation='bicubic', animated=True)
            # Add colorbar
            colorbar = plt.colorbar(self._animation_im, ax=ax, label='Ez Field (V/m)')
            # Set title and layout
            ax.set_title('Electric Field (Ez)')
            plt.tight_layout()
            # Create animation with explicit list of frames
            frames = list(range(steps))  # Ensure steps > 0
            anim = FuncAnimation(fig, self._animate_step, frames=frames,
                               interval=1, blit=True, repeat=False)
            # Show the animation
            plt.show()
            # After animation is closed, save results if requested
            if save: self.save_results()
            return self.results
            
        # If not animating, run normally
        for step in range(steps):
            # Update fields
            self.update_h_fields()
            self.update_e_field()
            # Apply sources
            for source in self.sources:
                x, y = source.position
                if 0 <= x < self.nx and 0 <= y < self.ny:
                    amplitude = source.signal.get_amplitude(self.t)
                    self.Ez[x, y] += amplitude
            # Save results if requested
            if save and step % 10 == 0:
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
        if save: self.save_results()
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
        """Load simulation results from an HDF5 file."""
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

class PML:
    """Perfectly Matched Layer configuration for absorbing boundary conditions."""
    def __init__(self, thickness: int = 10, sigma_max: float = 1.0, m: float = 3.0):
        """Initialize PML parameters."""
        self.thickness = thickness
        self.sigma_max = sigma_max
        self.m = m
        self.sigma = None  # Will be set when grid size is known
        
    def _setup(self, nx: int, ny: int) -> None:
        """Set up the PML conductivity profile for a grid of size (nx, ny)."""
        # Initialize PML conductivity array
        self.sigma = np.zeros((nx, ny))
        # Create PML profile
        for i in range(self.thickness):
            # Calculate normalized distance from boundary (0 to 1)
            x = (self.thickness - i) / self.thickness
            # Calculate conductivity using polynomial grading
            sigma_value = self.sigma_max * (x ** self.m)
            # Apply to all four boundaries
            # Left and right boundaries
            self.sigma[i, :] = sigma_value
            self.sigma[-(i+1), :] = sigma_value
            # Top and bottom boundaries
            self.sigma[:, i] = sigma_value
            self.sigma[:, -(i+1)] = sigma_value
            # Corners (use maximum of both directions)
            self.sigma[i, i] = max(self.sigma[i, i], sigma_value)
            self.sigma[i, -(i+1)] = max(self.sigma[i, -(i+1)], sigma_value)
            self.sigma[-(i+1), i] = max(self.sigma[-(i+1), i], sigma_value)
            self.sigma[-(i+1), -(i+1)] = max(self.sigma[-(i+1), -(i+1)], sigma_value)

class Boundaries:
    """Container for PML boundary conditions on each side of the simulation domain."""
    def __init__(self, top: 'PML' = None, bottom: 'PML' = None, 
                 left: 'PML' = None, right: 'PML' = None, all: 'PML' = None):
        """Initialize boundary conditions."""
        if all is not None:
            self.top = self.bottom = self.left = self.right = all
        else:
            self.top = top
            self.bottom = bottom
            self.left = left
            self.right = right
        self.sigma = None  # Will be set when grid size is known
        
    def _setup(self, nx: int, ny: int) -> None:
        """Set up the combined PML conductivity profile for all boundaries."""
        # Initialize combined conductivity array
        self.sigma = np.zeros((nx, ny))
        # Apply PML from each boundary
        if self.top is not None:
            self.top._setup(nx, ny)
            self.sigma[:, :self.top.thickness] = self.top.sigma[:, :self.top.thickness]
        if self.bottom is not None:
            self.bottom._setup(nx, ny)
            self.sigma[:, -self.bottom.thickness:] = self.bottom.sigma[:, -self.bottom.thickness:]
        if self.left is not None:
            self.left._setup(nx, ny)
            self.sigma[:self.left.thickness, :] = self.left.sigma[:self.left.thickness, :]
        if self.right is not None:
            self.right._setup(nx, ny)
            self.sigma[-self.right.thickness:, :] = self.right.sigma[-self.right.thickness:, :]
