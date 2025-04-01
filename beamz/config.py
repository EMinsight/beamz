import json
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

class SimulationConfig:
    """Handles saving and loading simulation configurations."""
    
    def __init__(self, simulation):
        """Initialize with a simulation instance.
        
        Args:
            simulation: The simulation instance to save/load
        """
        self.simulation = simulation
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert simulation configuration to a dictionary.
        
        Returns:
            Dict containing all simulation parameters
        """
        config = {
            'type': self.simulation.type,
            'size': self.simulation.size,
            'cell_size': self.simulation.cell_size,
            'dt': self.simulation.dt,
            'time': self.simulation.time,
            'num_steps': self.simulation.num_steps,
            'materials': self.simulation.materials,
            'sources': self.simulation.sources,
            'boundaries': self.simulation.boundaries,
            'timestamp': datetime.now().isoformat(),
            'version': self.simulation.__version__
        }
        return config
        
    def save(self, filepath: str) -> None:
        """Save simulation configuration to a JSON file.
        
        Args:
            filepath: Path to save the configuration file
        """
        config = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
            
    @classmethod
    def load(cls, filepath: str) -> Dict[str, Any]:
        """Load simulation configuration from a JSON file.
        
        Args:
            filepath: Path to the configuration file
            
        Returns:
            Dict containing the loaded configuration
        """
        with open(filepath, 'r') as f:
            config = json.load(f)
        return config
        
    @classmethod
    def create_simulation(cls, config: Dict[str, Any]) -> 'Simulation':
        """Create a new simulation instance from a configuration.
        
        Args:
            config: Dictionary containing simulation parameters
            
        Returns:
            New simulation instance with the loaded configuration
        """
        from .sim import Simulation  # Import here to avoid circular imports
        
        # Create simulation with basic parameters
        sim = Simulation(
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