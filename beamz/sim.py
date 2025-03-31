"""
Simulation module for BeamZ.
"""

class Simulation:
    """Base simulation class."""
    
    def __init__(self, type="2D", size=(100, 100), cell_size=0.1, dt=0.1, time=1.0):
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
        
        # Initialize field arrays
        if type == "2D":
            nx, ny = size
            self.Ez = None  # Will be initialized in subclasses
            self.Hx = None  # Will be initialized in subclasses
            self.Hy = None  # Will be initialized in subclasses
        else:
            raise ValueError("Only 2D simulations are supported at this time.")
    

    def run(self, steps=None):
        """Run the simulation.
        
        Args:
            steps (int, optional): Number of steps to run. If None, runs for full duration.
        """
        if steps is None:
            steps = self.num_steps
        # Implementation will be in subclasses
        raise NotImplementedError("Run method must be implemented in subclasses")
