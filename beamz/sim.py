import numpy as np
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self, type="2D", size=(100, 100), cell_size=0.1, dt=0.1, time=1.0, device="cpu"):
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

        # set the backend to use
        self.device = device
        # Physical constants
        self.c0 = 3e8  # Speed of light in vacuum
        self.epsilon_0 = 8.85e-12  # Vacuum permittivity
        self.mu_0 = 1.256e-6  # Vacuum permeability
        # Grid parameters
        self.nx, self.ny = size
        self.dx, self.dy = cell_size, cell_size
        # Time step (Courant condition)
        self.dt = 0.5 * min(self.dx, self.dy) / self.c0
        # Initialize fields
        self.Ez = np.zeros((self.nx, self.ny))
        self.Hx = np.zeros((self.nx, self.ny-1))
        self.Hy = np.zeros((nx-1, ny))
        # Material properties (default: vacuum)
        self.epsilon_r = np.ones((nx, ny))
        # Source parameters
        self.t = 0
        self.source_x = nx // 2
        self.source_y = ny // 2
        # Add conductivity array for PML
        self.sigma = np.zeros((nx, ny))
        # Add PML field components
        self.Ezx = np.zeros((nx, ny))  # x-component of Ez
        self.Ezy = np.zeros((nx, ny))  # y-component of Ez
        
        # Initialize field arrays
        if type == "2D":
            nx, ny = size
            self.Ez = None  # Will be initialized in subclasses
            self.Hx = None  # Will be initialized in subclasses
            self.Hy = None  # Will be initialized in subclasses
        else:
            raise ValueError("Only 2D simulations are supported at this time.")
    
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

    def run(self, steps=None):
        """Run the simulation.
        
        Args:
            steps (int, optional): Number of steps to run. If None, runs for full duration.
        """
        if steps is None:
            steps = self.num_steps
        # Implementation will be in subclasses
        raise NotImplementedError("Run method must be implemented in subclasses")
    
    def plot_field(self, field="Ez", t=0):
        """Plot the field at a given time"""
        #plt.imshow(self.Ez[t, :, :])
        #plt.colorbar()
        #plt.show()
        pass
