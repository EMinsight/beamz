from abc import ABC, abstractmethod

class Backend(ABC):
    """Abstract base class for FDTD computation backends."""
    
    @abstractmethod
    def zeros(self, shape):
        """Create an array of zeros with the given shape."""
        pass
    
    @abstractmethod
    def ones(self, shape):
        """Create an array of ones with the given shape."""
        pass
    
    @abstractmethod
    def copy(self, array):
        """Create a copy of the array."""
        pass
    
    @abstractmethod
    def to_numpy(self, array):
        """Convert the array to a numpy array."""
        pass
    
    @abstractmethod
    def from_numpy(self, array):
        """Convert a numpy array to the backend's array type."""
        pass
    
    @abstractmethod
    def roll(self, array, shift, axis=None):
        """Roll array elements along a given axis."""
        pass
    
    @abstractmethod
    def update_h_fields(self, Hx, Hy, Ez, sigma, dx, dy, dt, mu_0, eps_0):
        """Update magnetic field components."""
        pass
    
    @abstractmethod
    def update_e_field(self, Ez, Hx, Hy, sigma, epsilon_r, dx, dy, dt, eps_0):
        """Update electric field component."""
        pass 