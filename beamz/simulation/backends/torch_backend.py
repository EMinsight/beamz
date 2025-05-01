import numpy as np
import torch
from beamz.simulation.backends.base import Backend

class TorchBackend(Backend):
    """PyTorch backend for FDTD computations."""
    
    def __init__(self, device="cuda", **kwargs):
        """Initialize PyTorch backend.
        
        Args:
            device: Device to use for computation ("cuda", "cpu", or specific cuda device)
            **kwargs: Additional arguments to pass to torch
        """
        self.device_name = device
        
        # Check if CUDA is available when cuda is requested
        if "cuda" in device and not torch.cuda.is_available():
            print(f"CUDA is not available. Falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        # Set default dtype
        self.dtype = kwargs.get("dtype", torch.float32)
        
        print(f"PyTorch backend initialized on {self.device}")
    
    def zeros(self, shape):
        """Create an array of zeros with the given shape."""
        return torch.zeros(shape, dtype=self.dtype, device=self.device)
    
    def ones(self, shape):
        """Create an array of ones with the given shape."""
        return torch.ones(shape, dtype=self.dtype, device=self.device)
    
    def copy(self, array):
        """Create a copy of the array."""
        return array.clone()
    
    def to_numpy(self, array):
        """Convert the array to a numpy array."""
        if array.device.type != "cpu":
            return array.cpu().detach().numpy()
        return array.detach().numpy()
    
    def from_numpy(self, array):
        """Convert a numpy array to the backend's array type."""
        return torch.tensor(array, dtype=self.dtype, device=self.device)
    
    def update_h_fields(self, Hx, Hy, Ez, sigma, dx, dy, dt, mu_0, eps_0):
        """Update magnetic field components with conductivity (including PML)."""
        # Calculate magnetic conductivity from electric conductivity with impedance matching
        sigma_m_x = sigma[:, :-1] * mu_0 / eps_0
        sigma_m_y = sigma[:-1, :] * mu_0 / eps_0
        # Calculate curl of E for H-field updates
        curl_e_x = (Ez[:, 1:] - Ez[:, :-1]) / dy
        curl_e_y = (Ez[1:, :] - Ez[:-1, :]) / dx
        # Update Hx with semi-implicit scheme for magnetic conductivity
        denom_x = 1.0 + sigma_m_x * dt / (2.0 * mu_0)
        factor_x = (1.0 - sigma_m_x * dt / (2.0 * mu_0)) / denom_x
        source_x = (dt / mu_0) / denom_x
        Hx[:] = factor_x * Hx - source_x * curl_e_x
        # Update Hy with semi-implicit scheme for magnetic conductivity
        denom_y = 1.0 + sigma_m_y * dt / (2.0 * mu_0)
        factor_y = (1.0 - sigma_m_y * dt / (2.0 * mu_0)) / denom_y
        source_y = (dt / mu_0) / denom_y
        Hy[:] = factor_y * Hy + source_y * curl_e_y
        
        return Hx, Hy
    
    def update_e_field(self, Ez, Hx, Hy, sigma, epsilon_r, dx, dy, dt, eps_0):
        """Update electric field component with conductivity (including PML)."""
        # Calculate curl of H
        curl_h_x = torch.zeros_like(Ez)
        curl_h_y = torch.zeros_like(Ez)
        # Interior points calculation
        curl_h_x[1:-1, 1:-1] = (Hx[1:-1, 1:] - Hx[1:-1, :-1]) / dy
        curl_h_y[1:-1, 1:-1] = (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) / dx
        # For better numerical stability, use semi-implicit scheme for conductivity
        # First calculate the denominator
        denom = 1.0 + sigma[1:-1, 1:-1] * dt / (2.0 * eps_0 * epsilon_r[1:-1, 1:-1])
        # Then the numerator factors
        factor1 = (1.0 - sigma[1:-1, 1:-1] * dt / (2.0 * eps_0 * epsilon_r[1:-1, 1:-1])) / denom
        factor2 = (dt / (eps_0 * epsilon_r[1:-1, 1:-1])) / denom
        # Update Ez field with FDTD, conductivity term handles PML regions
        Ez[1:-1, 1:-1] = factor1 * Ez[1:-1, 1:-1] + factor2 * (-curl_h_x[1:-1, 1:-1] + curl_h_y[1:-1, 1:-1])
        
        return Ez 