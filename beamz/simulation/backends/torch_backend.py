import numpy as np
import torch
from beamz.simulation.backends.base import Backend

class TorchBackend(Backend):
    """PyTorch backend for FDTD computations."""
    
    def __init__(self, device="auto", **kwargs):
        """Initialize PyTorch backend.
        
        Args:
            device: Device to use for computation:
                   - "auto": automatically select the best available device
                   - "cuda": use NVIDIA GPU if available
                   - "mps": use Apple Metal (M-series chips) if available
                   - "cpu": use CPU
                   - or a specific device like "cuda:0", "mps:0"
            **kwargs: Additional arguments to pass to torch
        """
        self.device_name = device
        
        # Auto-detect best available device
        if device.lower() == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"PyTorch backend using CUDA GPU: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print(f"PyTorch backend using Apple Metal (MPS)")
            else:
                self.device = torch.device("cpu")
                print(f"PyTorch backend using CPU")
        # Try to use CUDA
        elif "cuda" in device:
            if not torch.cuda.is_available():
                print(f"CUDA is not available. Falling back to CPU.")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)
                print(f"PyTorch backend using CUDA GPU: {torch.cuda.get_device_name(0)}")
        # Try to use MPS (Apple Metal)
        elif "mps" in device:
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                print(f"Apple Metal (MPS) is not available. Falling back to CPU.")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)
                print(f"PyTorch backend using Apple Metal (MPS)")
        # Use CPU
        else:
            self.device = torch.device(device)
            print(f"PyTorch backend using CPU")
            
        # Set default dtype
        self.dtype = kwargs.get("dtype", torch.float32)
        
        # Performance optimization for Apple Silicon
        if self.device.type == "mps" and not kwargs.get("disable_optimizations", False):
            # Note: These optimizations are experimental
            print("Applying MPS-specific optimizations...")
            
            # Adjust memory management for better efficiency
            torch.mps.empty_cache()
            
            # Try to set MPS stream options if available
            try:
                # This is experimental and not documented but might help with performance
                torch._mps_synchronize = False
            except:
                pass
            
            # Try to avoid MPS graph capture which can be slow for small operations
            try:
                torch.mps._enable_capture_graphs = False
            except:
                pass
    
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
        
        # Using in-place operations to avoid memory allocations
        Hx.mul_(factor_x)
        Hx.sub_(source_x * curl_e_x)
        
        # Update Hy with semi-implicit scheme for magnetic conductivity
        denom_y = 1.0 + sigma_m_y * dt / (2.0 * mu_0)
        factor_y = (1.0 - sigma_m_y * dt / (2.0 * mu_0)) / denom_y
        source_y = (dt / mu_0) / denom_y
        
        # Using in-place operations
        Hy.mul_(factor_y)
        Hy.add_(source_y * curl_e_y)
        
        return Hx, Hy
    
    def update_e_field(self, Ez, Hx, Hy, sigma, epsilon_r, dx, dy, dt, eps_0):
        """Update electric field component with conductivity (including PML)."""
        # Pre-allocate tensors for curls to avoid repeated allocations
        curl_h_x = torch.zeros_like(Ez)
        curl_h_y = torch.zeros_like(Ez)
        
        # Interior points calculation (use in-place operations where possible)
        curl_h_x[1:-1, 1:-1] = (Hx[1:-1, 1:] - Hx[1:-1, :-1]) / dy
        curl_h_y[1:-1, 1:-1] = (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) / dx
        
        # Extract the inner region for calculations to avoid excessive indexing
        inner_sigma = sigma[1:-1, 1:-1]
        inner_epsilon_r = epsilon_r[1:-1, 1:-1]
        inner_Ez = Ez[1:-1, 1:-1]
        
        # Prepare factors for computation
        denom = 1.0 + inner_sigma * dt / (2.0 * eps_0 * inner_epsilon_r)
        factor1 = (1.0 - inner_sigma * dt / (2.0 * eps_0 * inner_epsilon_r)) / denom
        factor2 = (dt / (eps_0 * inner_epsilon_r)) / denom
        
        # Compute the combined curl contribution
        curl_combined = -curl_h_x[1:-1, 1:-1] + curl_h_y[1:-1, 1:-1]
        
        # Update inner Ez (avoiding excessive indexing)
        new_inner_Ez = factor1 * inner_Ez + factor2 * curl_combined
        Ez[1:-1, 1:-1] = new_inner_Ez
        
        return Ez 