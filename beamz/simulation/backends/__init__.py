from beamz.simulation.backends.base import Backend
from beamz.simulation.backends.numpy_backend import NumPyBackend

# Conditionally import PyTorch backend if PyTorch is available
try:
    import torch
    from beamz.simulation.backends.torch_backend import TorchBackend
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def get_backend(name="numpy", **kwargs):
    """
    Get the specified backend for FDTD computations.
    
    Args:
        name: Name of the backend ('numpy' or 'torch')
        **kwargs: Additional arguments to pass to the backend
        
    Returns:
        Backend: The backend instance
    
    Raises:
        ValueError: If the backend is not supported
    """
    if name.lower() == "numpy":
        return NumPyBackend(**kwargs)
    elif name.lower() == "torch":
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Install it with 'pip install torch'")
        return TorchBackend(**kwargs)
    else:
        raise ValueError(f"Unsupported backend: {name}. Choose from: 'numpy', 'torch'") 