"""
Backend implementations for FDTD simulations.
"""
import logging

logger = logging.getLogger(__name__)

def get_backend(name="numpy", **kwargs):
    """Selected the backend."""
    name = name.lower()
        
    if name == "numpy":
        from beamz.simulation.backends.numpy_backend import NumPyBackend
        return NumPyBackend(**kwargs)
    
    if name == "torch":
        try:
            import torch
            from beamz.simulation.backends.torch_backend import TorchBackend
            return TorchBackend(**kwargs)
        except ImportError:
            logger.warning("PyTorch not available, falling back to NumPy backend")
            from beamz.simulation.backends.numpy_backend import NumPyBackend
            return NumPyBackend(**kwargs)
    
    raise ValueError(f"Unknown backend: {name}")