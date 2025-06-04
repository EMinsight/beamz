"""
Backend implementations for FDTD simulations.
"""
import logging

logger = logging.getLogger(__name__)

def get_backend(name="numpy", **kwargs):
    """Select the backend."""
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
    
    if name == "jax":
        try:
            import jax
            from beamz.simulation.backends.jax_backend import JAXBackend
            return JAXBackend(**kwargs)
        except ImportError as e:
            logger.warning(f"JAX not available ({e}), falling back to NumPy backend")
            from beamz.simulation.backends.numpy_backend import NumPyBackend
            return NumPyBackend(**kwargs)
    
    raise ValueError(f"Unknown backend: {name}")

# Export available backends
__all__ = ['get_backend']