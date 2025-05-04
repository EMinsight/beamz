"""
Backend implementations for FDTD simulations.
"""

import time
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_backend(name="numpy", **kwargs):
    """
    Get a specific backend implementation.
    
    Args:
        name (str): Backend name ('numpy', 'torch', 'jax', or 'auto')
        **kwargs: Backend-specific options
    
    Returns:
        Backend: Backend implementation instance
    """
    name = name.lower()
    
    if name == "auto":
        return select_fastest_backend(**kwargs)
    
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
            from beamz.simulation.backends.jax_backend import JaxBackend
            return JaxBackend(**kwargs)
        except ImportError:
            logger.warning("JAX not available, falling back to NumPy backend")
            from beamz.simulation.backends.numpy_backend import NumPyBackend
            return NumPyBackend(**kwargs)
    
    raise ValueError(f"Unknown backend: {name}")

def select_fastest_backend(**kwargs):
    """
    Automatically select the fastest backend by running a small benchmark.
    
    Args:
        **kwargs: Backend-specific options
    
    Returns:
        Backend: The fastest backend implementation for the current system
    """
    # Small benchmark sizes
    grid_size = (100, 100)
    n_steps = 10
    
    backends = []
    times = []
    
    # Try NumPy backend
    from beamz.simulation.backends.numpy_backend import NumPyBackend
    backends.append(NumPyBackend(**kwargs))
    times.append(benchmark_backend(backends[-1], grid_size, n_steps))
    logger.info(f"NumPy backend benchmark: {times[-1]:.4f}s")
    
    # Try PyTorch backend
    try:
        import torch
        from beamz.simulation.backends.torch_backend import TorchBackend
        
        # Try CPU
        cpu_kwargs = kwargs.copy()
        cpu_kwargs["device"] = "cpu"
        backends.append(TorchBackend(**cpu_kwargs))
        times.append(benchmark_backend(backends[-1], grid_size, n_steps))
        logger.info(f"PyTorch CPU backend benchmark: {times[-1]:.4f}s")
        
        # Try CUDA if available
        if torch.cuda.is_available():
            cuda_kwargs = kwargs.copy()
            cuda_kwargs["device"] = "cuda"
            backends.append(TorchBackend(**cuda_kwargs))
            times.append(benchmark_backend(backends[-1], grid_size, n_steps))
            logger.info(f"PyTorch CUDA backend benchmark: {times[-1]:.4f}s")
        
        # Try MPS if available
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            mps_kwargs = kwargs.copy()
            mps_kwargs["device"] = "mps"
            backends.append(TorchBackend(**mps_kwargs))
            times.append(benchmark_backend(backends[-1], grid_size, n_steps))
            logger.info(f"PyTorch MPS backend benchmark: {times[-1]:.4f}s")
    except ImportError:
        logger.info("PyTorch not available for benchmarking")
    
    # Try JAX backend
    try:
        import jax
        from beamz.simulation.backends.jax_backend import JaxBackend
        backends.append(JaxBackend(**kwargs))
        times.append(benchmark_backend(backends[-1], grid_size, n_steps))
        logger.info(f"JAX backend benchmark: {times[-1]:.4f}s")
    except ImportError:
        logger.info("JAX not available for benchmarking")
    
    # Find the fastest backend
    fastest_idx = np.argmin(times)
    fastest_backend = backends[fastest_idx]
    fastest_time = times[fastest_idx]
    
    logger.info(f"Selected fastest backend: {type(fastest_backend).__name__} ({fastest_time:.4f}s)")
    return fastest_backend

def benchmark_backend(backend, grid_size, n_steps):
    """
    Benchmark a specific backend implementation.
    
    Args:
        backend: Backend implementation
        grid_size: Grid size for testing
        n_steps: Number of steps to run
    
    Returns:
        float: Time taken in seconds
    """
    try:
        # Create test data
        Hx = backend.zeros((grid_size[0], grid_size[1]-1))
        Hy = backend.zeros((grid_size[0]-1, grid_size[1]))
        Ez = backend.zeros(grid_size)
        sigma = backend.ones(grid_size) * 0.1
        eps_r = backend.ones(grid_size) * 2.0
        
        # Parameters
        dx = dy = 1e-6
        dt = 1e-15
        mu0 = 1.25663706212e-6
        eps0 = 8.85418781762e-12
        
        # Warm-up
        for _ in range(2):
            Hx, Hy = backend.update_h_fields(Hx, Hy, Ez, sigma, dx, dy, dt, mu0, eps0)
            Ez = backend.update_e_field(Ez, Hx, Hy, sigma, eps_r, dx, dy, dt, eps0)
        
        # Benchmark
        start_time = time.time()
        for _ in range(n_steps):
            Hx, Hy = backend.update_h_fields(Hx, Hy, Ez, sigma, dx, dy, dt, mu0, eps0)
            Ez = backend.update_e_field(Ez, Hx, Hy, sigma, eps_r, dx, dy, dt, eps0)
        end_time = time.time()
        
        return end_time - start_time
    except Exception as e:
        # Catch errors and return a very large time to avoid this backend
        logger.warning(f"Error benchmarking backend {type(backend).__name__}: {str(e)}")
        return float('inf') 