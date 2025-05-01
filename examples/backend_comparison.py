import numpy as np
import matplotlib.pyplot as plt
import time
from beamz.const import µm
from beamz.design.structures import RectangularWaveguide
from beamz.design.design import Design
from beamz.design.sources import GaussianSource
from beamz.simulation.fdtd import FDTD

def run_simulation(backend="numpy", backend_options=None, resolution=0.02*µm):
    """Run the simulation with the specified backend."""
    # Create a simple waveguide design
    waveguide = RectangularWaveguide(
        position=(1.5*µm, 1.5*µm),
        width=0.4*µm,
        height=3*µm,
        permittivity=12.0
    )
    
    # Create a design
    design = Design(width=3*µm, height=3*µm, structures=[waveguide])
    
    # Add a Gaussian source
    source = GaussianSource(
        position=(1.5*µm, 0.5*µm),
        width=0.2*µm,
        amplitude=1.0,
        frequency=200e12,  # 200 THz
        pulse_width=2e-14,  # 20 fs
        phase=0.0
    )
    design.add_source(source)
    
    # Create time array
    t_end = 100e-15  # 100 fs
    dt = resolution / (0.5 * 3e8)  # Courant factor of 0.5
    time = np.arange(0, t_end, dt)
    
    # Initialize and run FDTD simulation
    print(f"Running with {backend} backend...")
    start_time = time.time()
    
    # Create FDTD simulation
    fdtd = FDTD(
        design=design,
        time=time,
        resolution=resolution,
        backend=backend,
        backend_options=backend_options
    )
    
    # Run the simulation
    results = fdtd.run(save=True, live=False)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Simulation completed in {elapsed_time:.2f} seconds")
    
    return fdtd, results, elapsed_time

def compare_backends(resolution=0.02*µm):
    """Compare NumPy and PyTorch backends."""
    # Run with NumPy backend
    fdtd_numpy, results_numpy, time_numpy = run_simulation(backend="numpy", resolution=resolution)
    
    # Try running with PyTorch backend
    try:
        # Try with GPU first
        fdtd_torch, results_torch, time_torch = run_simulation(
            backend="torch",
            backend_options={"device": "cuda"},
            resolution=resolution
        )
    except (ImportError, RuntimeError):
        try:
            # Fall back to CPU if GPU is not available
            print("CUDA not available, using PyTorch on CPU instead")
            fdtd_torch, results_torch, time_torch = run_simulation(
                backend="torch",
                backend_options={"device": "cpu"},
                resolution=resolution
            )
        except ImportError:
            print("PyTorch is not installed. Install with: pip install torch")
            return
    
    # Plot results for comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot the final Ez field for both backends
    vmin, vmax = -1, 1
    
    # NumPy backend
    im1 = axes[0].imshow(results_numpy['Ez'][-1], origin='lower', cmap='RdBu', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'NumPy Backend ({time_numpy:.2f}s)')
    
    # PyTorch backend
    im2 = axes[1].imshow(results_torch['Ez'][-1], origin='lower', cmap='RdBu', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'PyTorch Backend ({time_torch:.2f}s)')
    
    # Add colorbar
    cbar = fig.colorbar(im1, ax=axes)
    cbar.set_label('Ez')
    
    plt.tight_layout()
    plt.savefig('backend_comparison.png', dpi=300)
    plt.show()
    
    # Print speedup
    speedup = time_numpy / time_torch
    print(f"Speedup with PyTorch: {speedup:.2f}x")
    
    return fdtd_numpy, fdtd_torch, speedup

if __name__ == '__main__':
    # Run with default resolution
    fdtd_numpy, fdtd_torch, speedup = compare_backends()
    
    # Optionally run with finer resolution to see more significant speedup
    # fdtd_numpy, fdtd_torch, speedup = compare_backends(resolution=0.01*µm) 