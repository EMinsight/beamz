"""
Example: JAX Backend Performance Benchmark for FDTD

This example demonstrates JAX performance benefits for pure FDTD simulation:
1. JIT compilation for fast CPU/GPU execution
2. Performance comparison with NumPy and PyTorch backends
3. Memory efficiency analysis
4. Multi-device support testing

IMPORTANT: This is PURE PERFORMANCE testing - no optimization, no inverse design,
no automatic differentiation. Just fast FDTD simulation!
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
from typing import Dict, Any

# Import BEAMZ components
from beamz.design import Design
from beamz.design.structures import Rectangle
from beamz.design.materials import Material
from beamz.design.sources import GaussianSource
from beamz.simulation.fdtd import FDTD
from beamz.simulation.backends import get_backend
from beamz.const import *

# Define additional time constants
fs = 1e-15  # femtosecond


def create_performance_test_design():
    """Create a moderately complex design for performance testing."""
    # Larger design for meaningful performance comparison
    design = Design(width=8*µm, height=6*µm)
    
    # Add substrate
    substrate = Material(permittivity=2.25)  # SiO2
    substrate_rect = Rectangle(
        material=substrate,
        position=(0, 0),
        width=design.width,
        height=design.height
    )
    design.add(substrate_rect)
    
    # Add several silicon structures for complexity
    silicon = Material(permittivity=12.0)  # Si
    
    # Add multiple rectangles to create field complexity
    for i in range(5):
        x_pos = -3*µm + i * 1.5*µm
        rect = Rectangle(
            material=silicon,
            position=(x_pos, 0),
            width=0.5*µm,
            height=2*µm
        )
        design.add(rect)
    
    # Create a simple time-dependent signal (just a constant for the benchmark)
    # In a real simulation, you'd create a proper time array and signal
    signal = 1.0  # Simple constant signal for performance testing
    
    # Add Gaussian pulse source
    source = GaussianSource(
        position=(-3.5*µm, 0),
        width=0.2*µm,  # Spatial width of the Gaussian
        signal=signal
    )
    design.add(source)
    
    return design


def benchmark_backend_performance(backend_name: str, design: Design, time_steps: np.ndarray, 
                                resolution: float, num_time_steps: int = 100):
    """Benchmark a specific backend's performance."""
    print(f"\n=== Benchmarking {backend_name.upper()} Backend ===")
    
    try:
        # Initialize simulation
        sim = FDTD(
            design=design,
            time=time_steps,
            resolution=resolution,
            backend=backend_name
        )
        
        # Get backend info
        if hasattr(sim.backend, 'get_device_info'):
            info = sim.backend.get_device_info()
            print(f"Device info: {info}")
        
        # Warmup compilation (for JAX)
        if backend_name == "jax" and hasattr(sim.backend, 'warmup_compilation'):
            field_shapes = {
                "Hx": sim.Hx.shape,
                "Hy": sim.Hy.shape,
                "Ez": sim.Ez.shape
            }
            sim.backend.warmup_compilation(field_shapes)
        
        # Benchmark simulation
        print(f"Running {num_time_steps} time steps...")
        
        start_time = time.time()
        
        # Run FDTD time steps
        for step in range(num_time_steps):
            if sim.is_3d:
                # 3D simulation (if implemented)
                pass
            else:
                # 2D simulation
                sim.Hx, sim.Hy = sim.backend.update_h_fields(
                    sim.Hx, sim.Hy, sim.Ez, sim.sigma,
                    sim.dx, sim.dy, sim.dt, 4*np.pi*1e-7, 8.854e-12
                )
                sim.Ez = sim.backend.update_e_field(
                    sim.Ez, sim.Hx, sim.Hy, sim.sigma, sim.epsilon_r,
                    sim.dx, sim.dy, sim.dt, 8.854e-12
                )
                
                # Update source
                if hasattr(sim, 'sources') and sim.sources:
                    for source in sim.sources:
                        # For benchmarking, just add a simple time-varying source
                        # In real simulations, sources would have proper time functions
                        source_value = np.sin(2 * np.pi * step * sim.dt * 1e15)  # 1 PHz frequency
                        
                        # Apply source (simplified) - find a reasonable location
                        source_x = int(sim.Ez.shape[0] * 0.1)  # 10% from left edge
                        source_y = int(sim.Ez.shape[1] * 0.5)  # Center
                        
                        if 0 <= source_x < sim.Ez.shape[0] and 0 <= source_y < sim.Ez.shape[1]:
                            if backend_name == "jax":
                                sim.Ez = sim.Ez.at[source_x, source_y].add(source_value * 0.1)
                            else:
                                sim.Ez[source_x, source_y] += source_value * 0.1
        
        # Ensure computation is complete (especially important for JAX/GPU)
        if backend_name == "jax":
            if hasattr(sim.Ez, 'block_until_ready'):
                sim.Ez.block_until_ready()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance metrics
        time_per_step = total_time / num_time_steps
        grid_size = sim.Ez.shape[0] * sim.Ez.shape[1]
        updates_per_second = (grid_size * num_time_steps) / total_time
        
        results = {
            "backend": backend_name,
            "total_time": total_time,
            "time_per_step": time_per_step,
            "grid_size": grid_size,
            "updates_per_second": updates_per_second,
            "memory_usage": getattr(sim.backend, 'memory_usage', lambda: None)()
        }
        
        print(f"Total time: {total_time:.4f} seconds")
        print(f"Time per step: {time_per_step:.6f} seconds")
        print(f"Grid points per second: {updates_per_second:.0f}")
        
        return results, sim
        
    except Exception as e:
        print(f"Benchmark failed for {backend_name}: {e}")
        return None, None


def compare_all_backends():
    """Compare performance across all available backends."""
    print("=== FDTD Backend Performance Comparison ===\n")
    
    # Create test design
    print("Creating test design...")
    design = create_performance_test_design()
    
    # Simulation parameters
    total_time = 500*fs
    dt = 1*fs
    time_steps = np.arange(0, total_time, dt)
    resolution = 50*nm  # Reasonable resolution for performance testing
    num_time_steps = 200  # Enough steps to measure performance
    
    print(f"Grid resolution: {resolution*1e9:.1f} nm")
    print(f"Time steps: {num_time_steps}")
    print(f"Time step size: {dt*1e15:.1f} fs")
    
    # Test all available backends
    backends_to_test = ["numpy"]
    
    # Check if JAX is available
    try:
        import jax
        backends_to_test.append("jax")
    except ImportError:
        print("JAX not available")
    
    # Check if PyTorch is available
    try:
        import torch
        backends_to_test.append("torch")
    except ImportError:
        print("PyTorch not available")
    
    results = {}
    simulations = {}
    
    # Benchmark each backend
    for backend_name in backends_to_test:
        result, sim = benchmark_backend_performance(
            backend_name, design, time_steps, resolution, num_time_steps
        )
        if result:
            results[backend_name] = result
            simulations[backend_name] = sim
    
    # Performance comparison
    if len(results) > 1:
        print(f"\n=== Performance Summary ===")
        
        # Use numpy as baseline
        baseline = results.get("numpy")
        if baseline:
            baseline_time = baseline["time_per_step"]
            baseline_throughput = baseline["updates_per_second"]
            
            print(f"{'Backend':<10} {'Time/Step':<12} {'Speedup':<10} {'Throughput':<15}")
            print("-" * 50)
            
            for name, result in results.items():
                speedup = baseline_time / result["time_per_step"]
                throughput = result["updates_per_second"]
                
                print(f"{name:<10} {result['time_per_step']:.6f}s   {speedup:.2f}x      {throughput:.0f} pts/s")
    
    # Plot field comparison if available
    if len(simulations) > 1:
        plot_field_comparison(simulations)
    
    return results


def plot_field_comparison(simulations: Dict[str, Any]):
    """Plot field patterns from different backends to verify consistency."""
    print("\nPlotting field comparison...")
    
    num_sims = len(simulations)
    fig, axes = plt.subplots(1, num_sims, figsize=(5*num_sims, 4))
    
    if num_sims == 1:
        axes = [axes]
    
    for i, (name, sim) in enumerate(simulations.items()):
        # Convert to numpy for plotting
        if hasattr(sim.backend, 'to_numpy'):
            ez_field = sim.backend.to_numpy(sim.Ez)
        else:
            ez_field = np.array(sim.Ez)
        
        # Handle complex fields by taking the real part
        if np.iscomplexobj(ez_field):
            ez_field = np.real(ez_field)
        
        # Handle NaN values
        if np.any(np.isnan(ez_field)) or np.any(np.isinf(ez_field)):
            print(f"Warning: {name} backend has NaN/inf values in field")
            ez_field = np.nan_to_num(ez_field, nan=0.0, posinf=0.0, neginf=0.0)
        
        im = axes[i].imshow(ez_field.T, cmap='RdBu', aspect='equal', 
                           origin='lower', interpolation='bilinear')
        axes[i].set_title(f'{name.upper()} Backend')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig('jax_fdtd_benchmark.png', dpi=150, bbox_inches='tight')
    plt.show()


def test_jax_specific_features():
    """Test JAX-specific performance features."""
    print("\n=== Testing JAX-Specific Features ===")
    
    try:
        import jax
        
        # Get JAX backend
        jax_backend = get_backend("jax")
        
        print("JAX Backend Features:")
        print(f"  JIT Compilation: {'Enabled' if jax_backend.use_jit else 'Disabled'}")
        print(f"  Multi-device: {jax_backend.num_devices} devices")
        print(f"  64-bit precision: {'Yes' if jax_backend.use_64bit else 'No'}")
        
        # Test basic operations with different array sizes
        test_sizes = [
            ("Small", (100, 100)),
            ("Medium", (500, 500)),
            ("Large", (1000, 1000))
        ]
        
        print(f"\nTesting array operations:")
        for name, shape in test_sizes:
            print(f"  {name} arrays {shape}:")
            
            # Test addition
            a = jax_backend.ones(shape)
            b = jax_backend.ones(shape)
            
            # Time the operation
            time_taken, _ = jax_backend.benchmark_function(lambda x, y: x + y, a, b, num_runs=10)
            
            throughput = (shape[0] * shape[1] * 10) / time_taken
            print(f"    Addition: {time_taken:.6f}s ({throughput:.0f} ops/s)")
        
        # Compare with NumPy
        numpy_backend = get_backend("numpy")
        comparison_shapes = {"medium": (500, 500), "large": (1000, 1000)}
        jax_backend.compare_with_numpy(numpy_backend, comparison_shapes)
        
    except ImportError:
        print("JAX not available for feature testing")
    except Exception as e:
        print(f"JAX feature testing failed: {e}")


def run_jax_performance_benchmark():
    """Run the complete JAX performance benchmark (no optimization - just speed testing)."""
    print("🚀 JAX FDTD Performance Benchmark")
    print("=" * 50)
    print("Testing JAX backend performance for pure FDTD simulation")
    print("(No optimization, no inverse design - just raw speed!)")
    print()
    
    # 1. Compare all available backends
    results = compare_all_backends()
    
    # 2. Test JAX-specific features
    test_jax_specific_features()
    
    # 3. Recommendations
    print(f"\n=== Recommendations ===")
    
    if "jax" in results:
        jax_result = results["jax"]
        numpy_result = results.get("numpy")
        
        if numpy_result:
            speedup = numpy_result["time_per_step"] / jax_result["time_per_step"]
            
            if speedup > 2:
                print(f"✅ JAX provides significant speedup ({speedup:.1f}x faster than NumPy)")
                print("   Recommended for production FDTD simulations")
            elif speedup > 1.2:
                print(f"✅ JAX provides moderate speedup ({speedup:.1f}x faster than NumPy)")
                print("   Consider using JAX for large simulations")
            else:
                print(f"⚠️  JAX speedup is minimal ({speedup:.1f}x)")
                print("   NumPy might be sufficient for your use case")
        
        # Check memory usage
        memory_info = jax_result.get("memory_usage")
        if memory_info:
            print(f"📊 JAX memory usage: {memory_info}")
    else:
        print("❌ JAX not available - install with: pip install jax")
    
    print(f"\n=== Summary ===")
    print("JAX Backend provides:")
    print("• JIT compilation for CPU/GPU acceleration")
    print("• Efficient vectorized operations")
    print("• Multi-device parallelization")
    print("• Memory-efficient computations")
    print("• Pure FDTD performance - no AD overhead")
    print("\nPerfect for high-performance FDTD simulations! 🎯")


if __name__ == "__main__":
    run_jax_performance_benchmark() 