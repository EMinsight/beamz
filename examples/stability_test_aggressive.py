#!/usr/bin/env python3
"""
Aggressive stability test to push FDTD simulations to their limits.
Tests stability at very high Courant numbers (>95% of theoretical limit).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from beamz import *
from beamz.helpers import check_fdtd_stability
import numpy as np
import matplotlib.pyplot as plt

def test_high_courant_stability():
    """Test stability at very high Courant numbers."""
    print("=== Testing High Courant Stability Limits ===")
    
    # Basic parameters
    WL = 1.55*µm
    N_MAX = 2.0
    X, Y = 8*µm, 6*µm  # Smaller domain for faster testing
    TIME = 15*WL/LIGHT_SPEED  # Shorter simulation
    
    # Test very high safety factors
    safety_factors = [0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
    results = []
    
    for sf in safety_factors:
        print(f"\nTesting safety factor: {sf}")
        
        try:
            # Calculate parameters
            DX, DT = calc_optimal_fdtd_params(WL, N_MAX, dims=2, safety_factor=sf)
            
            # Check stability
            is_stable, courant, limit = check_fdtd_stability(DT, DX, dy=DX, n_max=N_MAX, safety_factor=1.0)
            print(f"  Courant number: {courant:.4f}, Limit: {limit:.4f}, Stable: {is_stable}")
            
            # Create simple design
            design = Design(width=X, height=Y, material=Material(N_MAX**2), pml_size=WL/2)
            
            # Simple Gaussian source
            time_steps = np.arange(0, TIME, DT)
            signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                                 phase=0, ramp_duration=WL*2/LIGHT_SPEED, t_max=TIME/2)
            source = GaussianSource(position=(X/4, Y/2), width=WL/15, signal=signal)
            design += source
            
            # Run simulation
            sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
            
            # Run a short simulation to test stability
            try:
                max_steps = min(50, len(time_steps))  # Very short test
                sim_results = sim.run(steps=max_steps, save=False, live=False, save_memory_mode=True)
                
                # Check field values
                if hasattr(sim, 'Ez'):
                    max_field = np.max(np.abs(sim.Ez))
                else:
                    max_field = 0
                
                # Check for instability
                unstable = max_field > 1e6
                
            except Exception as e:
                print(f"    Simulation error: {e}")
                max_field = float('inf')
                unstable = True
                    
            print(f"  Max field: {max_field:.2e}, Unstable: {unstable}")
            results.append((sf, courant, is_stable, unstable, max_field))
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append((sf, 0, False, True, float('inf')))
    
    return results

def test_source_injection_stability():
    """Test how different source injection methods affect high-Courant stability."""
    print("\n=== Testing Source Injection at High Courant Numbers ===")
    
    WL = 1.55*µm
    N_MAX = 2.0
    X, Y = 8*µm, 6*µm
    TIME = 15*WL/LIGHT_SPEED
    
    # Use high safety factor
    DX, DT = calc_optimal_fdtd_params(WL, N_MAX, dims=2, safety_factor=0.95)
    time_steps = np.arange(0, TIME, DT)
    
    # Check Courant number
    is_stable, courant, limit = check_fdtd_stability(DT, DX, dy=DX, n_max=N_MAX, safety_factor=1.0)
    print(f"Testing at Courant number: {courant:.4f} (limit: {limit:.4f})")
    
    signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                          phase=0, ramp_duration=WL*2/LIGHT_SPEED, t_max=TIME/2)
    
    # Test 1: Gaussian source
    print("\nTest 1: Gaussian Source")
    design1 = Design(width=X, height=Y, material=Material(N_MAX**2), pml_size=WL/2)
    source1 = GaussianSource(position=(X/4, Y/2), width=WL/15, signal=signal)
    design1 += source1
    
    sim1 = FDTD(design=design1, time=time_steps, mesh="regular", resolution=DX)
    result1 = run_aggressive_stability_test(sim1, "Gaussian", courant)
    
    # Test 2: Mode source
    print("\nTest 2: Mode Source")
    design2 = Design(width=X, height=Y, material=Material(N_MAX**2), pml_size=WL/2)
    design2 += Rectangle(position=(0, Y/2-WL/4), width=X/3, height=WL/2, material=Material(N_MAX**2))
    source2 = ModeSource(design=design2, start=(X/8, Y/2-WL/4), end=(X/8, Y/2+WL/4), 
                        wavelength=WL, signal=signal, direction="+x")
    design2 += source2
    
    sim2 = FDTD(design=design2, time=time_steps, mesh="regular", resolution=DX)
    result2 = run_aggressive_stability_test(sim2, "Mode", courant)
    
    return result1, result2

def test_material_boundary_stability():
    """Test stability at material boundaries with high Courant numbers."""
    print("\n=== Testing Material Boundary Stability ===")
    
    WL = 1.55*µm
    N_CORE = 2.0
    N_CLAD = 1.444
    X, Y = 10*µm, 8*µm
    TIME = 20*WL/LIGHT_SPEED
    
    # Test different safety factors
    safety_factors = [0.90, 0.95, 0.98]
    results = []
    
    for sf in safety_factors:
        print(f"\nTesting safety factor: {sf}")
        
        DX, DT = calc_optimal_fdtd_params(WL, N_CORE, dims=2, safety_factor=sf)
        time_steps = np.arange(0, TIME, DT)
        
        # Check Courant number
        is_stable, courant, limit = check_fdtd_stability(DT, DX, dy=DX, n_max=N_CORE, safety_factor=1.0)
        print(f"  Courant number: {courant:.4f}")
        
        signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                              phase=0, ramp_duration=WL*3/LIGHT_SPEED, t_max=TIME/2)
        
        # Create waveguide structure
        design = Design(width=X, height=Y, material=Material(N_CLAD**2), pml_size=WL/2)
        design += Rectangle(position=(0, Y/2-WL/4), width=X/2, height=WL/2, material=Material(N_CORE**2))
        
        source = ModeSource(design=design, start=(X/8, Y/2-WL/4), end=(X/8, Y/2+WL/4), 
                          wavelength=WL, signal=signal, direction="+x")
        design += source
        
        sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
        result = run_aggressive_stability_test(sim, f"Material boundary SF={sf}", courant)
        results.append((sf, courant, result))
    
    return results

def run_aggressive_stability_test(sim, test_name, courant):
    """Run an aggressive stability test."""
    try:
        # Run for a limited number of steps
        max_steps = min(100, len(sim.time))
        sim_results = sim.run(steps=max_steps, save=False, live=False, save_memory_mode=True)
        
        # Check field values
        if hasattr(sim, 'Ez'):
            max_field = np.max(np.abs(sim.Ez))
            field_std = np.std(np.abs(sim.Ez))
        else:
            max_field = 0
            field_std = 0
        
        # Check for instability
        unstable = max_field > 1e6
        
        # Check for rapid growth
        if max_field > 1e3:
            print(f"    Warning: High field values detected")
        
        print(f"  {test_name}: Max={max_field:.2e}, Std={field_std:.2e}, Unstable={unstable}")
        
        return {
            'max_field': max_field,
            'field_std': field_std,
            'unstable': unstable,
            'courant': courant
        }
        
    except Exception as e:
        print(f"  {test_name} - Error: {e}")
        return {
            'max_field': float('inf'),
            'field_std': float('inf'),
            'unstable': True,
            'courant': courant,
            'error': str(e)
        }

def test_pml_stability():
    """Test PML stability at high Courant numbers."""
    print("\n=== Testing PML Stability ===")
    
    WL = 1.55*µm
    N_MAX = 2.0
    X, Y = 8*µm, 6*µm
    TIME = 15*WL/LIGHT_SPEED
    
    # Test different PML sizes
    pml_sizes = [WL/4, WL/2, WL, 2*WL]
    results = []
    
    for pml_size in pml_sizes:
        print(f"\nTesting PML size: {pml_size/µm:.2f}µm")
        
        DX, DT = calc_optimal_fdtd_params(WL, N_MAX, dims=2, safety_factor=0.95)
        time_steps = np.arange(0, TIME, DT)
        
        # Check Courant number
        is_stable, courant, limit = check_fdtd_stability(DT, DX, dy=DX, n_max=N_MAX, safety_factor=1.0)
        print(f"  Courant number: {courant:.4f}")
        
        signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                              phase=0, ramp_duration=WL*2/LIGHT_SPEED, t_max=TIME/2)
        
        design = Design(width=X, height=Y, material=Material(N_MAX**2), pml_size=pml_size)
        source = GaussianSource(position=(X/4, Y/2), width=WL/15, signal=signal)
        design += source
        
        sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
        result = run_aggressive_stability_test(sim, f"PML {pml_size/µm:.2f}µm", courant)
        results.append((pml_size, courant, result))
    
    return results

if __name__ == "__main__":
    print("Running aggressive stability tests...")
    
    # Run all tests
    high_courant_results = test_high_courant_stability()
    source_results = test_source_injection_stability()
    material_results = test_material_boundary_stability()
    pml_results = test_pml_stability()
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("AGGRESSIVE STABILITY TEST SUMMARY")
    print("="*70)
    
    print("\n1. High Courant Stability:")
    for sf, courant, stable, unstable, max_field in high_courant_results:
        status = "UNSTABLE" if unstable else "STABLE"
        print(f"   SF={sf:.3f}: Courant={courant:.4f}, {status}, Max={max_field:.2e}")
    
    print("\n2. Source Injection at High Courant:")
    for name, result in zip(["Gaussian", "Mode"], source_results):
        if isinstance(result, dict):
            print(f"   {name}: Max={result['max_field']:.2e}, Unstable={result['unstable']}")
    
    print("\n3. Material Boundary Stability:")
    for sf, courant, result in material_results:
        if isinstance(result, dict):
            print(f"   SF={sf:.2f}: Max={result['max_field']:.2e}, Unstable={result['unstable']}")
    
    print("\n4. PML Stability:")
    for pml_size, courant, result in pml_results:
        if isinstance(result, dict):
            print(f"   PML {pml_size/µm:.2f}µm: Max={result['max_field']:.2e}, Unstable={result['unstable']}")
    
    # Find the highest stable Courant number
    max_stable_courant = 0
    for sf, courant, stable, unstable, max_field in high_courant_results:
        if not unstable and courant > max_stable_courant:
            max_stable_courant = courant
    
    print(f"\n🎯 MAXIMUM STABLE COURANT NUMBER: {max_stable_courant:.4f}")
    print(f"   Theoretical limit: 0.7071")
    print(f"   Achievement: {max_stable_courant/0.7071*100:.1f}% of theoretical limit")