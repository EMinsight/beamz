#!/usr/bin/env python3
"""
Basic stability test for FDTD simulations.
Tests the fundamental Courant condition limits with simple sources.
"""

from beamz import *
from beamz.helpers import check_fdtd_stability
import numpy as np
import matplotlib.pyplot as plt

def test_courant_stability():
    """Test stability at different Courant numbers."""
    print("=== Testing Courant Stability Limits ===")
    
    # Basic parameters
    WL = 1.55*µm
    N_MAX = 2.0
    X, Y = 10*µm, 10*µm
    TIME = 20*WL/LIGHT_SPEED
    
    # Test different safety factors
    safety_factors = [0.99, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30]
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
            design = Design(width=X, height=Y, material=Material(N_MAX**2), pml_size=WL)
            
            # Simple Gaussian source
            time_steps = np.arange(0, TIME, DT)
            signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                                 phase=0, ramp_duration=WL*3/LIGHT_SPEED, t_max=TIME/2)
            source = GaussianSource(position=(X/4, Y/2), width=WL/10, signal=signal)
            design += source
            
            # Run simulation
            sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
            
            # Run a short simulation to test stability
            try:
                # Run for a limited number of steps
                max_steps = min(100, len(time_steps))
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

def test_source_injection_methods():
    """Test different source injection methods."""
    print("\n=== Testing Source Injection Methods ===")
    
    WL = 1.55*µm
    N_MAX = 2.0
    X, Y = 10*µm, 10*µm
    TIME = 20*WL/LIGHT_SPEED
    DX, DT = calc_optimal_fdtd_params(WL, N_MAX, dims=2, safety_factor=0.8)
    
    time_steps = np.arange(0, TIME, DT)
    signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                          phase=0, ramp_duration=WL*3/LIGHT_SPEED, t_max=TIME/2)
    
    # Test 1: Gaussian source
    print("\nTest 1: Gaussian Source")
    design1 = Design(width=X, height=Y, material=Material(N_MAX**2), pml_size=WL)
    source1 = GaussianSource(position=(X/4, Y/2), width=WL/10, signal=signal)
    design1 += source1
    
    sim1 = FDTD(design=design1, time=time_steps, mesh="regular", resolution=DX)
    max_field1 = run_stability_test(sim1, "Gaussian")
    
    # Test 2: Mode source
    print("\nTest 2: Mode Source")
    design2 = Design(width=X, height=Y, material=Material(N_MAX**2), pml_size=WL)
    design2 += Rectangle(position=(0, Y/2-WL/4), width=X/2, height=WL/2, material=Material(N_MAX**2))
    source2 = ModeSource(design=design2, start=(X/8, Y/2-WL/4), end=(X/8, Y/2+WL/4), 
                        wavelength=WL, signal=signal)
    design2 += source2
    
    sim2 = FDTD(design=design2, time=time_steps, mesh="regular", resolution=DX)
    max_field2 = run_stability_test(sim2, "Mode")
    
    return max_field1, max_field2

def run_stability_test(sim, source_type):
    """Run a stability test on a simulation."""
    try:
        # Run for a limited number of steps
        max_steps = min(200, len(sim.time))
        sim_results = sim.run(steps=max_steps, save=False, live=False, save_memory_mode=True)
        
        # Check field values
        if hasattr(sim, 'Ez'):
            max_field = np.max(np.abs(sim.Ez))
        else:
            max_field = 0
        
        # Check for instability
        unstable = max_field > 1e6
        
    except Exception as e:
        print(f"  {source_type} source - Error: {e}")
        max_field = float('inf')
        unstable = True
    
    print(f"  {source_type} source - Max field: {max_field:.2e}, Unstable: {unstable}")
    return max_field

if __name__ == "__main__":
    # Run stability tests
    courant_results = test_courant_stability()
    source_results = test_source_injection_methods()
    
    # Print summary
    print("\n=== SUMMARY ===")
    print("Courant Stability Results:")
    for sf, courant, stable, unstable, max_field in courant_results:
        status = "UNSTABLE" if unstable else "STABLE"
        print(f"  SF={sf:.2f}: Courant={courant:.4f}, {status}, Max={max_field:.2e}")
    
    print(f"\nSource Injection Results:")
    print(f"  Gaussian: Max={source_results[0]:.2e}")
    print(f"  Mode: Max={source_results[1]:.2e}")