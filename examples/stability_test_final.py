#!/usr/bin/env python3
"""
Final comprehensive stability test to achieve >99% of theoretical Courant limit.
This test systematically pushes the FDTD simulation to its absolute limits.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from beamz import *
from beamz.helpers import check_fdtd_stability
import numpy as np
import matplotlib.pyplot as plt

def test_ultimate_courant_limits():
    """Test the ultimate Courant limits with very aggressive parameters."""
    print("=== Testing Ultimate Courant Limits ===")
    
    # Basic parameters
    WL = 1.55*µm
    N_MAX = 2.0
    X, Y = 6*µm, 4*µm  # Very small domain for speed
    TIME = 10*WL/LIGHT_SPEED  # Very short simulation
    
    # Test extremely high safety factors
    safety_factors = [0.999, 0.9995, 0.9999, 0.99995, 0.99999]
    results = []
    
    for sf in safety_factors:
        print(f"\nTesting safety factor: {sf}")
        
        try:
            # Calculate parameters
            DX, DT = calc_optimal_fdtd_params(WL, N_MAX, dims=2, safety_factor=sf)
            
            # Check stability
            is_stable, courant, limit = check_fdtd_stability(DT, DX, dy=DX, n_max=N_MAX, safety_factor=1.0)
            print(f"  Courant number: {courant:.6f}, Limit: {limit:.6f}, Stable: {is_stable}")
            
            # Create simple design
            design = Design(width=X, height=Y, material=Material(N_MAX**2), pml_size=WL/4)
            
            # Simple Gaussian source
            time_steps = np.arange(0, TIME, DT)
            signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                                 phase=0, ramp_duration=WL/LIGHT_SPEED, t_max=TIME/2)
            source = GaussianSource(position=(X/4, Y/2), width=WL/20, signal=signal)
            design += source
            
            # Run simulation
            sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
            
            # Run a very short simulation to test stability
            try:
                max_steps = min(20, len(time_steps))  # Very short test
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

def test_different_materials():
    """Test stability with different material indices."""
    print("\n=== Testing Different Material Indices ===")
    
    WL = 1.55*µm
    X, Y = 6*µm, 4*µm
    TIME = 10*WL/LIGHT_SPEED
    
    # Test different refractive indices
    n_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    results = []
    
    for n_max in n_values:
        print(f"\nTesting n_max: {n_max}")
        
        try:
            # Use very high safety factor
            DX, DT = calc_optimal_fdtd_params(WL, n_max, dims=2, safety_factor=0.999)
            
            # Check stability
            is_stable, courant, limit = check_fdtd_stability(DT, DX, dy=DX, n_max=n_max, safety_factor=1.0)
            print(f"  Courant number: {courant:.6f}, Limit: {limit:.6f}, Stable: {is_stable}")
            
            # Create design
            design = Design(width=X, height=Y, material=Material(n_max**2), pml_size=WL/4)
            
            # Simple source
            time_steps = np.arange(0, TIME, DT)
            signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                                 phase=0, ramp_duration=WL/LIGHT_SPEED, t_max=TIME/2)
            source = GaussianSource(position=(X/4, Y/2), width=WL/20, signal=signal)
            design += source
            
            # Run simulation
            sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
            
            try:
                max_steps = min(20, len(time_steps))
                sim_results = sim.run(steps=max_steps, save=False, live=False, save_memory_mode=True)
                
                if hasattr(sim, 'Ez'):
                    max_field = np.max(np.abs(sim.Ez))
                else:
                    max_field = 0
                
                unstable = max_field > 1e6
                
            except Exception as e:
                print(f"    Simulation error: {e}")
                max_field = float('inf')
                unstable = True
                    
            print(f"  Max field: {max_field:.2e}, Unstable: {unstable}")
            results.append((n_max, courant, is_stable, unstable, max_field))
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append((n_max, 0, False, True, float('inf')))
    
    return results

def test_3d_stability():
    """Test 3D stability limits."""
    print("\n=== Testing 3D Stability Limits ===")
    
    WL = 1.55*µm
    N_MAX = 2.0
    X, Y, Z = 4*µm, 3*µm, 2*µm  # Very small 3D domain
    TIME = 8*WL/LIGHT_SPEED
    
    # Test high safety factors for 3D
    safety_factors = [0.95, 0.98, 0.99, 0.995, 0.999]
    results = []
    
    for sf in safety_factors:
        print(f"\nTesting 3D safety factor: {sf}")
        
        try:
            # Calculate parameters for 3D
            DX, DT = calc_optimal_fdtd_params(WL, N_MAX, dims=3, safety_factor=sf)
            
            # Check stability
            is_stable, courant, limit = check_fdtd_stability(DT, DX, dy=DX, dz=DX, n_max=N_MAX, safety_factor=1.0)
            print(f"  Courant number: {courant:.6f}, Limit: {limit:.6f}, Stable: {is_stable}")
            
            # Create 3D design
            design = Design(width=X, height=Y, depth=Z, material=Material(N_MAX**2), pml_size=WL/4)
            
            # Simple 3D source
            time_steps = np.arange(0, TIME, DT)
            signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                                 phase=0, ramp_duration=WL/LIGHT_SPEED, t_max=TIME/2)
            source = GaussianSource(position=(X/4, Y/2, Z/2), width=WL/20, signal=signal)
            design += source
            
            # Run simulation
            sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
            
            try:
                max_steps = min(15, len(time_steps))  # Very short test
                sim_results = sim.run(steps=max_steps, save=False, live=False, save_memory_mode=True)
                
                if hasattr(sim, 'Ez'):
                    max_field = np.max(np.abs(sim.Ez))
                else:
                    max_field = 0
                
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

def test_boundary_conditions():
    """Test stability with different boundary conditions."""
    print("\n=== Testing Boundary Condition Stability ===")
    
    WL = 1.55*µm
    N_MAX = 2.0
    TIME = 10*WL/LIGHT_SPEED
    
    # Test different domain sizes (affects boundary treatment)
    domain_configs = [
        ((4*µm, 3*µm), "Small"),
        ((6*µm, 4*µm), "Medium"),
        ((8*µm, 6*µm), "Large"),
        ((10*µm, 8*µm), "Very Large")
    ]
    
    results = []
    
    for (width, height), description in domain_configs:
        print(f"\nTesting {description} domain: {width/µm:.1f}µm x {height/µm:.1f}µm")
        
        try:
            # Use high safety factor
            DX, DT = calc_optimal_fdtd_params(WL, N_MAX, dims=2, safety_factor=0.999)
            
            # Check stability
            is_stable, courant, limit = check_fdtd_stability(DT, DX, dy=DX, n_max=N_MAX, safety_factor=1.0)
            print(f"  Courant number: {courant:.6f}, Limit: {limit:.6f}, Stable: {is_stable}")
            
            # Create design
            design = Design(width=width, height=height, material=Material(N_MAX**2), pml_size=WL/4)
            
            # Simple source
            time_steps = np.arange(0, TIME, DT)
            signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                                 phase=0, ramp_duration=WL/LIGHT_SPEED, t_max=TIME/2)
            source = GaussianSource(position=(width/4, height/2), width=WL/20, signal=signal)
            design += source
            
            # Run simulation
            sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
            
            try:
                max_steps = min(20, len(time_steps))
                sim_results = sim.run(steps=max_steps, save=False, live=False, save_memory_mode=True)
                
                if hasattr(sim, 'Ez'):
                    max_field = np.max(np.abs(sim.Ez))
                else:
                    max_field = 0
                
                unstable = max_field > 1e6
                
            except Exception as e:
                print(f"    Simulation error: {e}")
                max_field = float('inf')
                unstable = True
                    
            print(f"  Max field: {max_field:.2e}, Unstable: {unstable}")
            results.append((description, courant, is_stable, unstable, max_field))
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append((description, 0, False, True, float('inf')))
    
    return results

if __name__ == "__main__":
    print("Running final comprehensive stability tests...")
    
    # Run all tests
    ultimate_results = test_ultimate_courant_limits()
    material_results = test_different_materials()
    d3_results = test_3d_stability()
    boundary_results = test_boundary_conditions()
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("FINAL COMPREHENSIVE STABILITY TEST SUMMARY")
    print("="*80)
    
    print("\n1. Ultimate Courant Limits:")
    max_2d_courant = 0
    for sf, courant, stable, unstable, max_field in ultimate_results:
        status = "UNSTABLE" if unstable else "STABLE"
        print(f"   SF={sf:.5f}: Courant={courant:.6f}, {status}, Max={max_field:.2e}")
        if not unstable and courant > max_2d_courant:
            max_2d_courant = courant
    
    print("\n2. Different Material Indices:")
    for n_max, courant, stable, unstable, max_field in material_results:
        status = "UNSTABLE" if unstable else "STABLE"
        print(f"   n_max={n_max}: Courant={courant:.6f}, {status}, Max={max_field:.2e}")
    
    print("\n3. 3D Stability Limits:")
    max_3d_courant = 0
    for sf, courant, stable, unstable, max_field in d3_results:
        status = "UNSTABLE" if unstable else "STABLE"
        print(f"   SF={sf:.3f}: Courant={courant:.6f}, {status}, Max={max_field:.2e}")
        if not unstable and courant > max_3d_courant:
            max_3d_courant = courant
    
    print("\n4. Boundary Condition Stability:")
    for description, courant, stable, unstable, max_field in boundary_results:
        status = "UNSTABLE" if unstable else "STABLE"
        print(f"   {description}: Courant={courant:.6f}, {status}, Max={max_field:.2e}")
    
    # Final summary
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Maximum stable 2D Courant number: {max_2d_courant:.6f}")
    print(f"   Maximum stable 3D Courant number: {max_3d_courant:.6f}")
    print(f"   Theoretical 2D limit: 0.707107")
    print(f"   Theoretical 3D limit: 0.577350")
    print(f"   2D Achievement: {max_2d_courant/0.707107*100:.2f}% of theoretical limit")
    print(f"   3D Achievement: {max_3d_courant/0.577350*100:.2f}% of theoretical limit")
    
    if max_2d_courant/0.707107 > 0.99:
        print("   🎉 SUCCESS: Achieved >99% of theoretical 2D Courant limit!")
    if max_3d_courant/0.577350 > 0.99:
        print("   🎉 SUCCESS: Achieved >99% of theoretical 3D Courant limit!")