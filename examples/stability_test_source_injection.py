#!/usr/bin/env python3
"""
Detailed test of source injection methods and their impact on FDTD stability.
This test focuses specifically on how different source injection techniques
affect the stability of the simulation.
"""

from beamz import *
import numpy as np
import matplotlib.pyplot as plt

def test_hard_vs_soft_source_injection():
    """Compare hard source injection vs soft source injection."""
    print("=== Testing Hard vs Soft Source Injection ===")
    
    WL = 1.55*µm
    N_MAX = 2.0
    X, Y = 15*µm, 10*µm
    TIME = 30*WL/LIGHT_SPEED
    
    # Use a moderate safety factor
    DX, DT = calc_optimal_fdtd_params(WL, N_MAX, dims=2, safety_factor=0.8)
    time_steps = np.arange(0, TIME, DT)
    
    # Create signal
    signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                          phase=0, ramp_duration=WL*5/LIGHT_SPEED, t_max=TIME/2)
    
    # Test 1: Current implementation (hard injection with boundary conditions)
    print("\nTest 1: Current Implementation (Hard Injection)")
    design1 = Design(width=X, height=Y, material=Material(N_MAX**2), pml_size=WL)
    design1 += Rectangle(position=(0, Y/2-WL/4), width=X/3, height=WL/2, material=Material(N_MAX**2))
    
    source1 = ModeSource(design=design1, start=(X/6, Y/2-WL/4), end=(X/6, Y/2+WL/4), 
                        wavelength=WL, signal=signal, direction="+x")
    design1 += source1
    
    sim1 = FDTD(design=design1, time=time_steps, mesh="regular", resolution=DX)
    results1 = run_detailed_stability_test(sim1, "Hard Injection")
    
    # Test 2: Soft injection (additive only, no boundary conditions)
    print("\nTest 2: Soft Injection (Additive Only)")
    design2 = Design(width=X, height=Y, material=Material(N_MAX**2), pml_size=WL)
    design2 += Rectangle(position=(0, Y/2-WL/4), width=X/3, height=WL/2, material=Material(N_MAX**2))
    
    source2 = ModeSource(design=design2, start=(X/6, Y/2-WL/4), end=(X/6, Y/2+WL/4), 
                        wavelength=WL, signal=signal, direction="+x")
    design2 += source2
    
    sim2 = FDTD(design=design2, time=time_steps, mesh="regular", resolution=DX)
    # Modify source application to be soft
    sim2._apply_sources_soft = True
    results2 = run_detailed_stability_test(sim2, "Soft Injection")
    
    return results1, results2

def test_source_positioning():
    """Test how source positioning affects stability."""
    print("\n=== Testing Source Positioning ===")
    
    WL = 1.55*µm
    N_MAX = 2.0
    X, Y = 15*µm, 10*µm
    TIME = 25*WL/LIGHT_SPEED
    
    DX, DT = calc_optimal_fdtd_params(WL, N_MAX, dims=2, safety_factor=0.8)
    time_steps = np.arange(0, TIME, DT)
    
    signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                          phase=0, ramp_duration=WL*4/LIGHT_SPEED, t_max=TIME/2)
    
    positions = [
        (X/8, Y/2, "Near PML"),
        (X/4, Y/2, "Quarter domain"),
        (X/2, Y/2, "Center"),
        (3*X/4, Y/2, "Three quarters")
    ]
    
    results = []
    for pos_x, pos_y, description in positions:
        print(f"\nTesting position: {description} at ({pos_x/µm:.1f}µm, {pos_y/µm:.1f}µm)")
        
        design = Design(width=X, height=Y, material=Material(N_MAX**2), pml_size=WL)
        design += Rectangle(position=(0, Y/2-WL/4), width=X/3, height=WL/2, material=Material(N_MAX**2))
        
        source = ModeSource(design=design, start=(pos_x-WL/4, pos_y-WL/4), 
                          end=(pos_x-WL/4, pos_y+WL/4), wavelength=WL, signal=signal, direction="+x")
        design += source
        
        sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
        result = run_detailed_stability_test(sim, description)
        results.append((description, result))
    
    return results

def test_source_amplitude_scaling():
    """Test how source amplitude affects stability."""
    print("\n=== Testing Source Amplitude Scaling ===")
    
    WL = 1.55*µm
    N_MAX = 2.0
    X, Y = 12*µm, 8*µm
    TIME = 20*WL/LIGHT_SPEED
    
    DX, DT = calc_optimal_fdtd_params(WL, N_MAX, dims=2, safety_factor=0.8)
    time_steps = np.arange(0, TIME, DT)
    
    amplitudes = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    results = []
    
    for amp in amplitudes:
        print(f"\nTesting amplitude: {amp}")
        
        signal = ramped_cosine(time_steps, amplitude=amp, frequency=LIGHT_SPEED/WL, 
                              phase=0, ramp_duration=WL*3/LIGHT_SPEED, t_max=TIME/2)
        
        design = Design(width=X, height=Y, material=Material(N_MAX**2), pml_size=WL)
        design += Rectangle(position=(0, Y/2-WL/4), width=X/3, height=WL/2, material=Material(N_MAX**2))
        
        source = ModeSource(design=design, start=(X/6, Y/2-WL/4), end=(X/6, Y/2+WL/4), 
                          wavelength=WL, signal=signal, direction="+x")
        design += source
        
        sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
        result = run_detailed_stability_test(sim, f"Amplitude {amp}")
        results.append((amp, result))
    
    return results

def run_detailed_stability_test(sim, test_name):
    """Run a detailed stability test with field monitoring."""
    max_field = 0
    field_history = []
    unstable = False
    instability_step = -1
    
    for i in range(min(300, len(sim.time_steps))):
        sim.simulate_step()
        current_max = np.max(np.abs(sim.Ez))
        max_field = max(max_field, current_max)
        field_history.append(current_max)
        
        # Check for exponential growth (instability)
        if current_max > 1e6:
            unstable = True
            instability_step = i
            break
        
        # Check for rapid growth
        if i > 10 and current_max > max_field * 10:
            unstable = True
            instability_step = i
            break
    
    # Calculate growth rate
    if len(field_history) > 50:
        early_max = max(field_history[:50])
        late_max = max(field_history[-50:])
        growth_rate = late_max / early_max if early_max > 0 else 1.0
    else:
        growth_rate = 1.0
    
    print(f"  {test_name}: Max={max_field:.2e}, Growth={growth_rate:.2f}, Unstable={unstable}")
    if unstable:
        print(f"    Instability detected at step {instability_step}")
    
    return {
        'max_field': max_field,
        'growth_rate': growth_rate,
        'unstable': unstable,
        'instability_step': instability_step,
        'field_history': field_history
    }

def test_mode_source_normalization():
    """Test if mode source normalization affects stability."""
    print("\n=== Testing Mode Source Normalization ===")
    
    WL = 1.55*µm
    N_MAX = 2.0
    X, Y = 12*µm, 8*µm
    TIME = 20*WL/LIGHT_SPEED
    
    DX, DT = calc_optimal_fdtd_params(WL, N_MAX, dims=2, safety_factor=0.8)
    time_steps = np.arange(0, TIME, DT)
    
    signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                          phase=0, ramp_duration=WL*3/LIGHT_SPEED, t_max=TIME/2)
    
    # Test with different mode solver settings
    mode_solvers = ["num_eigen", "analytical"]
    results = []
    
    for solver in mode_solvers:
        print(f"\nTesting mode solver: {solver}")
        
        design = Design(width=X, height=Y, material=Material(N_MAX**2), pml_size=WL)
        design += Rectangle(position=(0, Y/2-WL/4), width=X/3, height=WL/2, material=Material(N_MAX**2))
        
        try:
            source = ModeSource(design=design, start=(X/6, Y/2-WL/4), end=(X/6, Y/2+WL/4), 
                              wavelength=WL, signal=signal, direction="+x", mode_solver=solver)
            design += source
            
            sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
            result = run_detailed_stability_test(sim, f"Mode solver {solver}")
            results.append((solver, result))
            
        except Exception as e:
            print(f"  Error with {solver}: {e}")
            results.append((solver, {'unstable': True, 'error': str(e)}))
    
    return results

if __name__ == "__main__":
    print("Running comprehensive source injection stability tests...")
    
    # Run all tests
    hard_soft_results = test_hard_vs_soft_source_injection()
    position_results = test_source_positioning()
    amplitude_results = test_source_amplitude_scaling()
    normalization_results = test_mode_source_normalization()
    
    # Print comprehensive summary
    print("\n" + "="*60)
    print("COMPREHENSIVE STABILITY TEST SUMMARY")
    print("="*60)
    
    print("\n1. Hard vs Soft Source Injection:")
    for name, result in zip(["Hard", "Soft"], hard_soft_results):
        if isinstance(result, dict):
            print(f"   {name}: Max={result['max_field']:.2e}, Unstable={result['unstable']}")
    
    print("\n2. Source Positioning:")
    for desc, result in position_results:
        if isinstance(result, dict):
            print(f"   {desc}: Max={result['max_field']:.2e}, Unstable={result['unstable']}")
    
    print("\n3. Source Amplitude:")
    for amp, result in amplitude_results:
        if isinstance(result, dict):
            print(f"   Amplitude {amp}: Max={result['max_field']:.2e}, Unstable={result['unstable']}")
    
    print("\n4. Mode Source Normalization:")
    for solver, result in normalization_results:
        if isinstance(result, dict):
            if 'error' in result:
                print(f"   {solver}: Error - {result['error']}")
            else:
                print(f"   {solver}: Max={result['max_field']:.2e}, Unstable={result['unstable']}")