#!/usr/bin/env python3
"""
Test field update algorithms and their impact on FDTD stability.
This test focuses on the core Maxwell equation updates and PML implementation.
"""

from beamz import *
import numpy as np
import matplotlib.pyplot as plt

def test_pml_implementation():
    """Test PML implementation for stability issues."""
    print("=== Testing PML Implementation ===")
    
    WL = 1.55*µm
    N_MAX = 2.0
    X, Y = 15*µm, 10*µm
    TIME = 25*WL/LIGHT_SPEED
    
    # Test different PML sizes
    pml_sizes = [WL/2, WL, 2*WL, 3*WL]
    results = []
    
    for pml_size in pml_sizes:
        print(f"\nTesting PML size: {pml_size/µm:.1f}µm")
        
        DX, DT = calc_optimal_fdtd_params(WL, N_MAX, dims=2, safety_factor=0.8)
        time_steps = np.arange(0, TIME, DT)
        
        signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                              phase=0, ramp_duration=WL*4/LIGHT_SPEED, t_max=TIME/2)
        
        design = Design(width=X, height=Y, material=Material(N_MAX**2), pml_size=pml_size)
        design += Rectangle(position=(0, Y/2-WL/4), width=X/3, height=WL/2, material=Material(N_MAX**2))
        
        source = ModeSource(design=design, start=(X/6, Y/2-WL/4), end=(X/6, Y/2+WL/4), 
                          wavelength=WL, signal=signal, direction="+x")
        design += source
        
        sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
        result = run_field_update_test(sim, f"PML {pml_size/µm:.1f}µm")
        results.append((pml_size, result))
    
    return results

def test_backend_comparison():
    """Compare different backends for stability."""
    print("\n=== Testing Backend Comparison ===")
    
    WL = 1.55*µm
    N_MAX = 2.0
    X, Y = 12*µm, 8*µm
    TIME = 20*WL/LIGHT_SPEED
    
    backends = ["numpy", "jax", "torch"]
    results = []
    
    for backend in backends:
        print(f"\nTesting backend: {backend}")
        
        try:
            DX, DT = calc_optimal_fdtd_params(WL, N_MAX, dims=2, safety_factor=0.8)
            time_steps = np.arange(0, TIME, DT)
            
            signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                                  phase=0, ramp_duration=WL*3/LIGHT_SPEED, t_max=TIME/2)
            
            design = Design(width=X, height=Y, material=Material(N_MAX**2), pml_size=WL)
            design += Rectangle(position=(0, Y/2-WL/4), width=X/3, height=WL/2, material=Material(N_MAX**2))
            
            source = ModeSource(design=design, start=(X/6, Y/2-WL/4), end=(X/6, Y/2+WL/4), 
                              wavelength=WL, signal=signal, direction="+x")
            design += source
            
            sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX, backend=backend)
            result = run_field_update_test(sim, f"Backend {backend}")
            results.append((backend, result))
            
        except Exception as e:
            print(f"  Error with {backend}: {e}")
            results.append((backend, {'unstable': True, 'error': str(e)}))
    
    return results

def test_material_boundaries():
    """Test stability at material boundaries."""
    print("\n=== Testing Material Boundaries ===")
    
    WL = 1.55*µm
    N_CORE = 2.0
    N_CLAD = 1.444
    X, Y = 15*µm, 10*µm
    TIME = 25*WL/LIGHT_SPEED
    
    # Test different boundary sharpness
    boundary_tests = [
        ("Sharp", "Sharp material boundaries"),
        ("Gradient", "Gradual material transitions")
    ]
    
    results = []
    
    for boundary_type, description in boundary_tests:
        print(f"\nTesting {description}")
        
        DX, DT = calc_optimal_fdtd_params(WL, N_CORE, dims=2, safety_factor=0.8)
        time_steps = np.arange(0, TIME, DT)
        
        signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                              phase=0, ramp_duration=WL*4/LIGHT_SPEED, t_max=TIME/2)
        
        design = Design(width=X, height=Y, material=Material(N_CLAD**2), pml_size=WL)
        
        if boundary_type == "Sharp":
            # Sharp boundary
            design += Rectangle(position=(0, Y/2-WL/4), width=X/3, height=WL/2, material=Material(N_CORE**2))
        else:
            # Gradual boundary (simulated with multiple rectangles)
            for i in range(5):
                width = X/3 + i * WL/20
                height = WL/2 + i * WL/20
                pos_x = 0
                pos_y = Y/2 - height/2
                n_eff = N_CLAD + (N_CORE - N_CLAD) * i / 4
                design += Rectangle(position=(pos_x, pos_y), width=width, height=height, 
                                  material=Material(n_eff**2))
        
        source = ModeSource(design=design, start=(X/6, Y/2-WL/4), end=(X/6, Y/2+WL/4), 
                          wavelength=WL, signal=signal, direction="+x")
        design += source
        
        sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
        result = run_field_update_test(sim, description)
        results.append((boundary_type, result))
    
    return results

def test_yee_grid_consistency():
    """Test Yee grid implementation consistency."""
    print("\n=== Testing Yee Grid Consistency ===")
    
    WL = 1.55*µm
    N_MAX = 2.0
    X, Y = 12*µm, 8*µm
    TIME = 20*WL/LIGHT_SPEED
    
    # Test different resolutions
    resolutions = [WL/20, WL/15, WL/10, WL/8]
    results = []
    
    for res in resolutions:
        print(f"\nTesting resolution: {res/µm:.3f}µm")
        
        # Calculate time step for this resolution
        dt_max = res / (LIGHT_SPEED * np.sqrt(2))  # 2D
        dt = 0.8 * dt_max * N_MAX  # Apply safety factor
        
        time_steps = np.arange(0, TIME, dt)
        
        signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                              phase=0, ramp_duration=WL*3/LIGHT_SPEED, t_max=TIME/2)
        
        design = Design(width=X, height=Y, material=Material(N_MAX**2), pml_size=WL)
        design += Rectangle(position=(0, Y/2-WL/4), width=X/3, height=WL/2, material=Material(N_MAX**2))
        
        source = ModeSource(design=design, start=(X/6, Y/2-WL/4), end=(X/6, Y/2+WL/4), 
                          wavelength=WL, signal=signal, direction="+x")
        design += source
        
        sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=res)
        result = run_field_update_test(sim, f"Resolution {res/µm:.3f}µm")
        results.append((res, result))
    
    return results

def run_field_update_test(sim, test_name):
    """Run a field update stability test."""
    max_field = 0
    field_history = []
    energy_history = []
    unstable = False
    instability_step = -1
    
    for i in range(min(250, len(sim.time_steps))):
        sim.simulate_step()
        
        # Monitor field magnitudes
        current_max = np.max(np.abs(sim.Ez))
        max_field = max(max_field, current_max)
        field_history.append(current_max)
        
        # Monitor total energy (rough estimate)
        if hasattr(sim, 'Hx') and hasattr(sim, 'Hy'):
            h_energy = np.sum(np.abs(sim.Hx)**2) + np.sum(np.abs(sim.Hy)**2)
            e_energy = np.sum(np.abs(sim.Ez)**2)
            total_energy = h_energy + e_energy
            energy_history.append(total_energy)
        
        # Check for instability
        if current_max > 1e6:
            unstable = True
            instability_step = i
            break
        
        # Check for rapid growth
        if i > 20 and current_max > max_field * 5:
            unstable = True
            instability_step = i
            break
    
    # Calculate energy growth rate
    energy_growth = 1.0
    if len(energy_history) > 50:
        early_energy = np.mean(energy_history[:50])
        late_energy = np.mean(energy_history[-50:])
        energy_growth = late_energy / early_energy if early_energy > 0 else 1.0
    
    print(f"  {test_name}: Max={max_field:.2e}, Energy growth={energy_growth:.2f}, Unstable={unstable}")
    if unstable:
        print(f"    Instability at step {instability_step}")
    
    return {
        'max_field': max_field,
        'energy_growth': energy_growth,
        'unstable': unstable,
        'instability_step': instability_step,
        'field_history': field_history,
        'energy_history': energy_history
    }

def test_boundary_conditions():
    """Test boundary condition implementation."""
    print("\n=== Testing Boundary Conditions ===")
    
    WL = 1.55*µm
    N_MAX = 2.0
    X, Y = 12*µm, 8*µm
    TIME = 20*WL/LIGHT_SPEED
    
    # Test different domain sizes (affects boundary treatment)
    domain_sizes = [(10*µm, 6*µm), (15*µm, 10*µm), (20*µm, 12*µm)]
    results = []
    
    for width, height in domain_sizes:
        print(f"\nTesting domain: {width/µm:.1f}µm x {height/µm:.1f}µm")
        
        DX, DT = calc_optimal_fdtd_params(WL, N_MAX, dims=2, safety_factor=0.8)
        time_steps = np.arange(0, TIME, DT)
        
        signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                              phase=0, ramp_duration=WL*3/LIGHT_SPEED, t_max=TIME/2)
        
        design = Design(width=width, height=height, material=Material(N_MAX**2), pml_size=WL)
        design += Rectangle(position=(0, height/2-WL/4), width=width/3, height=WL/2, material=Material(N_MAX**2))
        
        source = ModeSource(design=design, start=(width/6, height/2-WL/4), end=(width/6, height/2+WL/4), 
                          wavelength=WL, signal=signal, direction="+x")
        design += source
        
        sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
        result = run_field_update_test(sim, f"Domain {width/µm:.1f}x{height/µm:.1f}")
        results.append(((width, height), result))
    
    return results

if __name__ == "__main__":
    print("Running comprehensive field update stability tests...")
    
    # Run all tests
    pml_results = test_pml_implementation()
    backend_results = test_backend_comparison()
    boundary_results = test_material_boundaries()
    yee_results = test_yee_grid_consistency()
    domain_results = test_boundary_conditions()
    
    # Print comprehensive summary
    print("\n" + "="*60)
    print("FIELD UPDATE STABILITY TEST SUMMARY")
    print("="*60)
    
    print("\n1. PML Implementation:")
    for pml_size, result in pml_results:
        if isinstance(result, dict):
            print(f"   PML {pml_size/µm:.1f}µm: Max={result['max_field']:.2e}, Unstable={result['unstable']}")
    
    print("\n2. Backend Comparison:")
    for backend, result in backend_results:
        if isinstance(result, dict):
            if 'error' in result:
                print(f"   {backend}: Error - {result['error']}")
            else:
                print(f"   {backend}: Max={result['max_field']:.2e}, Unstable={result['unstable']}")
    
    print("\n3. Material Boundaries:")
    for boundary_type, result in boundary_results:
        if isinstance(result, dict):
            print(f"   {boundary_type}: Max={result['max_field']:.2e}, Unstable={result['unstable']}")
    
    print("\n4. Yee Grid Consistency:")
    for res, result in yee_results:
        if isinstance(result, dict):
            print(f"   Resolution {res/µm:.3f}µm: Max={result['max_field']:.2e}, Unstable={result['unstable']}")
    
    print("\n5. Boundary Conditions:")
    for (width, height), result in domain_results:
        if isinstance(result, dict):
            print(f"   Domain {width/µm:.1f}x{height/µm:.1f}: Max={result['max_field']:.2e}, Unstable={result['unstable']}")