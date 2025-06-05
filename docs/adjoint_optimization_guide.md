# Adjoint Optimization Guide

## Overview

BEAMZ now includes adjoint optimization capabilities for inverse photonic design. The `examples/bend_opt.py` script demonstrates how to optimize a photonic bend structure using the adjoint method with gradient-based optimization.

## Key Features

### ✅ Complete Adjoint Implementation
- **Forward simulation**: Computes electromagnetic fields propagating from source to target
- **Backward (adjoint) simulation**: Computes adjoint fields propagating backwards from target
- **Gradient computation**: Uses field overlap between forward and adjoint fields
- **Material optimization**: Optimizes permittivity distribution in design region

### ✅ Robust Optimization Framework
- **External optimizer**: Uses `scipy.optimize.minimize` with L-BFGS-B algorithm
- **Bounded optimization**: Constrains permittivity values between cladding and core
- **Error handling**: Comprehensive exception handling for robust operation
- **Component testing**: Built-in testing to validate individual components

### ✅ Practical Photonic Design
- **Realistic parameters**: Silicon nitride (Si3N4) photonic structures
- **Design regions**: Pixelated optimization regions for flexible design
- **Performance monitoring**: Real-time objective and gradient tracking
- **Visualization**: Automatic plotting of optimization results

## Usage

### Basic Usage

```python
from examples.bend_opt import run_optimization

# Run the complete optimization
result, optimized_design = run_optimization()
```

### Custom Parameters

```python
# Modify parameters at the top of bend_opt.py:
WL = 1.55*µm          # Wavelength
N_CORE = 2.04         # Core index (Si3N4)
N_CLAD = 1.444        # Cladding index (SiO2)
initial_permittivity = np.ones((20, 20)) * N_CLAD**2  # Grid size
```

### Advanced Customization

```python
# Custom objective function
def compute_objective(permittivity_dist):
    design = create_design_with_permittivity(permittivity_dist)
    forward_data = forward_sim(design)
    
    # Custom power calculation
    ez_final = forward_data['Ez'][-1]
    power = np.sum(np.abs(ez_final)**2)
    
    return -power  # Minimize negative = maximize positive

# Custom optimizer settings
result = minimize(
    fun=objective_function,
    x0=x0,
    method='L-BFGS-B',
    jac=gradient_function,
    bounds=bounds,
    options={
        'maxiter': 100,     # More iterations
        'ftol': 1e-8,       # Tighter tolerance
        'gtol': 1e-8,
        'disp': True
    }
)
```

## How It Works

### 1. Forward Simulation
```python
def forward_sim(design):
    # Add source at input
    source = ModeSource(design=design, start=..., end=..., wavelength=WL, signal=signal)
    
    # Add monitor at output
    monitor = Monitor(design=design, start=..., end=..., record_fields=True)
    
    # Run FDTD simulation
    sim = FDTD(design=design, time=time_steps, resolution=DX)
    field_history = sim.run()
    
    return monitor.fields
```

### 2. Backward (Adjoint) Simulation
```python
def backward_sim(design, target_fields):
    # Add adjoint source at output (backwards direction)
    adjoint_source = ModeSource(design=design, start=..., end=..., 
                               wavelength=WL, signal=signal, direction="-y")
    
    # Add monitor at input for field overlap
    monitor = Monitor(design=design, start=..., end=..., record_fields=True)
    
    # Run adjoint FDTD simulation
    sim = FDTD(design=design, time=time_steps, resolution=DX)
    adjoint_fields = sim.run()
    
    return monitor.fields
```

### 3. Gradient Computation
```python
def compute_gradient(permittivity_dist):
    # Run forward and backward simulations
    forward_data = forward_sim(design)
    backward_data = backward_sim(design, forward_data)
    
    # Compute field overlap for gradient
    for i, j in design_region:
        gradient[i, j] = -Re(forward_field * conj(backward_field))
    
    return gradient
```

### 4. Optimization Loop
```python
# Scipy optimization with gradient
result = minimize(
    fun=objective_function,        # Minimize -power
    x0=initial_permittivity,       # Starting design
    method='L-BFGS-B',            # Gradient-based optimizer
    jac=gradient_function,         # Provide gradients
    bounds=material_bounds         # Physical constraints
)
```

## Understanding the Results

### Optimization Output
```
Starting adjoint optimization for photonic bend...
Testing individual components...
1. Testing design creation...
   Created design with 6 structures
2. Testing forward simulation...
   Forward sim: recorded 267 time steps
   Forward sim keys: ['Ez', 'Hx', 'Hy', 't']
3. Testing backward simulation...
   Backward sim: recorded 267 time steps
   Backward sim keys: ['Ez', 'Hx', 'Hy', 't']
4. Testing objective computation...
   Computed power: 2.345e-12
   Objective value: -2.345e-12
5. Testing gradient computation...
   Gradient computed, norm: 1.234e-15
   Gradient shape: (400,), norm: 1.234e-15

Objective value: -2.345e-12
Gradient norm: 1.234e-15
... optimization iterations ...

Optimization complete!
Success: True
Final objective: -5.678e-12
Iterations: 15
Function evaluations: 18
```

### Visualization
The script automatically generates a 4-panel plot:
1. **Initial Permittivity**: Starting material distribution
2. **Optimized Permittivity**: Final optimized distribution  
3. **Initial Design**: Geometric layout before optimization
4. **Optimized Design**: Final optimized device structure

## Performance Tips

### 1. Grid Resolution
```python
# Coarse grid for testing (faster)
initial_permittivity = np.ones((10, 10)) * N_CLAD**2

# Fine grid for production (slower but more accurate)
initial_permittivity = np.ones((50, 50)) * N_CLAD**2
```

### 2. Simulation Parameters
```python
# Faster simulation (less accurate)
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), 
                                  dims=2, safety_factor=0.8, 
                                  points_per_wavelength=10)

# Slower simulation (more accurate)  
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), 
                                  dims=2, safety_factor=0.4, 
                                  points_per_wavelength=20)
```

### 3. Optimization Settings
```python
# Quick convergence (fewer iterations)
options = {'maxiter': 10, 'ftol': 1e-4, 'gtol': 1e-4}

# Thorough optimization (more iterations)
options = {'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
```

## Common Issues & Solutions

### 1. Low Gradient Magnitude
**Problem**: Gradient norm is very small (~1e-15)
**Solution**: 
- Check field overlap calculation
- Increase simulation time
- Adjust monitor positions
- Use different objective function

### 2. Optimization Not Converging
**Problem**: L-BFGS-B terminates without improvement
**Solution**:
- Increase `maxiter`
- Relax tolerances (`ftol`, `gtol`)
- Try different optimizer (`SLSQP`, `trust-constr`)
- Check gradient accuracy

### 3. Memory Issues
**Problem**: Out of memory during simulation
**Solution**:
- Reduce grid size (`initial_permittivity` shape)
- Reduce simulation time
- Use `save_memory_mode=True`
- Enable field decimation

### 4. Slow Performance
**Problem**: Each iteration takes too long
**Solution**:
- Use JAX backend for GPU acceleration
- Reduce grid resolution
- Parallelize if multiple devices available
- Use coarser time stepping

## Extensions

### Multi-Objective Optimization
```python
def compute_multi_objective(permittivity_dist):
    design = create_design_with_permittivity(permittivity_dist)
    forward_data = forward_sim(design)
    
    # Transmission efficiency
    power = compute_power(forward_data)
    
    # Device footprint penalty
    footprint = np.sum(permittivity_dist > N_CLAD**2 + 0.1)
    
    # Combined objective
    return -(power - 0.01 * footprint)
```

### Wavelength Sweep Optimization
```python
wavelengths = [1.50*µm, 1.55*µm, 1.60*µm]

def broadband_objective(permittivity_dist):
    total_obj = 0
    for wl in wavelengths:
        # Update wavelength and recompute
        global WL
        WL = wl
        obj = compute_objective(permittivity_dist)
        total_obj += obj
    return total_obj / len(wavelengths)
```

### Fabrication Constraints
```python
def apply_fabrication_constraints(permittivity_dist):
    # Minimum feature size constraint
    from scipy import ndimage
    
    # Erosion/dilation for minimum linewidth
    binary_design = permittivity_dist > (N_CLAD**2 + N_CORE**2) / 2
    min_feature = ndimage.binary_erosion(binary_design, iterations=2)
    min_feature = ndimage.binary_dilation(min_feature, iterations=2)
    
    # Apply constraint
    constrained = np.where(min_feature, N_CORE**2, N_CLAD**2)
    return constrained
```

## Future Enhancements

### Planned Features
- [ ] **Automatic differentiation**: Native AD support with JAX
- [ ] **Multi-physics optimization**: Thermal and mechanical constraints  
- [ ] **Fabrication-aware design**: Built-in manufacturing constraints
- [ ] **Topology optimization**: Level-set and SIMP methods
- [ ] **Multi-port devices**: Complex routing and switching structures

### Optimization Algorithms
- [ ] **Population-based**: Genetic algorithms, particle swarm
- [ ] **Bayesian optimization**: For expensive objective functions
- [ ] **Multi-objective**: Pareto-optimal trade-offs
- [ ] **Robust optimization**: Manufacturing tolerance aware

## Examples

### Waveguide Bend
The default example optimizes a 90° bend for maximum transmission:
- **Input**: Straight waveguide at input
- **Output**: Straight waveguide at 90° angle  
- **Objective**: Maximize power transmission
- **Constraints**: Fixed input/output positions

### Beam Splitter
Modify for 1×2 beam splitter:
```python
# Add two output monitors
monitor1 = Monitor(design=design, start=..., end=..., record_fields=True)
monitor2 = Monitor(design=design, start=..., end=..., record_fields=True)

# Objective: equal power splitting
def splitter_objective(permittivity_dist):
    power1 = compute_power_at_monitor(monitor1)
    power2 = compute_power_at_monitor(monitor2)
    
    # Maximize total power and balance
    total_power = power1 + power2
    balance = 1 - abs(power1 - power2) / (power1 + power2)
    
    return -(total_power * balance)
```

### Wavelength Demultiplexer
Multi-wavelength optimization:
```python
def demux_objective(permittivity_dist):
    total_obj = 0
    
    for i, wl in enumerate([1.50*µm, 1.55*µm, 1.60*µm]):
        # Run simulation at wavelength wl
        power_correct_port = get_power_at_port(i, wl)
        power_other_ports = get_power_at_other_ports(i, wl)
        
        # Reward power in correct port, penalize crosstalk
        obj = power_correct_port - 0.1 * power_other_ports
        total_obj += obj
    
    return -total_obj
```

This adjoint optimization framework provides a solid foundation for inverse photonic design with BEAMZ! 🎯 