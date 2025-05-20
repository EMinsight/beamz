# Basic Dipole Simulation

This tutorial demonstrates how to create a simple dipole simulation using BEAMZ. We'll simulate light propagation from a point source in a dielectric medium.

## Full Code Example

You heard that right. This is it! This is all it takes!

```python
from beamz import *
import numpy as np

# Define simulation parameters
WL = 0.6*µm  # wavelength of the source
TIME = 40*WL/LIGHT_SPEED  # total simulation duration
N_CLAD = 1  # refractive index of cladding
N_CORE = 2  # refractive index of core
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD))

# Create the design
design = Design(8*µm, 8*µm, material=Material(N_CLAD**2), pml_size=WL*1.5)
design += Rectangle(width=4*µm, height=4*µm, material=Material(N_CORE**2))

# Define the signal
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                      phase=0, ramp_duration=3*WL/LIGHT_SPEED, t_max=TIME/2)
design += GaussianSource(position=(4*µm, 5*µm), width=WL/6, signal=signal)

# Run the simulation
sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
sim.run(live=True, save_memory_mode=True, accumulate_power=True)
sim.plot_power(db_colorbar=True)
```

## Step-by-Step Explanation

But just in case, let's walk through it with some additional details.

### 1. Import Required Libraries
```python
from beamz import *
import numpy as np
```
We import the BEAMZ library and NumPy for numerical operations.

### 2. Define Simulation Parameters
```python
WL = 0.6*µm  # wavelength
TIME = 40*WL/LIGHT_SPEED  # simulation duration
N_CLAD = 1  # cladding refractive index
N_CORE = 2  # core refractive index
```
These parameters define the basic properties of our simulation.

### 3. Create the Design
```python
design = Design(8*µm, 8*µm, material=Material(N_CLAD**2), pml_size=WL*1.5)
design += Rectangle(width=4*µm, height=4*µm, material=Material(N_CORE**2))
```
We create a simulation domain with a rectangular dielectric region.

### 4. Define the Source
```python
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                      phase=0, ramp_duration=3*WL/LIGHT_SPEED, t_max=TIME/2)
design += GaussianSource(position=(4*µm, 5*µm), width=WL/6, signal=signal)
```
We create a Gaussian source with a ramped cosine signal.

### 5. Run the Simulation
```python
sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
sim.run(live=True, axis_scale=[-1, 1], save_memory_mode=True, accumulate_power=True)
sim.plot_power(db_colorbar=True)
```
We run the FDTD simulation and visualize the results.