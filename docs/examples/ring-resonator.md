# Ring Resonator Simulation

This tutorial demonstrates how to simulate a ring resonator using BEAMZ. Ring resonators are important components in integrated photonics for filtering and sensing applications.

## Overview

In this tutorial, you will learn:

- How to create a ring resonator structure
- How to simulate light coupling between a waveguide and a ring
- How to analyze resonance effects
- How to visualize the results

## Code Example

```python
from beamz import *
import numpy as np

# Parameters
WL = 1.55*µm  # wavelength
TIME = 120*WL/LIGHT_SPEED  # simulation duration
X = 20*µm  # domain width
Y = 19*µm  # domain height
N_CORE = 2.04  # Si3N4 refractive index
N_CLAD = 1.444  # SiO2 refractive index
WG_WIDTH = 0.565*µm  # waveguide width
RING_RADIUS = 6*µm  # ring radius
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD))

# Create the design
design = Design(width=X, height=Y, material=Material(N_CLAD**2), pml_size=WL)
# Create the bus waveguide
design += Rectangle(position=(0,WL*2), width=X, height=WG_WIDTH, 
                   material=Material(N_CORE**2))
# Create the ring resonator
design += Ring(position=(X/2, WL*2+WG_WIDTH+RING_RADIUS+WG_WIDTH/2+0.2*WG_WIDTH), 
              inner_radius=RING_RADIUS-WG_WIDTH/2, 
              outer_radius=RING_RADIUS+WG_WIDTH/2, 
              material=Material(N_CORE**2))

# Define the signal & source
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                      phase=0, ramp_duration=WL*20/LIGHT_SPEED, t_max=TIME/3)
design += ModeSource(design=design, 
                    start=(WL*2, WL*2+WG_WIDTH/2-1.5*µm), 
                    end=(WL*2, WL*2+WG_WIDTH/2+1.5*µm), 
                    wavelength=WL, signal=signal)
design.show()

# Run the simulation
sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX, backend="numpy")
sim.run(live=True, save_memory_mode=True, accumulate_power=True)
sim.plot_power(db_colorbar=True)
```

## Step-by-Step Explanation

### 1. Import Required Libraries
```python
from beamz import *
import numpy as np
```
We import the necessary libraries for the simulation.

### 2. Define Simulation Parameters
```python
WL = 1.55*µm  # wavelength
TIME = 120*WL/LIGHT_SPEED  # simulation duration
X = 20*µm  # domain width
Y = 19*µm  # domain height
N_CORE = 2.04  # Si3N4 refractive index
N_CLAD = 1.444  # SiO2 refractive index
WG_WIDTH = 0.565*µm  # waveguide width
RING_RADIUS = 6*µm  # ring radius
```
These parameters define the ring resonator properties and simulation settings.

### 3. Create the Design
```python
design = Design(width=X, height=Y, material=Material(N_CLAD**2), pml_size=WL)
# Create the bus waveguide
design += Rectangle(position=(0,WL*2), width=X, height=WG_WIDTH, 
                   material=Material(N_CORE**2))
# Create the ring resonator
design += Ring(position=(X/2, WL*2+WG_WIDTH+RING_RADIUS+WG_WIDTH/2+0.2*WG_WIDTH), 
              inner_radius=RING_RADIUS-WG_WIDTH/2, 
              outer_radius=RING_RADIUS+WG_WIDTH/2, 
              material=Material(N_CORE**2))
```
We create a bus waveguide and a ring resonator with specific dimensions.

### 4. Define the Source
```python
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                      phase=0, ramp_duration=WL*20/LIGHT_SPEED, t_max=TIME/3)
design += ModeSource(design=design, 
                    start=(WL*2, WL*2+WG_WIDTH/2-1.5*µm), 
                    end=(WL*2, WL*2+WG_WIDTH/2+1.5*µm), 
                    wavelength=WL, signal=signal)
```
We create a mode source to excite the bus waveguide.

### 5. Run the Simulation
```python
sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX, backend="numpy")
sim.run(live=True, save_memory_mode=True, accumulate_power=True)
sim.plot_power(db_colorbar=True)
```
We run the FDTD simulation and visualize the results.