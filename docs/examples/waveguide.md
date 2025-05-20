# Waveguide Simulation

This tutorial demonstrates how to simulate light propagation in a simple waveguide using BEAMZ. We'll create a straight waveguide and observe how light propagates through it.

## Overview

In this tutorial, you will learn:

- How to set up a waveguide simulation
- How to use mode sources
- How to configure simulation parameters for waveguide analysis
- How to visualize the results

## Code Example

```python
from beamz import *
import numpy as np
from beamz.helpers import calc_optimal_fdtd_params

# Define simulation parameters
WL = 1.55*µm  # wavelength
TIME = 90*WL/LIGHT_SPEED  # simulation duration
N_CORE = 2.04  # Si3N4 refractive index
N_CLAD = 1.444  # SiO2 refractive index
WG_WIDTH = 0.565*µm  # waveguide width
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD))

# Create the design
design = Design(width=18*µm, height=7*µm, material=Material(N_CLAD**2), pml_size=WL)
design += Rectangle(position=(0,3.5*µm-WG_WIDTH/2), width=18*µm, height=WG_WIDTH, 
                   material=Material(N_CORE**2))

# Create the signal
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                      phase=0, ramp_duration=WL*30/LIGHT_SPEED, t_max=TIME/2)
design += ModeSource(design=design, start=(2*µm, 3.5*µm-1.2*µm), 
                    end=(2*µm, 3.5*µm+1.2*µm), wavelength=WL, signal=signal)
design.show()

# Run the simulation
sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
sim.run(live=True, save_memory_mode=True, accumulate_power=True)
sim.plot_power(db_colorbar=True)
```

## Step-by-Step Explanation

### 1. Import Required Libraries
```python
from beamz import *
import numpy as np
from beamz.helpers import calc_optimal_fdtd_params
```
We import the necessary libraries for the simulation.

### 2. Define Simulation Parameters
```python
WL = 1.55*µm  # wavelength
TIME = 90*WL/LIGHT_SPEED  # simulation duration
N_CORE = 2.04  # Si3N4 refractive index
N_CLAD = 1.444  # SiO2 refractive index
WG_WIDTH = 0.565*µm  # waveguide width
```
These parameters define the waveguide properties and simulation settings.

### 3. Create the Design
```python
design = Design(width=18*µm, height=7*µm, material=Material(N_CLAD**2), pml_size=WL)
design += Rectangle(position=(0,3.5*µm-WG_WIDTH/2), width=18*µm, height=WG_WIDTH, 
                   material=Material(N_CORE**2))
```
We create a straight waveguide using a rectangle with specific dimensions.

### 4. Define the Source
```python
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                      phase=0, ramp_duration=WL*30/LIGHT_SPEED, t_max=TIME/2)
design += ModeSource(design=design, start=(2*µm, 3.5*µm-1.2*µm), 
                    end=(2*µm, 3.5*µm+1.2*µm), wavelength=WL, signal=signal)
```
We create a mode source to excite the waveguide.

### 5. Run the Simulation
```python
sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
sim.run(live=True, save_memory_mode=True, accumulate_power=True)
sim.plot_power(db_colorbar=True)
```
We run the FDTD simulation and visualize the results.