# MMI Coupler Simulation

This tutorial demonstrates how to simulate a Multi-Mode Interference (MMI) coupler using BEAMZ. MMI couplers are important components in integrated photonics for power splitting and combining.

## Overview

In this tutorial, you will learn:

- How to create an MMI coupler structure
- How to simulate light propagation in multi-mode regions
- How to analyze power splitting
- How to visualize the results

## Code Example

```python
from beamz import *
import numpy as np

# Parameters
X = 20*µm  # domain width
Y = 10*µm  # domain height
WL = 1.55*µm  # wavelength
TIME = 40*WL/LIGHT_SPEED  # simulation duration
N_CORE = 2.04  # Si3N4 refractive index
N_CLAD = 1.444  # SiO2 refractive index
WG_W = 0.565*µm  # width of the waveguide
H = 3.5*µm  # height of the MMI
W = 9*µm  # length of the MMI
OFFSET = 1.05*µm  # offset of the output waveguides
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=2, safety_factor=0.60)

# Design the MMI with input and output waveguides
design = Design(width=X, height=Y, material=Material(N_CLAD**2), pml_size=WL)
# Input waveguide
design += Rectangle(position=(0, Y/2-WG_W/2), width=X/2, height=WG_W, 
                   material=Material(N_CORE**2))
# Output waveguides
design += Rectangle(position=(X/2, Y/2 + OFFSET - WG_W/2), width=X/2, height=WG_W, 
                   material=Material(N_CORE**2))
design += Rectangle(position=(X/2, Y/2 - OFFSET - WG_W/2), width=X/2, height=WG_W, 
                   material=Material(N_CORE**2))
# MMI section
design += Rectangle(position=(X/2-W/2, Y/2-H/2), width=W, height=H, 
                   material=Material(N_CORE**2))

# Define the source
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                      phase=0, ramp_duration=WL*5/LIGHT_SPEED, t_max=TIME/2)
source = ModeSource(design=design, start=(2*µm, Y/2-1.2*µm), 
                   end=(2*µm, Y/2+1.2*µm), wavelength=WL, signal=signal)
design += source
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
```
We import the necessary libraries for the simulation.

### 2. Define Simulation Parameters
```python
X = 20*µm  # domain width
Y = 10*µm  # domain height
WL = 1.55*µm  # wavelength
TIME = 40*WL/LIGHT_SPEED  # simulation duration
N_CORE = 2.04  # Si3N4 refractive index
N_CLAD = 1.444  # SiO2 refractive index
WG_W = 0.565*µm  # waveguide width
H = 3.5*µm  # MMI height
W = 9*µm  # MMI length
OFFSET = 1.05*µm  # output waveguide offset
```
These parameters define the MMI coupler properties and simulation settings.

### 3. Create the Design
```python
design = Design(width=X, height=Y, material=Material(N_CLAD**2), pml_size=WL)
# Input waveguide
design += Rectangle(position=(0, Y/2-WG_W/2), width=X/2, height=WG_W, 
                   material=Material(N_CORE**2))
# Output waveguides
design += Rectangle(position=(X/2, Y/2 + OFFSET - WG_W/2), width=X/2, height=WG_W, 
                   material=Material(N_CORE**2))
design += Rectangle(position=(X/2, Y/2 - OFFSET - WG_W/2), width=X/2, height=WG_W, 
                   material=Material(N_CORE**2))
# MMI section
design += Rectangle(position=(X/2-W/2, Y/2-H/2), width=W, height=H, 
                   material=Material(N_CORE**2))
```
We create the MMI coupler structure with input and output waveguides.

### 4. Define the Source
```python
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, 
                      phase=0, ramp_duration=WL*5/LIGHT_SPEED, t_max=TIME/2)
source = ModeSource(design=design, start=(2*µm, Y/2-1.2*µm), 
                   end=(2*µm, Y/2+1.2*µm), wavelength=WL, signal=signal)
design += source
```
We create a mode source to excite the input waveguide.

### 5. Run the Simulation
```python
sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
sim.run(live=True, save_memory_mode=True, accumulate_power=True)
sim.plot_power(db_colorbar=True)
```
We run the FDTD simulation and visualize the results.