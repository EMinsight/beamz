from beamz import *
import numpy as np

WL = 1.55*µm
TIME = 20*WL/LIGHT_SPEED
T = np.linspace(0, TIME, int(TIME/(0.015*WL/LIGHT_SPEED)))

# Setup the fixed design structures
design = Design(width=10*µm, height=10*µm, material=Material(2.1), pml_size=WL/4)
design.add(Rectangle(position=(0,3*µm), width=1.2*µm, height=0.9*µm, material=Material(4)))
design.add(Rectangle(position=(3*µm, 0*µm), width=0.9*µm, height=1.2*µm, material=Material(4)))

# Define the optimization region of the design
design.add(Rectangle(position=(1.2*µm, 1.2*µm), width=6*µm, height=6*µm, material=VariableMaterial(2.1, 4), optimize=True))

# Define the sources and the monitor
signal = ramped_cosine(T, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=TIME/3, t_max=TIME)
design.add(ModeSource(design=design, start=(0.5*µm, 2.7*µm), end=(0.5*µm, 4.1*µm), wavelength=WL, signal=signal))
design.show()

# Setup the forward and backward simulations
forward_sim = FDTD(design=design, time=T, mesh="regular", resolution=WL/40)
backward_sim = forward_sim.reverse()

# Run the forward simulation
# Run the backward simulation
# Calculate the gradient
# Update the design



#sim.run(live=True, axis_scale=[-1,1], save_animation=True, clean_visualization=True, animation_filename="resring2.mp4")