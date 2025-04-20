from beamz import *
import numpy as np

WL = 1.55 * µm
TIME = 20 * WL / LIGHT_SPEED
T = np.linspace(0, TIME, int(TIME / (0.015 * WL / LIGHT_SPEED)))

# Setup the design
design = Design(width=10*µm, height=10*µm, material=Material(2.1), pml_size=WL/4)
design.add(Rectangle(position=(0, 3*µm), width=1.2*µm, height=0.9*µm, material=Material(4)))
design.add(Rectangle(position=(3*µm, 0), width=0.9*µm, height=1.2*µm, material=Material(4)))

# Add optimizable region
design.add(Rectangle(position=(1.2*µm, 1.2*µm), width=6*µm, height=6*µm, material=Material(4)))

#design.add(Rectangle(position=(1.2*µm, 1.2*µm), width=6*µm, height=6*µm, material=VariableMaterial(2.1, 4), optimize=True))
# , optimize=True

# Add source and monitor
signal = ramped_cosine(T, amplitude=1.0, frequency=LIGHT_SPEED/WL, ramp_duration=TIME/3, t_max=TIME)
design.add(ModeSource(design=design, start=(0.5*µm, 2.7*µm), end=(0.5*µm, 4.1*µm), wavelength=WL, signal=signal))


#monitor = PowerMonitor(position=(9.5*µm, 2.7*µm), size=(0, 1.4*µm), axis="x")  # Simplified monitor
#design.add(monitor)
design.show()

# Setup FDTD simulator
sim = FDTD(design=design, time=T, mesh="regular", resolution=WL/30)
sim.run(live=True)
sim.plot_power(db_colorbar=True)

# Define the objective function: Maximize power at the monitor
#def objective(sim_result):
#    return sim_result[monitor].total_power()

# Optimize
#opt = AdjointOptimizer(simulation=sim, objective_fn=objective, learning_rate=0.01)
#opt.optimize(iterations=50)