from beamz import *
import numpy as np
from beamz.optimization.adjoint import AdjointOptimizer
from beamz.design.materials import VarMaterial
WL = 1.55 * µm
TIME = 20 * WL / LIGHT_SPEED
T = np.linspace(0, TIME, int(TIME / (0.015 * WL / LIGHT_SPEED)))

# Setup the design with sources, monitors, and a design region
design = Design(width=10*µm, height=10*µm, material=Material(2.1), pml_size=WL/4)
design.add(Rectangle(position=(0, 3*µm), width=1.2*µm, height=0.9*µm, material=Material(4)))
design.add(Rectangle(position=(3*µm, 0), width=0.9*µm, height=1.2*µm, material=Material(4)))
design.add(Rectangle(position=(1.2*µm, 1.2*µm), width=6*µm, height=6*µm, 
                     material=VarMaterial(permittivity=[2.1, 4], permeability=[1, 1], conductivity=[0, 0]), optimize=True))
signal = ramped_cosine(T, amplitude=1.0, frequency=LIGHT_SPEED/WL, ramp_duration=TIME/3, t_max=TIME)
source = ModeSource(design=design, start=(0.5*µm, 2.5*µm), end=(0.5*µm, 4.5*µm), wavelength=WL, signal=signal)
monitor = Monitor(start=(2.5*µm, 0.6*µm), end=(4.5*µm, 0.6*µm))
design.add(source)
design.add(monitor)
design.show()

# Setup the optimization using the adjoint method (based on the FDTD)
sim = FDTD(design=design, time=T, mesh="regular", resolution=WL/30)
sim.run(live=True)
sim.plot_power(db_colorbar=True)

def objective(sim_result): return -sim_result[monitor].total_power()
opt = AdjointOptimizer(simulation=sim, objective_fn=objective, filter_radius=0.3*µm)
objective_history = opt.optimize(iterations=10)

# Check the optimized design
sim.run(live=True)
sim.plot_power(db_colorbar=True)