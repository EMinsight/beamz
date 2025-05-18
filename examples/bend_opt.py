from beamz import *
import numpy as np
from beamz.design.materials import VarMaterial
from beamz.helpers import calc_optimal_fdtd_params

# Define basic parameters of the simulation
X = 10*µm
Y = 10*µm
WL = 1.55 * µm
TIME = 10 * WL / LIGHT_SPEED
N_CORE = 2.04 # Si3N4
N_CLAD = 1.444 # SiO2
WG_WIDTH = 0.565*µm
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=2)

# Setup the design with sources, monitors, and a design region
design = Design(width=X, height=Y, material=Material(N_CLAD**2), pml_size=WL)
# Define waveguide input and output
design += Rectangle(position=(0, 4*µm-WG_WIDTH/2), width=4*µm, height=WG_WIDTH, material=Material(N_CORE**2))
design += Rectangle(position=(4*µm-WG_WIDTH/2, 0), width=WG_WIDTH, height=4*µm, material=Material(N_CORE**2))

# Define design region with variable material for optimization
#design_region = Rectangle(position=(1.2*µm, 1.2*µm), width=6*µm, height=6*µm)
#n_min, n_max = 1.0, 3.0  # Min and max refractive indices
#var_material = VarMaterial(permittivity=[n_min**2, (n_min**2 + n_max**2) / 2, n_max**2])
#design_region.material = var_material
#design.add(design_region)

# Define source and its signal
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, ramp_duration=TIME/3, t_max=TIME)
source = ModeSource(design=design, start=(WL*1.2, 4*µm-WG_WIDTH/2-WG_WIDTH), end=(WL*1.2, 4*µm+WG_WIDTH/2+WG_WIDTH),
                    wavelength=WL, signal=signal)
design += source

# Define the monitor and its function
monitor = Monitor(start=(4*µm-WG_WIDTH/2 -WG_WIDTH, WL*1.2), end=(4*µm+WG_WIDTH/2+WG_WIDTH, WL*1.2), accumulate_power=True, record_fields=True)
design += monitor

# Show the design
design.show()

# Setup the FDTD simulation
sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
sim.run(live=False, save_memory_mode=True, accumulate_power=True)

# Show powertotal over domain
#sim.plot_power(db_colorbar=True)

import matplotlib.pyplot as plt
print("Power:", monitor.power_accumulated)
print("Ez:", monitor.fields)

# Define objective function for optimization
def objective(sim_result):
    # Negative sign since we want to maximize power at monitor
    return -sim_result[monitor].total_power()

# Define filter to enforce minimum feature size
def apply_filter(params, radius=3):
    # Simple Gaussian filter
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(params, sigma=radius)

# Project parameters to enforce binary design
def apply_projection(params, beta=8, eta=0.5):
    # Projection function to encourage 0 or 1 values
    return (np.tanh(beta * eta) + np.tanh(beta * (params - eta))) / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))

# Define adjoint optimization
def adjoint_optimization(num_iterations=50, step_size=0.01, filter_radius=2, beta_init=2, beta_max=16):
    # Initialize tracking of objective function
    objectives = []
    current_mask = design_mask.copy()
    
    # Optimization loop
    for iteration in range(num_iterations):
        # Apply filtering and projection to current mask
        beta = beta_init + (beta_max - beta_init) * iteration / (num_iterations - 1)
        filtered_mask = apply_filter(current_mask, radius=filter_radius)
        projected_mask = apply_projection(filtered_mask, beta=beta)
        # Update design region material distribution
        sim.design_region.update_material(projected_mask)
        # Forward simulation
        sim.reset()
        forward_result = sim.run(save_fields=True, live=False)
        current_objective = objective(forward_result)
        objectives.append(current_objective)
        # Print status
        print(f"Iteration {iteration+1}/{num_iterations}: Objective = {current_objective:.4e}")
        # Last iteration - just show results
        if iteration == num_iterations - 1: break
        # Setup and run adjoint simulation
        # Adjoint source is the negative derivative of the objective with respect to fields
        adjoint_source = MonitorSource(monitor=monitor, design=design, amplitude=-1.0)
        sim.add(adjoint_source)
        # Run adjoint simulation (backward in time)
        sim.reset()
        sim.time_reversed = True
        adjoint_result = sim.run(save_fields=True, live=False)
        sim.time_reversed = False
        # Calculate gradients using forward and adjoint fields
        gradient = sim.calculate_gradient(forward_result, adjoint_result)
        # Apply chain rule for filtering and projection
        gradient_filtered = apply_filter(gradient, radius=filter_radius)
        gradient_projected = gradient_filtered * beta * (1 - np.tanh(beta * (filtered_mask - 0.5))**2)
        # Update design parameters with gradient descent
        current_mask = current_mask - step_size * gradient_projected
        # Clip to ensure parameters stay within [0, 1]
        current_mask = np.clip(current_mask, 0, 1)
        # Remove adjoint source for next iteration
        sim.remove(adjoint_source)
    
    return objectives, current_mask


# Initialize mask for design parameterization
#design_mask = np.zeros(sim.design_region.shape).fill(0.5)


# Run initial simulation to see performance before optimization
#sim.run(live=True)
#initial_result = sim.run()
#initial_objective = objective(initial_result)
#sim.plot_power(db_colorbar=True)

# Run optimization
#objectives, optimized_mask = adjoint_optimization(num_iterations=20, step_size=0.01)

# Apply final mask to the design
#filtered_mask = apply_filter(optimized_mask, radius=2)
#projected_mask = apply_projection(filtered_mask, beta=16)
#sim.design_region.update_material(projected_mask)

# Check the optimized design
#final_result = sim.run(live=True)
#final_objective = objective(final_result)
#print(f"Final objective value: {final_objective:.4e}")
#print(f"Improvement: {(final_objective - initial_objective) / initial_objective * 100:.2f}%")
#sim.plot_power(db_colorbar=True)

# Plot optimization progress
#import matplotlib.pyplot as plt
#plt.figure(figsize=(10, 5))
#plt.plot(objectives)
#plt.xlabel('Iteration')
#plt.ylabel('Objective Value')
#plt.title('Optimization Progress')
#plt.grid(True)
#plt.show()