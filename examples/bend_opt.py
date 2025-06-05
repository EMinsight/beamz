from beamz import *
import numpy as np# Optionally: Visualize the design region evolution
import matplotlib.pyplot as plt
import optax  # Adam optimizer library

# Parameters
X = 13*µm # domain width
Y = 13*µm # domain height
WL = 1.55*µm # wavelength
TIME = 40*WL/LIGHT_SPEED # total simulation duration
N_CORE = 2.04 # Si3N4
N_CLAD = 1.444 # SiO2
WG_W = 0.565*µm # width of the waveguide
S = 5*µm
OFFSET = 1.05*µm # offset of the output waveguides from center of the MMI
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=2, safety_factor=0.60) 

# Initialize design region permittivity matrix for inverse design
design_region_size = 10  # Higher resolution for better design
design_reg_mat = np.ones((design_region_size, design_region_size)) * N_CLAD**2
# Add some random noise to the design region
design_reg_mat += np.random.normal(0, 0.1, design_reg_mat.shape)
# Create CustomMaterial for the design region
design_bounds = ((X/2-S/2, X/2+S/2), (Y/2-S/2, Y/2+S/2))  # Design region bounds
custom_material = CustomMaterial(
    permittivity_grid=design_reg_mat,
    bounds=design_bounds,
    interpolation='linear'
)

# Design the MMI with input and output waveguides
design = Design(width=X, height=Y, material=Material(N_CLAD**2), pml_size=WL)
design += Rectangle(position=(0, Y/2-WG_W/2), width=X/2, height=WG_W, material=Material(N_CORE**2))
design += Rectangle(position=(X/2-WG_W/2, Y/2-WG_W/2), width=WG_W, height=Y/2+WG_W/2, material=Material(N_CORE**2))
# Use CustomMaterial for the design region
design += Rectangle(position=(X/2-S/2, Y/2-S/2), width=S, height=S, material=custom_material)
design.show()

grid = RegularGrid(design=design, resolution=DX)
grid.show()


# Define the source
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=WL*3/LIGHT_SPEED, t_max=TIME/2)

def forward_sim(design):
    # Create a copy of the design to make it easier to switch the source with the detector
    design_copy_forward = design.copy()
    # Add a source
    design += ModeSource(design=design_copy_forward, start=(2*µm, Y/2-1.2*µm), end=(2*µm, Y/2+1.2*µm), wavelength=WL, signal=signal)
    # Add a monitor
    design += Monitor(design=design, start=(2*µm, Y/2-1.2*µm), end=(2*µm, Y/2+1.2*µm), wavelength=WL)
    design.show()
    # Run the simulation and show results
    sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
    field_history = sim.run(live=True, save_memory_mode=True, accumulate_power=True)
    sim.plot_power(db_colorbar=True)
    # return the entire forward sim field history
    return field_history

def backward_sim(design, field_history):
    # Switch the source with the monitor. To do so, we create a new copy of the design first
    # Create a copy of the design to make it easier to switch the source with the detector
    design_copy = design.copy()
    # Add the source at new position
    design += ModeSource(design=design, start=(X/2-1.2*µm, 11*µm), end=(X/2+1.2*µm, 11*µm), wavelength=WL, signal=signal, direction="-y")
    # Add the monitor at the old position
    design += Monitor(design=design, start=(2*µm, Y/2-1.2*µm), end=(2*µm, Y/2+1.2*µm), wavelength=WL)
    design.show()
    # Run the simulation and show results
    sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
    field_overlap = sim.run(live=True, save_memory_mode=True, accumulate_power=True)
    sim.plot_power(db_colorbar=True)
    # return the field overlap history
    return field_overlap

def ADAM_optimizer(field_overlap, design_reg_mat, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    ADAM optimizer using optax for updating design region permittivity.
    
    Args:
        field_overlap: Backward simulation field overlap (gradient information)
        design_reg_mat: Current design region permittivity matrix
        alpha: Learning rate (default: 0.001)
        beta1: First moment decay rate (default: 0.9)
        beta2: Second moment decay rate (default: 0.999)
        epsilon: Small constant for numerical stability (default: 1e-8)
    
    Returns:
        updated_design_reg_mat: Updated design region permittivity
        opt_state: Optimizer state for next iteration
    """
    # Initialize Adam optimizer
    optimizer = optax.adam(learning_rate=alpha, b1=beta1, b2=beta2, eps=epsilon)
    
    # Initialize optimizer state (this should be done once and passed between iterations)
    if not hasattr(ADAM_optimizer, 'opt_state'): ADAM_optimizer.opt_state = optimizer.init(design_reg_mat)
    # Use field_overlap as gradient (this represents sensitivity of objective to design changes)
    gradient = field_overlap
    
    # Apply gradient clipping to avoid instability
    gradient = np.clip(gradient, -1.0, 1.0)
    
    # Compute parameter updates using Adam
    updates, ADAM_optimizer.opt_state = optimizer.update(gradient, ADAM_optimizer.opt_state, design_reg_mat)
    
    # Apply updates to design region
    updated_design_reg_mat = optax.apply_updates(design_reg_mat, updates)
    
    # Apply bounds to keep permittivity in valid range
    updated_design_reg_mat = np.clip(updated_design_reg_mat, N_CLAD**2, N_CORE**2)
    
    return updated_design_reg_mat, ADAM_optimizer.opt_state

def run_optimization_loop(num_iterations=5):
    """Run the complete optimization loop."""
    global design_reg_mat, custom_material, design
    
    print(f"Starting optimization with {num_iterations} iterations...")
    print(f"Initial design region shape: {design_reg_mat.shape}")
    print(f"Initial permittivity range: {design_reg_mat.min():.3f} to {design_reg_mat.max():.3f}")
    
    objective_history = []
    
    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")
        
        # Run forward simulation
        print("Running forward simulation...")
        field_history = forward_sim(design)
        
        # Run backward simulation
        print("Running backward simulation...")
        field_overlap = backward_sim(design, field_history)
        
        # Extract gradient information (simplified)
        # In practice, you'd compute proper gradients from field overlaps
        gradient_field = np.random.normal(0, 0.1, design_reg_mat.shape)  # Placeholder
        
        # Update design using ADAM optimizer
        print("Updating design with ADAM optimizer...")
        design_reg_mat, opt_state = ADAM_optimizer(
            field_overlap=gradient_field,
            design_reg_mat=design_reg_mat,
            alpha=0.01
        )
        
        # Update the CustomMaterial with new permittivity grid
        custom_material.update_grid('permittivity', design_reg_mat)
        
        # Compute objective (power coupling - simplified)
        objective = np.sum(np.abs(design_reg_mat - N_CLAD**2))  # Placeholder metric
        objective_history.append(objective)
        
        print(f"Objective: {objective:.6f}")
        print(f"Updated permittivity range: {design_reg_mat.min():.3f} to {design_reg_mat.max():.3f}")
        
        # Visualize progress every few iterations
        if iteration % 2 == 0:
            plt.figure(figsize=(8, 6))
            plt.imshow(design_reg_mat, cmap='viridis', aspect='equal')
            plt.title(f'Design Region - Iteration {iteration + 1}')
            plt.colorbar(label='Permittivity')
            plt.savefig(f'design_iteration_{iteration + 1}.png', dpi=150, bbox_inches='tight')
            plt.show()
    
    # Plot optimization history
    plt.figure(figsize=(10, 6))
    plt.plot(objective_history, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Iteration')
    plt.ylabel('Objective')
    plt.title('Optimization Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig('optimization_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nOptimization completed!")
    print(f"Final objective: {objective_history[-1]:.6f}")
    
    return design_reg_mat, objective_history

# Run the optimization (commented out for now)
# optimized_design, history = run_optimization_loop(num_iterations=3)

#field_history = forward_sim(design) # this is a tensor of shape (time_steps, 2)
#field_overlap = backward_sim(design, field_history) # this is a tensor of shape (time_steps, 2)