from beamz import *
import numpy as np
import matplotlib.pyplot as plt
import optax  # Adam optimizer library
from scipy.ndimage import zoom, gaussian_filter

# Parameters
X, Y = 13*µm, 13*µm # domain height
WL = 1.55*µm # wavelength
TIME = 10*WL/LIGHT_SPEED  # Further reduced to 10 wavelengths for stability testing
N_CORE, N_CLAD = 2.04, 1.444 # Si3N4
WG_W = 0.565*µm # width of the waveguide
S = 5*µm
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=2, safety_factor=0.6, points_per_wavelength=13) 

# Initialize design region permittivity matrix for inverse design
design_region_size = 40  # Higher resolution for better design
design_reg_mat = np.ones((design_region_size, design_region_size)) * (N_CLAD**2 + N_CORE**2) / 2
design_reg_mat += np.random.normal(0, 0.1, design_reg_mat.shape)  # Reduced from 0.1 to 0.05
design_bounds = ((X/2-S/2, X/2+S/2), (Y/2-S/2, Y/2+S/2))  # Design region bounds
custom_material = CustomMaterial(permittivity_grid=design_reg_mat, bounds=design_bounds, interpolation='cubic')

# Design the MMI with input and output waveguides
design = Design(width=X, height=Y, material=Material(N_CLAD**2), pml_size=WL)
design += Rectangle(position=(0, Y/2-WG_W/2), width=X/2, height=WG_W, material=Material(N_CORE**2))
design += Rectangle(position=(X/2-WG_W/2, Y/2-WG_W/2), width=WG_W, height=Y/2+WG_W/2, material=Material(N_CORE**2))
design += Rectangle(position=(X/2-S/2, Y/2-S/2), width=S, height=S, material=custom_material)
design.show()

# Define the source
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=WL*3/LIGHT_SPEED, t_max=TIME/2)

def forward_sim(design):
    # Create a copy of the design to make it easier to switch the source with the detector
    design_copy_forward = design.copy()
    design_copy_forward += ModeSource(design_copy_forward, start=(2*µm, Y/2-1.2*µm), end=(2*µm, Y/2+1.2*µm), wavelength=WL, signal=signal)
    design_copy_forward += Monitor(design_copy_forward, start=(2*µm, Y/2-1.2*µm), end=(2*µm, Y/2+1.2*µm))
    #design_copy_forward.show()
    grid = RegularGrid(design=design_copy_forward, resolution=DX)
    grid.show()

    # Run the simulation and show results - IMPORTANT: save_memory_mode=False to keep field history
    sim = FDTD(design=design_copy_forward, time=time_steps, mesh="regular", resolution=DX)
    field_history = sim.run(live=False, save_memory_mode=False, accumulate_power=False, save=True)
    return field_history

def backward_sim(design, field_history):
    # Switch the source with the monitor. To do so, we create a new copy of the design first
    # Create a copy of the design to make it easier to switch the source with the detector
    design_copy_backward = design.copy()
    design_copy_backward += ModeSource(design_copy_backward, start=(X/2-1.2*µm, 11*µm), end=(X/2+1.2*µm, 11*µm), wavelength=WL, signal=signal, direction="-y")
    design_copy_backward += Monitor(design_copy_backward, start=(2*µm, Y/2-1.2*µm), end=(2*µm, Y/2+1.2*µm))
    #design_copy_backward.show()
    # Run the simulation and show results
    sim = FDTD(design=design_copy_backward, time=time_steps, mesh="regular", resolution=DX)
    
    # Initialize backward simulation
    sim.initialize_simulation(save=True, live=False, save_memory_mode=False)

    # Initialize overlap accumulation
    total_overlap = 0.0
    total_power = 0.0
    gradient_accumulator = np.zeros((design_region_size, design_region_size), dtype=complex)
    
    # Get design region bounds in grid coordinates
    grid_dx = sim.dx
    grid_dy = sim.dy
    x_start_idx = int((design_bounds[0][0] - 0) / grid_dx)
    x_end_idx = int((design_bounds[0][1] - 0) / grid_dx)
    y_start_idx = int((design_bounds[1][0] - 0) / grid_dy)
    y_end_idx = int((design_bounds[1][1] - 0) / grid_dy)
    
    # Run backward simulation step by step and calculate overlap with forward fields
    while sim.step():
        # Calculate overlap with forward field history at each step
        overlap = sim.calculate_field_overlap(field_history, field='Ez')
        total_overlap += overlap
        power = sim.get_monitor_power(0)  # Get power from first monitor
        total_power += np.abs(power)
        
        # Calculate spatial gradient for adjoint method
        if sim.current_step < len(field_history['Ez']):
            # Get current backward fields
            Ez_backward = sim.get_current_fields()['Ez']
            # Get corresponding forward fields
            Ez_forward = field_history['Ez'][sim.current_step]
            
            # Extract design region from both fields
            Ez_back_design = Ez_backward[y_start_idx:y_end_idx, x_start_idx:x_end_idx]
            Ez_forward_design = Ez_forward[y_start_idx:y_end_idx, x_start_idx:x_end_idx]
            
            # Resize to design region resolution if needed
            if Ez_back_design.shape != (design_region_size, design_region_size):
                scale_y = design_region_size / Ez_back_design.shape[0]
                scale_x = design_region_size / Ez_back_design.shape[1]
                Ez_back_design = zoom(Ez_back_design, (scale_y, scale_x), order=1)
                Ez_forward_design = zoom(Ez_forward_design, (scale_y, scale_x), order=1)
            
            # Compute gradient contribution (adjoint method)
            # Gradient of objective w.r.t. permittivity = 2π/λ * ε₀ * ω * Im(E_forward* × E_backward)
            gradient_contribution = np.real(Ez_forward_design * np.conj(Ez_back_design))
            gradient_accumulator += gradient_contribution
        
        if sim.current_step % 50 == 0:
            print(f"Step {sim.current_step}/{sim.num_steps}")
            print(f"Overlap: {overlap:.6f}")
    
    # Finalize simulation
    results = sim.finalize_simulation()
    
    print(f"Backward simulation complete. Total overlap: {total_overlap:.6f}")
    
    # Normalize gradient by number of time steps
    spatial_gradient = gradient_accumulator / len(time_steps)
    
    return total_overlap, total_power, spatial_gradient

def ADAM_optimizer(field_overlap, design_reg_mat, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    # Initialize Adam optimizer
    optimizer = optax.adam(learning_rate=alpha, b1=beta1, b2=beta2, eps=epsilon)
    # Initialize optimizer state (this should be done once and passed between iterations)
    if not hasattr(ADAM_optimizer, 'opt_state'): ADAM_optimizer.opt_state = optimizer.init(design_reg_mat)
    # Use field_overlap as gradient (this represents sensitivity of objective to design changes)
    gradient = field_overlap
    gradient = np.clip(gradient, -1.0, 1.0)     # Apply gradient clipping to avoid instability
    # Compute parameter updates using Adam
    updates, ADAM_optimizer.opt_state = optimizer.update(gradient, ADAM_optimizer.opt_state, design_reg_mat)
    # Apply updates to design region
    updated_design_reg_mat = optax.apply_updates(design_reg_mat, updates)
    # Apply bounds to keep permittivity in valid range - use same bounds as initialization
    min_perm = N_CLAD**2; max_perm = N_CORE**2
    updated_design_reg_mat = np.clip(updated_design_reg_mat, min_perm, max_perm)
    return updated_design_reg_mat, ADAM_optimizer.opt_state

def run_optimization_loop(num_iterations=5):
    """Run the complete optimization loop."""
    global design_reg_mat, custom_material, design
    

    objective_history = []
    
    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")
        
        # Update the design with current material parameters BEFORE running simulations
        if iteration > 0:  # Skip first iteration since design is already initialized
            print("Updating design region material...")
            # Update the CustomMaterial with new permittivity grid
            custom_material.update_grid('permittivity', design_reg_mat)
            
            # Rebuild the design with updated material
            # Create a fresh design with the same base structures but updated custom material
            design = Design(width=X, height=Y, material=Material(N_CLAD**2), pml_size=WL)
            design += Rectangle(position=(0, Y/2-WG_W/2), width=X/2, height=WG_W, material=Material(N_CORE**2))
            design += Rectangle(position=(X/2-WG_W/2, Y/2-WG_W/2), width=WG_W, height=Y/2+WG_W/2, material=Material(N_CORE**2))
            design += Rectangle(position=(X/2-S/2, Y/2-S/2), width=S, height=S, material=custom_material)
            
            print(f"Updated design region with permittivity range: {design_reg_mat.min():.3f} to {design_reg_mat.max():.3f}")
        
        # Run forward simulation
        print("Running forward simulation...")
        field_history = forward_sim(design)
        
        # Run backward simulation
        print("Running backward simulation...")
        total_overlap, total_power, spatial_gradient = backward_sim(design, field_history)
        
        # The spatial_gradient now contains proper adjoint gradients for each pixel
        # Apply some smoothing and normalization
        gradient_field = spatial_gradient.real  # Use real part for gradients
        
        # Normalize gradient to prevent exploding/vanishing gradients
        gradient_norm = np.linalg.norm(gradient_field)
        if gradient_norm > 0:
            gradient_field = gradient_field / gradient_norm * 0.1  # Scale to reasonable magnitude
        
        # Add small amount of regularization to encourage smooth designs
        gradient_field = gaussian_filter(gradient_field, sigma=0.5)

        # Update design using ADAM optimizer
        print("Updating design with ADAM optimizer...")
        design_reg_mat, opt_state = ADAM_optimizer(
            field_overlap=gradient_field,
            design_reg_mat=design_reg_mat,
            alpha=0.1  # Reduced learning rate for stability with proper gradients
        )

        # Compute objective using transmitted power as the main metric
        # We want to maximize power transmission from input to output
        average_power = total_power / len(time_steps)
        
        # Use negative power as objective (minimize negative = maximize positive)
        objective = -average_power
        objective_history.append(objective)
        
        print(f"Field overlap magnitude: {np.abs(total_overlap):.6f}")
        print(f"Total transmitted power: {total_power:.6f}")
        print(f"Average power per step: {average_power:.6f}")
        print(f"Objective (negative avg power): {objective:.6f}")
        print(f"Gradient field range: {gradient_field.min():.6f} to {gradient_field.max():.6f}")
        print(f"Gradient field norm: {np.linalg.norm(gradient_field):.6f}")
        print(f"Updated permittivity range: {design_reg_mat.min():.3f} to {design_reg_mat.max():.3f}")
        
        # Visualize progress every few iterations  
        if iteration % 2 == 0:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot design region
            im1 = axes[0].imshow(design_reg_mat, cmap='gray', aspect='equal')
            axes[0].set_title(f'Design Region - Iteration {iteration + 1}')
            plt.colorbar(im1, ax=axes[0], label='Permittivity')
            
            # Plot gradient field
            im2 = axes[1].imshow(gradient_field, cmap='RdBu_r', aspect='equal')
            axes[1].set_title(f'Gradient Field - Iteration {iteration + 1}')
            plt.colorbar(im2, ax=axes[1], label='Gradient')
            
            plt.tight_layout()
            plt.savefig(f'design_and_gradient_iteration_{iteration + 1}.png', dpi=150, bbox_inches='tight')
            plt.show()
    
    # Plot optimization history
    plt.figure(figsize=(10, 6))
    # Convert objective back to transmitted power for plotting
    transmission_history = [-obj for obj in objective_history]
    plt.plot(transmission_history, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Iteration')
    plt.ylabel('Transmitted Power')
    plt.title('Transmission Optimization Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig('transmission_optimization_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nOptimization completed!")
    print(f"Final objective: {objective_history[-1]:.6f}")
    print(f"Final transmitted power: {-objective_history[-1]:.6f}")
    
    return design_reg_mat, objective_history



# Test with fewer iterations first
optimized_design, history = run_optimization_loop(num_iterations=5) # Reduced for testing with proper gradients