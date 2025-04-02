from beamz.sim import Simulation, StandardGrid
from beamz.materials import Material
from beamz.sources import PointSource, Wave
from beamz.const import LIGHT_SPEED
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_field(sim, field="Ez", save_animation=False):
    """Create an animation of the field evolution.
    
    Args:
        sim (Simulation): Simulation object with loaded results
        field (str): Field to animate ("Ez", "Hx", or "Hy")
        save_animation (bool): Whether to save the animation to a file
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(sim.results[field][0], cmap='RdBu')
    plt.colorbar(im, label=field)
    ax.set_title(f"{field} field evolution")
    ax.axis('off')
    
    def update(frame):
        im.set_array(sim.results[field][frame])
        ax.set_title(f"{field} field at t = {sim.results['t'][frame]:.2e} s")
        return [im]
    
    # Create animation
    anim = FuncAnimation(
        fig, update, frames=len(sim.results['t']),
        interval=50,  # 50ms between frames
        blit=True
    )
    
    if save_animation:
        # Save animation to file
        anim.save(
            f"simulation_results/{sim.name}_{field}_animation.gif",
            writer='pillow',
            fps=20
        )
        print(f"Animation saved to simulation_results/{sim.name}_{field}_animation.gif")
    
    plt.show()

# Parameters
wavelength = 1.55 # all units in µm
frequency = LIGHT_SPEED / wavelength # (m/s)/m = Hz
ramp = 10 / frequency # s
sim_time = 30 / frequency # Total simulation time

# Materials
Air = Material(
    name="air",
    permittivity=1.00058986,  # Relative permittivity at STP (0°C, 1 atm)
    permeability=1.00000037,  # Relative permeability at STP (0°C, 1 atm)
    conductivity=0.0,         # Perfect insulator
    color="white"
)

# Sources
dipole = PointSource(
    position=(150, 150),  # Center of the grid
    signal=Wave(
        direction=(0, 1),
        amplitude=10.0,
        frequency=LIGHT_SPEED/wavelength,
        ramp_up_time=10*wavelength/LIGHT_SPEED,  # Convert to seconds
        ramp_down_time=10*wavelength/LIGHT_SPEED
    )
)

# Combine all the information into a single simulation object
sim = Simulation(
    name="dipole_sim",
    type="2D",
    size=(300, 300),
    grid=StandardGrid(cell_size=wavelength/20),  # 20 cells per wavelength
    structures=None,
    sources=[dipole],
    monitors=None,
    device="cpu"
)

# Set simulation time
sim.time = sim_time
sim.num_steps = int(sim_time / sim.dt)

# Set up PML region
sim.setup_pml(thickness=50, sigma_max=1.0, m=3.5)

# Run the simulation
results = sim.run(save=True, animate_live=True)

# Get the most recent simulation file
simulation_files = glob.glob(os.path.join("simulation_results", "dipole_sim_*.h5"))
if not simulation_files:
    print("No simulation files found!")
    exit(1)

latest_file = max(simulation_files, key=os.path.getctime)
print(f"\nLoading results from: {latest_file}")

# Load and visualize results
loaded_sim = Simulation.load_results(latest_file)

# Create a figure with subplots for different times
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot at different times to show wave propagation
times = [sim_time/100, sim_time/10, sim_time]
for i, t in enumerate(times):
    # Find closest time step
    t_idx = np.argmin(np.abs(np.array(loaded_sim.results['t']) - t))
    
    # Plot the field
    im = axes[i].imshow(loaded_sim.results['Ez'][t_idx], cmap='RdBu', interpolation='bicubic')
    axes[i].set_title(f"t = {loaded_sim.results['t'][t_idx]:.2e} s")
    axes[i].axis('off')

# Add colorbar
plt.colorbar(im, ax=axes.ravel().tolist(), label='Ez')
plt.suptitle('Electric Field (Ez) at Different Times')
plt.tight_layout()
plt.show()

# Create and show animation
print("\nCreating animation...")
animate_field(loaded_sim, field="Ez", save_animation=True)