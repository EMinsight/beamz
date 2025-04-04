import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def plot_field_at_time(sim, time_point):
    """Plot the electric field at a specific time point.
    
    Args:
        sim: Simulation object containing the results
        time_point: Time at which to plot the field (in seconds)
    """
    # Find closest time step
    t_idx = np.argmin(np.abs(np.array(sim.results['t']) - time_point))
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot the field
    im = plt.imshow(sim.results['Ez'][t_idx], cmap='RdBu', interpolation='bicubic')
    plt.title(f'Electric Field (Ez) at t = {sim.results["t"][t_idx]:.2e} s')
    plt.colorbar(label='Ez')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def animate_field(sim, field="Ez", save_animation=False):
    """Create an animation of the field evolution.
    
    Args:
        sim (Simulation): Simulation object with loaded results
        field (str): Field to animate ("Ez", "Hx", or "Hy")
        save_animation (bool): Whether to save the animation to a file
    """
    print("Creating animation...")
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