from beamz import *
import numpy as np
from beamz.design.structures import RectPML

ANGLE = 5
W = 100*µm; H = 23*µm
WL = 1.55*µm
TIME = 115*WL/LIGHT_SPEED
WG_WIDTH = 3.2*µm; CLAD = 1.4485**2 ; CORE = 1.4795**2

# Calculate maximum refractive index
n_core = np.sqrt(CORE)
n_clad = np.sqrt(CLAD)
n_max = max(n_core, n_clad)


def calc_optimal_fdtd_params(wavelength, n_max, safety_factor=0.5, points_per_wavelength=20):
    """
    Calculate optimal FDTD grid resolution and time step based on wavelength and material properties.
    
    Args:
        wavelength: Light wavelength in vacuum
        n_max: Maximum refractive index in the simulation
        safety_factor: Multiplier for Courant condition (0.5 recommended for stability)
        points_per_wavelength: Number of grid points per wavelength in the highest index material
        
    Returns:
        tuple: (resolution, dt) - optimal spatial resolution and time step
    """
    # Calculate wavelength in the highest index material
    lambda_material = wavelength / n_max
    # Calculate optimal grid resolution based on desired points per wavelength
    resolution = lambda_material / points_per_wavelength
    # Calculate speed of light in the material
    c_material = LIGHT_SPEED / n_max
    # Calculate time step using Courant condition for 2D simulation
    dt_max = resolution / (c_material * np.sqrt(2))
    # Apply safety factor
    dt = safety_factor * dt_max
    print(f"Calculated parameters:")
    print(f"  - Material wavelength: {lambda_material*1e6:.3f} µm")
    print(f"  - Grid resolution: {resolution*1e6:.3f} µm ({wavelength/resolution:.1f} points per vacuum wavelength)")
    print(f"  - Maximum time step: {dt_max*1e15:.3f} fs")
    print(f"  - Recommended time step: {dt*1e15:.3f} fs (safety factor: {safety_factor})")
    
    return resolution, dt


# Calculate optimal resolution and time step
resolution, dt = calc_optimal_fdtd_params(WL, n_max, safety_factor=0.2, points_per_wavelength=10)

# Create time array
T = np.arange(0, TIME, dt)

design = Design(width=W, height=H, material=Material(CLAD), pml_size=2*WL)
design.add(RectPML(position=(W-4*WL,0), width=4*WL, height=H, orientation="right"))
design.add(Rectangle(position=(0,WL*4), width=W, height=WG_WIDTH, material=Material(CORE)))
design.add(Rectangle(position=(WL*8,WL*4), width=W, height=WG_WIDTH, material=Material(CORE)).rotate(ANGLE, point=(WL*8, WL*4)))
signal = ramped_cosine(T, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=WL*6/LIGHT_SPEED, t_max=TIME/3)
design.add(ModeSource(design=design, start=(2.7*WL, 2.1*WL), end=(2.7*WL, WG_WIDTH+6*WL), wavelength=WL, signal=signal))


# Add exit monitor at the upper waveguide
design.add(Monitor(
    design=design,
    start=(W-6*WL, H-6*WL),
    end=(W-6*WL, H-2*WL),
    name="Upper_Exit"
))

# Add exit monitor at the lower waveguide
design.add(Monitor(
    design=design,
    start=(W-6*WL, 2.1*WL),
    end=(W-6*WL, WG_WIDTH+6*WL),
    name="Lower_Exit"
))

design.show()
sim = FDTD(design=design, time=T, mesh="regular", resolution=resolution)
sim.run(live=True, axis_scale=[-1,1], accumulate_power=True, save_memory_mode=True)
sim.plot_power(log_scale=False, db_colorbar=True)

# Plot monitor field data
sim.plot_monitors(field='Ez')

# Plot monitor power data
sim.plot_monitors(power=True, db_scale=True)

# Show coupling ratio calculation
print("\nCoupling Analysis:")
for monitor in design.monitors:
    if monitor.power_accumulated is not None:
        power_sum = np.sum(monitor.power_accumulated)
        print(f"Monitor '{monitor.name}': Total Power = {power_sum:.6e}")

# Calculate coupling ratio if we have both monitors
if len(design.monitors) >= 2:
    monitor1 = design.monitors[0]
    monitor2 = design.monitors[1]
    
    if monitor1.power_accumulated is not None and monitor2.power_accumulated is not None:
        power1 = np.sum(monitor1.power_accumulated)
        power2 = np.sum(monitor2.power_accumulated)
        total_power = power1 + power2
        
        if total_power > 0:
            coupling_ratio = power2 / total_power
            print(f"\nCoupling Ratio: {coupling_ratio:.2%}")
            print(f"Upper WG: {power1/total_power:.2%}, Lower WG: {power2/total_power:.2%}")