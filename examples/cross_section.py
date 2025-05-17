from beamz import *
import numpy as np

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
    return resolution, dt

ANGLE = 90
W = 30*µm; H = 30*µm
WL = 1.55*µm
TIME = 40*WL/LIGHT_SPEED

# Get material properties - proper calculation of relative permittivity from refractive index
WG_WIDTH = 3.2*µm 
n_clad = 1.42
n_core = 1.45
# Calculate relative permittivity (εr = n^2)
CLAD = n_clad**2
CORE = n_core**2

# Calculate optimal resolution and time step
resolution, dt = calc_optimal_fdtd_params(WL, max(n_core, n_clad), points_per_wavelength=30)

# Create time array
T = np.arange(0, TIME, dt)

# Create design with proper material definitions
design = Design(width=W, height=H, material=Material(CLAD), pml_size=3*WL)
design.add(Rectangle(position=(0,H/2-WG_WIDTH/2), width=W, height=WG_WIDTH, material=Material(CORE)))
design.add(Rectangle(position=(0,H/2-WG_WIDTH/2), width=W, height=WG_WIDTH, material=Material(CORE)).rotate(90))
signal = ramped_cosine(T, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=WL*10/LIGHT_SPEED, t_max=TIME/2)
design.add(ModeSource(design=design, start=(3.5*WL,H/2-1.5*WG_WIDTH), end=(3.5*WL, H/2+1.5*WG_WIDTH), wavelength=WL, signal=signal))
design.show()
sim = FDTD(design=design, time=T, mesh="regular", resolution=resolution)
sim.estimate_memory_usage()
sim.run(live=True, axis_scale=[-0.5,0.5], accumulate_power=True, save_memory_mode=True)
sim.plot_power(db_colorbar=True)