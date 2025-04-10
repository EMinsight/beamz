import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

C_0 = 3e8  # Speed of light in vacuum

def compute_derivative_matrices(N, dL):
    """ Compute simple finite-difference matrices for a 1D waveguide. """
    Dxf = sp.diags([-1, 1], [0, 1], shape=(N, N), format='csr') / dL
    Dxb = sp.diags([-1, 1], [-1, 0], shape=(N, N), format='csr') / dL
    return Dxf, Dxb

def solve_modes(eps, omega, dL, m=2):
    """ Solve for the modes of a simple 1D waveguide. """
    k0 = omega / C_0
    N = eps.size
    # Compute derivative matrices
    Dxf, Dxb = compute_derivative_matrices(N, dL)
    # Define the eigenvalue problem matrix
    A = sp.diags(eps, 0) + (Dxf @ Dxb) * (1 / k0) ** 2
    # Solve for eigenmodes
    vals, vecs = spl.eigs(A, k=m, sigma=np.max(eps))
    return np.sqrt(vals), vecs

# Test on a simple ridge waveguide
lambda0 = 1.55e-6
dL = lambda0 / 100
omega = 2 * np.pi * C_0 / lambda0
Lx = 10 * lambda0
Nx = int(Lx / dL)

eps = np.ones(Nx)
wg_width = lambda0
wg_region = np.arange(Nx//2 - int(wg_width/dL/2), Nx//2 + int(wg_width/dL/2))
eps[wg_region] = 4  # Waveguide permittivity

vals, vecs = solve_modes(eps, omega, dL, m=3)

# Plot the waveguide structure
plt.figure(figsize=(10, 4))
plt.plot(np.linspace(-Lx, Lx, Nx) / lambda0, eps, label='Permittivity Profile')
plt.xlabel('x position (λ₀)')
plt.ylabel('Relative Permittivity')
plt.title('Waveguide Structure')
plt.grid(True)
plt.legend()
plt.show()

# Plot the first mode profile and compare to Gaussian
x_positions = np.linspace(-Lx, Lx, Nx) / lambda0
mode_profile = np.abs(vecs[:, 0])

# Fit a Gaussian to the mode profile
def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

# Initial guess for Gaussian parameters
initial_guess = (1, 0, wg_width / lambda0)
params, _ = curve_fit(gaussian, x_positions, mode_profile, p0=initial_guess)

# Plot mode profile and fitted Gaussian
plt.figure(figsize=(10, 4))
plt.plot(x_positions, mode_profile, label='Mode Profile')
plt.plot(x_positions, gaussian(x_positions, *params), label='Fitted Gaussian', linestyle='--')
plt.xlabel('x position (λ₀)')
plt.ylabel('Mode profile (normalized)')
plt.title(f'Effective index: {vals[0].real:.3f}')
plt.legend()
plt.grid(True)
plt.show()

# Solve for more modes
num_modes = 5  # Number of modes to solve for
vals, vecs = solve_modes(eps, omega, dL, m=num_modes)

# Plot each mode profile
for i in range(num_modes):
    plt.figure(figsize=(10, 4))
    plt.plot(x_positions, np.abs(vecs[:, i]), label=f'Mode {i+1} Profile')
    plt.plot(x_positions, gaussian(x_positions, *params), label='Fitted Gaussian', linestyle='--')
    plt.xlabel('x position (λ₀)')
    plt.ylabel('Mode profile (normalized)')
    plt.title(f'Mode {i+1} Effective index: {vals[i].real:.3f}')
    plt.legend()
    plt.grid(True)
    plt.show()