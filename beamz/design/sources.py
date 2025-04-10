import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from beamz.const import LIGHT_SPEED, µm
import matplotlib.pyplot as plt

class PointSource():
    """Uniform current source with a zero size."""
    def __init__(self, position=(0,0), signal=0):
        self.position = position
        self.signal = signal


class ModeSource():
    """Calculates and injects the mode profiles for a cross section given a start and end point."""
    def __init__(self, design, start, end, wavelength=1.55*µm, signal=0):
        self.start = start
        self.end = end
        self.wavelength = wavelength
        self.design = design
        self.signal = signal

    def get_eps_1d(self):
        """Calculate the 1D permittivity profile by stepping along the line from start to end point."""
        x0, y0 = self.start
        x1, y1 = self.end
        num_points = int(np.hypot(x1 - x0, y1 - y0) / (self.wavelength / 200))  # Increase resolution
        print("num_points:", num_points)
        x, y = np.linspace(x0, x1, num_points), np.linspace(y0, y1, num_points)
        eps_1d = np.zeros(num_points)
        for i, (x_i, y_i) in enumerate(zip(x, y)):
            eps_1d[i] = self.design.get_material_value(x_i, y_i)
        return eps_1d
    
    def get_mode_profiles(self, eps_1d):
        """Calculate the mode profiles for a cross section given a 1D permittivity profile."""
        omega = 2 * np.pi * LIGHT_SPEED / self.wavelength
        dL = self.wavelength
        vals, vecs = solve_line_modes(eps_1d, self.start, self.end, omega, dL)
        return vals, vecs

    def show(self):
        """Show the mode profiles for a cross section given a 1D permittivity profile."""
        eps_1d = self.get_eps_1d()
        vals, vecs = self.get_mode_profiles(eps_1d)

        # Plot the 1D permittivity profile and all mode profiles on the same subplot
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plot permittivity profile
        ax1.plot(eps_1d, color='black', label='1D Permittivity Profile')
        ax1.set_xlabel('Position along the line')
        ax1.set_ylabel('Relative Permittivity', color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        # Create a second y-axis for the mode profiles
        ax2 = ax1.twinx()
        colors = ['crimson', 'blue', 'green', 'orange', 'purple']  # Add more colors if needed
        for i in range(vecs.shape[1]):
            ax2.plot(np.abs(vecs[:, i])**2, color=colors[i % len(colors)], label=f'Mode {i+1} Effective index: {vals[i].real:.3f}')
        ax2.set_ylabel('Mode Amplitude')
        ax2.tick_params(axis='y')
        ax2.set_xlim(0, len(eps_1d))

        # Add title, legend, and grid
        plt.title('Mode Profiles')
        fig.tight_layout()
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.grid(True)
        plt.show()


# PlaneWave: Uniform current distribution on an infinite extent plane.

def solve_modes(eps, omega, dL, m=2):
    """Solve for the modes of a simple 1D waveguide."""
    k0 = omega / LIGHT_SPEED
    N = eps.size
    # Compute derivative matrices
    Dxf, Dxb = compute_derivative_matrices(N, dL)
    # Define the eigenvalue problem matrix
    A = sp.diags(eps, 0) + (Dxf @ Dxb) * (1 / k0) ** 2
    # Solve for eigenmodes
    vals, vecs = spl.eigs(A, k=m, sigma=np.max(eps))
    # Normalize mode profiles
    vecs = vecs / np.sqrt(np.sum(np.abs(vecs)**2, axis=0))
    return np.sqrt(vals), vecs

def compute_derivative_matrices(N, dL):
    """ Compute simple finite-difference matrices for a 1D waveguide. """
    Dxf = sp.diags([-1, 1], [0, 1], shape=(N, N), format='csr') / dL
    Dxb = sp.diags([-1, 1], [-1, 0], shape=(N, N), format='csr') / dL
    return Dxf, Dxb

def solve_line_modes(eps_1d, start, end, omega, dL, m=2):
    """Solve for modes along a 1D line in a 1D permittivity profile."""
    # Extract line coordinates
    x0, y0 = start
    x1, y1 = end
    num_points = int(np.hypot(x1 - x0, y1 - y0) / dL)
    x, y = np.linspace(x0, x1, num_points), np.linspace(y0, y1, num_points)
    # Solve for modes using the 1D permittivity
    vals, vecs = solve_modes(eps_1d, omega, dL, m)
    return vals, vecs