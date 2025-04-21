import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from beamz.const import LIGHT_SPEED, µm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class PointSource():
    """Uniform current source with a zero size."""
    def __init__(self, position=(0,0), signal=0):
        self.position = position
        self.signal = signal

class LineSource():
    """A current distribution along a line."""
    def __init__(self, start, end, direction=None, distribution=None, signal=0):
        self.start = start
        self.end = end
        self.signal = signal
        self.distribution = distribution
        self.direction = direction

class ModeSource():
    """Calculates and injects the mode profiles for a cross section given a start and end point."""
    def __init__(self, design, start, end, wavelength=1.55*µm, signal=0, direction="+x", npml=10, num_modes=2, 
                 grid_resolution=500):
        """
        Args:
            design: Design object containing the structures
            start: Starting point of the source line (x,y)
            end: End point of the source line (x,y)
            wavelength: Source wavelength
            signal: Time-dependent signal
            direction: Direction of propagation ("+x", "-x", "+y", "-y")
            npml: Number of PML layers to use at boundaries
            num_modes: Number of modes to calculate
            grid_resolution: Points per wavelength for grid resolution (higher = finer)
        """
        self.start = start
        self.end = end
        self.wavelength = wavelength
        self.design = design
        self.signal = signal
        self.direction = direction
        self.npml = npml
        self.num_modes = num_modes
        self.grid_resolution = grid_resolution
        # Calculate and store mode profiles
        self.dL = self.wavelength / grid_resolution  # Sampling resolution
        eps_1d = self.get_eps_1d()
        self.omega = 2 * np.pi * LIGHT_SPEED / self.wavelength
        self.effective_indices, self.mode_vectors = solve_modes(eps_1d, self.omega, self.dL, npml=self.npml, m=self.num_modes)
        self.mode_profiles = []
        for mode_number in range(self.mode_vectors.shape[1]):
            self.mode_profiles.append(self.get_xy_mode_line(self.mode_vectors, mode_number))

    def get_eps_1d(self):
        """Calculate the 1D permittivity profile by stepping along the line from start to end point."""
        x0, y0 = self.start
        x1, y1 = self.end
        num_points = int(np.hypot(x1 - x0, y1 - y0) / self.dL)  # Use the class dL value
        x, y = np.linspace(x0, x1, num_points), np.linspace(y0, y1, num_points)
        eps_1d = np.zeros(num_points)
        for i, (x_i, y_i) in enumerate(zip(x, y)):
            eps_1d[i], _, _ = self.design.get_material_value(x_i, y_i)
        return eps_1d
    
    def get_xy_mode_line(self, vecs, mode_number):
        """Get the mode profile for a specific mode along the line."""
        x0, y0 = self.start
        x1, y1 = self.end
        num_points = vecs.shape[0]  # Number of points along the line
        x = np.linspace(x0, x1, num_points)
        y = np.linspace(y0, y1, num_points)
        
        # Create mode profile for the specified mode
        mode_profile = []
        for j in range(num_points):  # For each point
            amplitude = np.abs(vecs[j, mode_number])  # Absolute value of the field
            mode_profile.append([amplitude, x[j], y[j]])
            
        return mode_profile

    def show(self):
        """Show the mode profiles for a cross section given a 1D permittivity profile."""
        eps_1d = self.get_eps_1d()
        N = eps_1d.size
        # Recalculate physical coordinates for plotting (assuming linear path)
        # Use total length and N to get coordinates corresponding to eps_1d indices
        line_length = np.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])
        # Create coordinate array from 0 to line_length
        coords = np.linspace(0, line_length, N) / µm # Plot in microns
        plot_unit = 'µm'

        vals, vecs = solve_modes(eps_1d, self.omega, self.dL, npml=self.npml, m=self.num_modes)

        fig, ax1 = plt.subplots(figsize=(10, 5))
        # Plot permittivity profile vs physical coordinates
        ax1.plot(coords, eps_1d, color='black', label='1D Permittivity Profile')
        ax1.set_xlabel(f'Position along the line ({plot_unit})')
        ax1.set_ylabel('Relative Permittivity', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_xlim(coords[0], coords[-1]) # Set limits based on coordinate range
        # Create a second y-axis for the mode profiles
        ax2 = ax1.twinx()
        colors = ['crimson', 'blue', 'green', 'orange', 'purple']
        for i in range(vecs.shape[1]):
            ax2.plot(coords, np.abs(vecs[:, i])**2, color=colors[i % len(colors)], 
                     label=f'Mode {i+1} Effective index: {vals[i].real:.3f}')
        ax2.set_ylabel('Mode Intensity (|E|²)') # Changed label for clarity
        ax2.tick_params(axis='y')
        # Ensure y-axis starts at 0 for intensity
        ax2.set_ylim(bottom=0)

        # Add shaded regions for PML
        if self.npml > 0 and N > self.npml:
            pml_width_left = coords[self.npml-1] - coords[0]
            pml_width_right = coords[-1] - coords[N-self.npml]
            # Left PML region
            ax1.add_patch(patches.Rectangle((coords[0], ax1.get_ylim()[0]), pml_width_left, 
                                            ax1.get_ylim()[1]-ax1.get_ylim()[0], 
                                            facecolor='gray', alpha=0.2, label='PML Region'))
            # Right PML region
            ax1.add_patch(patches.Rectangle((coords[N-self.npml], ax1.get_ylim()[0]), pml_width_right, 
                                            ax1.get_ylim()[1]-ax1.get_ylim()[0], 
                                            facecolor='gray', alpha=0.2))
            # Adjust xlim slightly to make patches fully visible if needed
            ax1.set_xlim(coords[0] - 0.01*line_length/µm, coords[-1] + 0.01*line_length/µm)

        plt.title('Mode Profiles')
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # Avoid duplicate PML label if patch was added
        unique_labels = {} 
        for line, label in zip(lines1 + lines2, labels1 + labels2):
            if label not in unique_labels:
                unique_labels[label] = line
        ax2.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
        
        plt.grid(True)
        fig.tight_layout()
        plt.show()


# Improved mode solver functions
def compute_derivative_matrices(N, dL, npml=0):
    """Compute finite-difference matrices for a 1D waveguide with PML boundaries.
    
    Args:
        N: Number of grid points
        dL: Grid spacing
        npml: Number of PML grid points on each side
    
    Returns:
        tuple: (Dxf, Dxb) forward and backward derivative matrices
    """
    # Basic derivative matrices
    Dxf = sp.diags([-1, 1], [0, 1], shape=(N, N), format='csr') / dL
    Dxb = sp.diags([-1, 1], [-1, 0], shape=(N, N), format='csr') / dL
    # Apply PML if requested
    if npml > 1: # Need at least 2 points for PML to have a non-zero thickness
        # Create PML scaling array
        sc_array = np.ones(N, dtype=complex)
        # Define PML parameters
        amax = 3.0  # PML strength 
        m = 3      # PML power (cubic profile often works well)
        # Calculate scaling factor based on normalized distance into PML (0=interface, 1=outer edge)
        def pml_scale(dist_norm):
            return 1.0 / (1.0 + 1j * amax * dist_norm**m)
        # Left PML: Interface at index npml-1, Outer edge at index 0
        for i in range(npml):
            # Normalized distance from interface: increases from 0 at i=npml-1 to 1 at i=0
            dist_norm = ((npml - 1) - i) / (npml - 1) 
            sc_array[i] = pml_scale(dist_norm)

        # Right PML: Interface at index N-npml, Outer edge at index N-1
        for k in range(npml):
            # Index in array is j = N - npml + k
            # Normalized distance from interface: increases from 0 at k=0 to 1 at k=npml-1
            dist_norm = k / (npml - 1)
            sc_array[N - npml + k] = pml_scale(dist_norm)

        # Create scaling matrices and apply to derivative operators
        sc_x = sp.diags(sc_array, 0)
        inv_sc_x = sp.diags(1/sc_array, 0)
        Dxf = inv_sc_x @ Dxf @ sc_x
        Dxb = inv_sc_x @ Dxb @ sc_x

    return Dxf, Dxb

def solve_modes(eps, omega, dL, npml=0, m=2):
    """Solve for the modes of a simple 1D waveguide.
    
    Args:
        eps: Permittivity profile
        omega: Angular frequency
        dL: Grid spacing
        npml: Number of PML layers
        m: Number of modes to compute
        
    Returns:
        tuple: (vals, vecs) effective indices and mode profiles
    """
    k0 = omega / LIGHT_SPEED
    N = eps.size
    # Compute derivative matrices with PML
    Dxf, Dxb = compute_derivative_matrices(N, dL, npml)
    # Define the eigenvalue problem matrix
    diag_eps = sp.diags(eps.flatten(), 0)
    # Use averaged operator for potentially better symmetry
    A = diag_eps + 0.5 * (Dxf @ Dxb + Dxb @ Dxf) * (1 / k0) ** 2
    # Solve for eigenmodes
    n_max = np.sqrt(np.max(eps))
    vals, vecs = spl.eigs(A, k=m, sigma=n_max**2, which='LM')
    # Normalize mode profiles
    vecs = vecs / np.sqrt(np.sum(np.square(np.abs(vecs)), axis=0))
    # Compute effective indices from eigenvalues
    neff = np.sqrt(vals)
    return neff, vecs