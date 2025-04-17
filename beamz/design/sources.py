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
    def __init__(self, design, start, end, wavelength=1.55*µm, signal=0, direction="+x", npml=10, num_modes=2):
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
        """
        self.start = start
        self.end = end
        self.wavelength = wavelength
        self.design = design
        self.signal = signal
        self.direction = direction
        self.npml = npml
        self.num_modes = num_modes
        
        # Calculate and store mode profiles
        eps_1d = self.get_eps_1d()
        self.omega = 2 * np.pi * LIGHT_SPEED / self.wavelength
        self.dL = self.wavelength / 200  # Sampling resolution
        
        # Calculate mode profiles
        self.effective_indices, self.mode_vectors = self.get_mode_profiles(eps_1d)
        self.mode_profiles = []
        
        # Store profiles for each mode
        for mode_number in range(self.mode_vectors.shape[1]):
            self.mode_profiles.append(self.get_xy_mode_line(self.mode_vectors, mode_number))

    def get_eps_1d(self):
        """Calculate the 1D permittivity profile by stepping along the line from start to end point."""
        x0, y0 = self.start
        x1, y1 = self.end
        num_points = int(np.hypot(x1 - x0, y1 - y0) / (self.wavelength / 200))  # Increase resolution
        x, y = np.linspace(x0, x1, num_points), np.linspace(y0, y1, num_points)
        eps_1d = np.zeros(num_points)
        for i, (x_i, y_i) in enumerate(zip(x, y)):
            eps_1d[i], _, _ = self.design.get_material_value(x_i, y_i)
        return eps_1d
    
    def get_mode_profiles(self, eps_1d):
        """Calculate the mode profiles for a cross section given a 1D permittivity profile.
        
        Args:
            eps_1d: 1D array of relative permittivity values along the line
            
        Returns:
            tuple: (vals, vecs)
                vals: Complex array of effective indices (n_eff) for each mode
                    - Real part represents the phase velocity (n_eff)
                    - Imaginary part represents loss/gain
                    - Shape: (num_modes,)
                vecs: Complex array of mode field profiles
                    - Each column represents a different mode
                    - Each row represents a point along the line
                    - Values are complex field amplitudes
                    - Shape: (num_points, num_modes)
        """
        vals, vecs = solve_line_modes(eps_1d, self.start, self.end, self.omega, 
                                      self.dL, npml=self.npml, m=self.num_modes, filtering=True)
        return vals, vecs
    
    def get_xy_mode_line(self, vecs, mode_number):
        """Get the mode profile for a specific mode along the line.
        
        Args:
            vecs: Complex array of mode field profiles
                - Each column represents a different mode
                - Each row represents a point along the line
                - Values are complex field amplitudes
            mode_number: Index of the mode to extract (0-based)
                
        Returns:
            list: List of [amplitude, x, y] points for the specified mode
                - amplitude: Absolute value of the mode field at that point
                - x: x-coordinate of the point
                - y: y-coordinate of the point
        """
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
            ax2.plot(np.abs(vecs[:, i])**2, color=colors[i % len(colors)], 
                     label=f'Mode {i+1} Effective index: {vals[i].real:.3f}')
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
    if npml > 0:
        # Create PML scaling array
        sc_array = np.ones(N, dtype=complex)
        
        # Define PML parameters
        amax = 3
        m = 3  # PML power
        
        # Left PML
        for i in range(npml):
            xn = (npml - i) / npml
            sc_array[i] = 1.0/(1.0 + 1j * amax * xn**m)
            
        # Right PML
        for i in range(npml):
            xn = (i + 1) / npml
            sc_array[N - 1 - i] = 1.0/(1.0 + 1j * amax * xn**m)
            
        # Create scaling matrices and apply to derivative operators
        sc_x = sp.diags(sc_array, 0)
        inv_sc_x = sp.diags(1/sc_array, 0)
        
        Dxf = inv_sc_x @ Dxf @ sc_x
        Dxb = inv_sc_x @ Dxb @ sc_x
    
    return Dxf, Dxb

def solver_eigs(A, m, guess_value=1.0):
    """Solve for eigenmodes of operator A.
    
    Args:
        A: Sparse linear operator matrix
        m: Number of eigenmodes to return
        guess_value: Estimate for the eigenvalues
    
    Returns:
        tuple: (values, vectors) eigenvalues and eigenvectors
    """
    values, vectors = spl.eigs(A, k=m, sigma=guess_value, which='LM')
    return values, vectors

def filter_modes(values, vectors, filters=None):
    """Filter modes based on criteria functions.
    
    Args:
        values: Array of effective index values
        vectors: Array of mode profiles
        filters: List of functions that return True for modes to keep
        
    Returns:
        tuple: (filtered_values, filtered_vectors)
    """
    # If no filters, just return original data
    if filters is None:
        return values, vectors
    
    # Initialize with all True
    keep_elements = np.ones(values.shape, dtype=bool)
    
    # Apply each filter
    for f in filters:
        keep_f = f(values)
        keep_elements = np.logical_and(keep_elements, keep_f)
    
    # Get indices to keep
    keep_indices = np.where(keep_elements)[0]
    
    # Return filtered values and vectors
    return values[keep_indices], vectors[:, keep_indices]

def normalize_modes(vectors):
    """Normalize each mode such that sum(|vec|^2)=1.
    
    Args:
        vectors: Array with shape (n_points, n_vectors)
        
    Returns:
        normalized_vectors: Normalized mode profiles
    """
    powers = np.sum(np.square(np.abs(vectors)), axis=0)
    return vectors / np.sqrt(powers)

def solve_modes(eps, omega, dL, npml=0, m=2, filtering=True):
    """Solve for the modes of a simple 1D waveguide.
    
    Args:
        eps: Permittivity profile
        omega: Angular frequency
        dL: Grid spacing
        npml: Number of PML layers
        m: Number of modes to compute
        filtering: Whether to filter out unphysical modes
        
    Returns:
        tuple: (vals, vecs) effective indices and mode profiles
    """
    k0 = omega / LIGHT_SPEED
    N = eps.size
    
    # Compute derivative matrices with PML
    Dxf, Dxb = compute_derivative_matrices(N, dL, npml)
    
    # Define the eigenvalue problem matrix
    diag_eps = sp.diags(eps.flatten(), 0)
    A = diag_eps + (Dxf @ Dxb) * (1 / k0) ** 2
    
    # Solve for eigenmodes
    n_max = np.sqrt(np.max(eps))
    vals, vecs = solver_eigs(A, m, guess_value=n_max**2)
    
    # Filter out unphysical modes if requested
    if filtering:
        # Define filter functions
        filter_re = lambda vals: np.real(vals) > 0.0
        filter_im = lambda vals: np.abs(np.imag(vals)) <= 1e-10
        filters = [filter_re, filter_im]
        vals, vecs = filter_modes(vals, vecs, filters=filters)
    
    # Normalize mode profiles
    vecs = normalize_modes(vecs)
    
    # Compute effective indices from eigenvalues
    neff = np.sqrt(vals)
    
    return neff, vecs

def solve_line_modes(eps_1d, start, end, omega, dL, npml=0, m=2, filtering=True):
    """Solve for modes along a 1D line in a permittivity profile.
    
    Args:
        eps_1d: 1D permittivity profile
        start: Starting point (x,y)
        end: End point (x,y)
        omega: Angular frequency
        dL: Grid spacing
        npml: Number of PML layers
        m: Number of modes to compute
        filtering: Whether to filter out unphysical modes
        
    Returns:
        tuple: (vals, vecs) effective indices and mode profiles
    """
    return solve_modes(eps_1d, omega, dL, npml, m, filtering)

def insert_mode(eps, start, end, omega, dL, npml=0, mode_number=0, target=None):
    """Insert a mode into a target field array.
    
    Args:
        eps: Permittivity profile
        start: Starting point (x,y)
        end: End point (x,y)
        omega: Angular frequency
        dL: Grid spacing
        npml: Number of PML layers
        mode_number: Which mode to insert (0-based)
        target: Target field array (if None, creates a new one)
        
    Returns:
        target: Field array with inserted mode
    """
    # Extract cross-section of permittivity
    x0, y0 = start
    x1, y1 = end
    num_points = int(np.hypot(x1 - x0, y1 - y0) / dL)
    x, y = np.linspace(x0, x1, num_points), np.linspace(y0, y1, num_points)
    
    # Create coords arrays
    x_coords = np.round(x / dL).astype(int)
    y_coords = np.round(y / dL).astype(int)
    
    # Extract permittivity profile
    eps_cross = np.zeros(num_points)
    for i, (xi, yi) in enumerate(zip(x_coords, y_coords)):
        eps_cross[i] = eps[xi, yi]
    
    # Create target array if not provided
    if target is None:
        target = np.zeros(eps.shape, dtype=complex)
    
    # Solve for modes
    _, mode_field = solve_modes(eps_cross, omega, dL, npml, m=mode_number+1)
    
    # Insert mode into target array
    for i, (xi, yi) in enumerate(zip(x_coords, y_coords)):
        target[xi, yi] = mode_field[i, mode_number]
    
    return target