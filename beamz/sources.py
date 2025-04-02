import numpy as np

# PointSource: Uniform current source with a zero size.
class PointSource():
    def __init__(self, position, signal):
        self.position = position
        self.signal = signal
    

# ModeSource: Injects current source to excite modal profile on finite extent plane.

# PlaneWave: Uniform current distribution on an infinite extent plane.


# ================================

# Wave: Source time dependence that ramps up to continuous oscillation and holds until end of simulation.
class Wave():
    def __init__(self, frequency, amplitude, phase=0):
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase

    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))
    

def add_source():
    """Add a normalized Gaussian pulse source"""
    t0 = 30e-15  # Pulse center
    sigma = 10e-15  # Pulse width
    # Add amplitude scaling factor
    amplitude = 1.0  # Normalize the source amplitude
    ##self.Ez[self.source_x, self.source_y] += amplitude * np.exp(-((self.t - t0)**2)/(2*sigma**2))
    #self.t += self.dt

# GaussianPulse: Source time dependence that describes a Gaussian pulse.

# CustomSourceTime: Custom source time dependence consisting of a real or complex envelope modulated at a central frequency, as shown below.