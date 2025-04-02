import numpy as np
from typing import Tuple

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
    def __init__(self, direction: Tuple[float, float], amplitude: float, frequency: float,
                 ramp_up_time: float = 0, ramp_down_time: float = 0, phase: float = 0):
        """Initialize a wave source.
        
        Args:
            direction (Tuple[float, float]): Direction vector (x, y)
            amplitude (float): Wave amplitude
            frequency (float): Wave frequency in Hz
            ramp_up_time (float): Time to ramp up the wave
            ramp_down_time (float): Time to ramp down the wave
            phase (float): Initial phase
        """
        self.direction = direction
        self.amplitude = amplitude
        self.frequency = frequency
        self.ramp_up_time = ramp_up_time
        self.ramp_down_time = ramp_down_time
        self.phase = phase

    def get_amplitude(self, t: float) -> float:
        """Get the wave amplitude at time t.
        
        Args:
            t (float): Current time
            
        Returns:
            float: Wave amplitude
        """
        # Calculate ramp factor
        if t < self.ramp_up_time:
            ramp_factor = t / self.ramp_up_time
        else:
            ramp_factor = 1.0
            
        # Calculate wave
        wave = np.sin(2 * np.pi * self.frequency * t + self.phase)
        
        return self.amplitude * ramp_factor * wave

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