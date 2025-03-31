import numpy as np

# PointSource: Uniform current source with a zero size.

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

# GaussianPulse: Source time dependence that describes a Gaussian pulse.

# CustomSourceTime: Custom source time dependence consisting of a real or complex envelope modulated at a central frequency, as shown below.