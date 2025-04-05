import math as m
import numpy as np
import matplotlib.pyplot as plt

class Wave:
    def __init__(self, amplitude=1, frequency=1, phase=0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

    def get_amplitude(self, t):
        """Get the wave amplitude at time t."""
        return self.amplitude * np.cos(2 * np.pi * self.frequency * t + self.phase)
    

# TODO:
class SigmoidRamp:
    def __init__(self, carrier=None, duration=0, padding=0):
        self.carrier = carrier
        self.duration = duration
        self.padding = padding

    def sigmoid(self, t, k=10):
        """Sigmoid function centered at duration/2 with steepness k."""
        # Compute the argument
        x = k * (t - self.duration/2)
        mask_neg = x < -100
        mask_pos = x > 100
        mask_mid = ~(mask_neg | mask_pos)
        result = np.zeros_like(x)
        result[mask_neg] = 0
        result[mask_pos] = 1
        result[mask_mid] = 1 / (1 + np.exp(-x[mask_mid]))
        return result

    def ramp_up_sigmoid(self, t):
        """Ramp up from 0 to 1 using sigmoid."""
        result = np.zeros_like(t)
        mask = (t >= 0) & (t <= self.duration)
        result[mask] = self.sigmoid(t[mask])
        result[t > self.duration] = 1
        return result

    def ramp_down_sigmoid(self, t):
        """Ramp down from 1 to 0 using sigmoid."""
        result = np.ones_like(t)
        mask = (t >= self.padding) & (t <= self.duration + self.padding)
        result[mask] = 1 - self.sigmoid(t[mask] - self.padding)
        result[t > self.duration + self.padding] = 0
        return result

    def get_amplitude(self, t):
        """Get the wave amplitude at time t."""
        if self.carrier is None:
            return np.zeros_like(t)
        # Get the carrier signal
        carrier_signal = self.carrier.get_amplitude(t)
        # Get the ramp up and down factors
        ramp_up = self.ramp_up_sigmoid(t)
        ramp_down = self.ramp_down_sigmoid(t)
        # Apply the ramps while preserving the sign of the carrier signal
        return carrier_signal * ramp_up * ramp_down
    


# GaussianPulse: Source time dependence that describes a Gaussian pulse.

# CustomSourceTime: Custom source time dependence consisting of a real or complex envelope modulated at a central frequency, as shown below.