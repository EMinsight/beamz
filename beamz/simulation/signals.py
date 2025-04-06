import math as m
import numpy as np
import matplotlib.pyplot as plt

def cosine(t,amplitude, frequency, phase):
    return amplitude * np.cos(2 * np.pi * frequency * t + phase)

def sigmoid(t, duration=1, min=0, max=1, t0=0):
    return min + (max - min) * (1 / (1 + np.exp(-10 * (t - duration/2 - t0) / duration)))

# GaussianPulse: Source time dependence that describes a Gaussian pulse.

# CustomSourceTime: Custom source time dependence consisting of a real or complex envelope modulated at a central frequency, as shown below.



def plot_signal(signal, t):
    # Convert time to seconds
    t_seconds = t
    # Determine appropriate time unit and scaling factor
    if t_seconds[-1] < 1e-12:
        t_scaled = t_seconds * 1e15  # Convert to fs
        unit = 'fs'
    elif t_seconds[-1] < 1e-9:  # Less than 1 ns
        t_scaled = t_seconds * 1e12  # Convert to ps
        unit = 'ps'
    elif t_seconds[-1] < 1e-6:  # Less than 1 µs
        t_scaled = t_seconds * 1e9   # Convert to ns
        unit = 'ns'
    elif t_seconds[-1] < 1e-3:  # Less than 1 ms
        t_scaled = t_seconds * 1e6   # Convert to µs
        unit = 'µs'
    elif t_seconds[-1] < 1:     # Less than 1 s
        t_scaled = t_seconds * 1e3   # Convert to ms
        unit = 'ms'
    else:
        t_scaled = t_seconds
        unit = 's'
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_scaled, signal, color='black')
    ax.set_xlim(t_scaled[0], t_scaled[-1])
    ax.set_xlabel(f'Time ({unit})')
    ax.set_ylabel('Amplitude')
    ax.set_title('Signal')
    plt.tight_layout()
    plt.show()