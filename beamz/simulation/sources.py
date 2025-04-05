class PointSource():
    """Uniform current source with a zero size."""
    def __init__(self, position=(0,0), signal=0):
        self.position = position
        self.signal = signal
    

# ModeSource: Injects current source to excite modal profile on finite extent plane.

# PlaneWave: Uniform current distribution on an infinite extent plane.