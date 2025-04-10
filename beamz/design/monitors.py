from beamz.const import µm, LIGHT_SPEED

class ModeMonitor():
    """Monitors the mode profiles at a given point."""
    def __init__(self, design, start, end, wavelength=1.55*µm):
        self.start = start
        self.end = end
        self.wavelength = wavelength
        self.design = design

    def get_mode_profiles(self, eps_2d):
        """Calculate the mode profiles for a cross section given a 2D permittivity profile."""
        pass