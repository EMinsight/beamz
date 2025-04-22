class Monitor():
    """Monitors the fields along a line during an FDTD simulation."""
    def __init__(self, start, end):
        self.E_z = []
        self.H_x = []
        self.H_y = []
        self.start = start
        self.end = end
        self.grid_points = []

    def get_grid_points(self):
        """Collect the grid points along the line of the monitor"""

    def detect(self, E_z, H_x, H_y):
        """Detect the E_z, H_x, H_y field values along the line of the monitor."""