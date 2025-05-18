import numpy as np

class Monitor():
    """Monitors the fields along a line during an FDTD simulation."""
    def __init__(self, design=None, start=(0,0), end=(0,0), record_fields=True, accumulate_power=True):
        self.fields = {'Ez': [], 'Hx': [], 'Hy': [], 't': []}
        self.power_accumulated = None
        self.power_accumulation_count = 0
        self.start = start
        self.end = end
        self.design = design
        self.position = ((start[0] + end[0])/2, (start[1] + end[1])/2)  # Center point
        
    def get_grid_points(self, dx, dy):
        """Collect the grid points along the line of the monitor"""
        # Convert physical coordinates to grid indices
        start_x_grid = int(round(self.start[0] / dx))
        start_y_grid = int(round(self.start[1] / dy))
        end_x_grid = int(round(self.end[0] / dx))
        end_y_grid = int(round(self.end[1] / dy))
        # Create a line of points between start and end
        if abs(end_x_grid - start_x_grid) > abs(end_y_grid - start_y_grid):
            # Horizontal line dominant
            num_points = abs(end_x_grid - start_x_grid) + 1
            x_indices = np.linspace(start_x_grid, end_x_grid, num_points, dtype=int)
            y_indices = np.linspace(start_y_grid, end_y_grid, num_points, dtype=int)
        else:
            # Vertical line dominant
            num_points = abs(end_y_grid - start_y_grid) + 1
            x_indices = np.linspace(start_x_grid, end_x_grid, num_points, dtype=int)
            y_indices = np.linspace(start_y_grid, end_y_grid, num_points, dtype=int)
        # Return list of grid points
        return list(zip(x_indices, y_indices))
    
    def record_fields(self, Ez, Hx, Hy, t, dx, dy, save_memory=False, accumulate_power=False):
        """Record field data at the monitor location."""
        # Get grid points along monitor line
        grid_points = self.get_grid_points(dx, dy)
        # Extract field values at these points
        Ez_values = []
        Hx_values = []
        Hy_values = []
        for x_idx, y_idx in grid_points:
            # Bounds checking
            if 0 <= y_idx < Ez.shape[0] and 0 <= x_idx < Ez.shape[1]:
                Ez_values.append(float(Ez[y_idx, x_idx]))
            else: Ez_values.append(0.0)
            # Handle Hx (one row less than Ez)
            if 0 <= y_idx < Hx.shape[0] and 0 <= x_idx < Hx.shape[1]:
                Hx_values.append(float(Hx[y_idx, x_idx]))
            else: Hx_values.append(0.0)
            # Handle Hy (one column less than Ez)
            if 0 <= y_idx < Hy.shape[0] and 0 <= x_idx < Hy.shape[1]:
                Hy_values.append(float(Hy[y_idx, x_idx]))
            else: Hy_values.append(0.0)
        
        # Calculate power if needed
        if accumulate_power:
            # Extend Hx and Hy to match Ez dimensions if needed
            Sx = np.array([-Ez_val * Hy_val for Ez_val, Hy_val in zip(Ez_values, Hy_values)])
            Sy = np.array([Ez_val * Hx_val for Ez_val, Hx_val in zip(Ez_values, Hx_values)])
            # Calculate power magnitude (|S|²)
            power_mag = Sx**2 + Sy**2
            # Initialize or accumulate power
            if self.power_accumulated is None: self.power_accumulated = power_mag
            else: self.power_accumulated += power_mag
            self.power_accumulation_count += 1
            
        # Save field data if not only accumulating power or in memory saving mode
        if not save_memory:
            self.fields['Ez'].append(Ez_values)
            self.fields['Hx'].append(Hx_values)
            self.fields['Hy'].append(Hy_values)
            self.fields['t'].append(t)
        elif not accumulate_power:
            # In memory saving mode, keep only the latest values
            self.fields['Ez'] = [Ez_values]
            self.fields['Hx'] = [Hx_values]
            self.fields['Hy'] = [Hy_values]
            self.fields['t'] = [t]
    
    def add_to_plot(self, ax, facecolor="navy", edgecolor="navy", alpha=1, linestyle="-"):
        ax.plot((self.start[0], self.end[0]), (self.start[1], self.end[1]), '-', lw=4, color=facecolor, label='Monitor')
        ax.plot((self.start[0], self.end[0]), (self.start[1], self.end[1]), '-', lw=1, color=edgecolor)