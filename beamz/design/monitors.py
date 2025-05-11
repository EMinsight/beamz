import numpy as np

class Monitor():
    """Monitors the fields along a line during an FDTD simulation."""
    def __init__(self, design=None, start=(0,0), end=(0,0), name=None):
        self.fields = {
            'Ez': [],
            'Hx': [],
            'Hy': [],
            't': []
        }
        self.power_accumulated = None
        self.power_accumulation_count = 0
        self.start = start
        self.end = end
        self.design = design
        self.name = name if name else f"Monitor_{id(self)}"
        self.position = ((start[0] + end[0])/2, (start[1] + end[1])/2)  # Center point
        
    def get_grid_points(self, dx, dy):
        """Collect the grid points along the line of the monitor
        
        Args:
            dx: Grid spacing in x direction
            dy: Grid spacing in y direction
            
        Returns:
            List of (x_idx, y_idx) grid points
        """
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
        """Record field data at the monitor location
        
        Args:
            Ez: Electric field
            Hx, Hy: Magnetic field components
            t: Current time
            dx, dy: Grid spacing
            save_memory: If True, only keep the latest field values
            accumulate_power: If True, accumulate power instead of saving fields
        """
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
            else:
                Ez_values.append(0.0)
                
            # Handle Hx (one row less than Ez)
            if 0 <= y_idx < Hx.shape[0] and 0 <= x_idx < Hx.shape[1]:
                Hx_values.append(float(Hx[y_idx, x_idx]))
            else:
                Hx_values.append(0.0)
                
            # Handle Hy (one column less than Ez)
            if 0 <= y_idx < Hy.shape[0] and 0 <= x_idx < Hy.shape[1]:
                Hy_values.append(float(Hy[y_idx, x_idx]))
            else:
                Hy_values.append(0.0)
        
        # Calculate power if needed
        if accumulate_power:
            # Extend Hx and Hy to match Ez dimensions if needed
            Sx = np.array([-Ez_val * Hy_val for Ez_val, Hy_val in zip(Ez_values, Hy_values)])
            Sy = np.array([Ez_val * Hx_val for Ez_val, Hx_val in zip(Ez_values, Hx_values)])
            
            # Calculate power magnitude (|S|²)
            power_mag = Sx**2 + Sy**2
            
            # Initialize or accumulate power
            if self.power_accumulated is None:
                self.power_accumulated = power_mag
            else:
                self.power_accumulated += power_mag
                
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

    def plot_fields(self, field='Ez', figsize=(10, 6)):
        """Plot the field data recorded by this monitor
        
        Args:
            field: Field to plot ('Ez', 'Hx', or 'Hy')
            figsize: Figure size tuple
            
        Returns:
            matplotlib figure and axes
        """
        import matplotlib.pyplot as plt
        
        if len(self.fields['t']) == 0:
            print(f"No field data recorded for monitor {self.name}")
            return None, None
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate monitor length for x-axis
        monitor_length = np.sqrt((self.end[0] - self.start[0])**2 + 
                               (self.end[1] - self.start[1])**2)
        
        # Create position array (normalized from 0 to 1 along monitor)
        num_points = len(self.fields[field][0])
        positions = np.linspace(0, monitor_length, num_points)
        
        # Create time array and convert to appropriate units
        times = np.array(self.fields['t'])
        max_time = np.max(times)
        
        if max_time < 1e-12:  # femtoseconds
            times_scaled = times * 1e15
            time_unit = 'fs'
        elif max_time < 1e-9:  # picoseconds
            times_scaled = times * 1e12
            time_unit = 'ps'
        else:  # nanoseconds
            times_scaled = times * 1e9
            time_unit = 'ns'
        
        # Calculate appropriate spatial unit
        if monitor_length < 1e-6:  # nanometers
            position_scaled = positions * 1e9
            position_unit = 'nm'
        elif monitor_length < 1e-3:  # micrometers
            position_scaled = positions * 1e6
            position_unit = 'µm'
        else:  # millimeters
            position_scaled = positions * 1e3
            position_unit = 'mm'
            
        # Create 2D data array for pcolormesh
        data = np.array(self.fields[field])
        
        # Plot the data
        im = ax.pcolormesh(position_scaled, times_scaled, data, 
                         shading='auto', cmap='RdBu_r')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f'{field} Amplitude')
        
        # Add labels and title
        ax.set_xlabel(f'Position along monitor ({position_unit})')
        ax.set_ylabel(f'Time ({time_unit})')
        ax.set_title(f'{field} Field at Monitor: {self.name}')
        
        return fig, ax
        
    def plot_power(self, figsize=(10, 6), log_scale=False, db_scale=False):
        """Plot the power data recorded by this monitor
        
        Args:
            figsize: Figure size tuple
            log_scale: If True, use logarithmic color scale
            db_scale: If True, use dB scale (10*log10)
            
        Returns:
            matplotlib figure and axes
        """
        import matplotlib.pyplot as plt
        
        # Check if we have accumulated power
        if self.power_accumulated is not None and self.power_accumulation_count > 0:
            # Use accumulated power
            power = self.power_accumulated / self.power_accumulation_count
            
            # Calculate monitor length for x-axis
            monitor_length = np.sqrt((self.end[0] - self.start[0])**2 + 
                                  (self.end[1] - self.start[1])**2)
            
            # Create position array
            num_points = len(power)
            positions = np.linspace(0, monitor_length, num_points)
            
            # Calculate appropriate spatial unit
            if monitor_length < 1e-6:  # nanometers
                position_scaled = positions * 1e9
                position_unit = 'nm'
            elif monitor_length < 1e-3:  # micrometers
                position_scaled = positions * 1e6
                position_unit = 'µm'
            else:  # millimeters
                position_scaled = positions * 1e3
                position_unit = 'mm'
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Apply log scale if requested
            if db_scale and np.max(power) > 0:
                # Convert to dB
                min_power = np.max(power) * 1e-10  # -100 dB floor
                power_db = 10 * np.log10(np.maximum(power, min_power) / np.max(power))
                ax.plot(position_scaled, power_db)
                ax.set_ylabel('Power (dB)')
            elif log_scale and np.min(power) > 0:
                ax.semilogy(position_scaled, power)
                ax.set_ylabel('Power (log scale)')
            else:
                ax.plot(position_scaled, power)
                ax.set_ylabel('Power')
                
            # Add labels and title
            ax.set_xlabel(f'Position along monitor ({position_unit})')
            ax.set_title(f'Average Power at Monitor: {self.name}')
            ax.grid(True)
            
            return fig, ax
        elif len(self.fields['Ez']) > 0:
            # Calculate power from field data
            power_over_time = []
            
            for t_idx in range(len(self.fields['t'])):
                Ez = np.array(self.fields['Ez'][t_idx])
                Hx = np.array(self.fields['Hx'][t_idx])
                Hy = np.array(self.fields['Hy'][t_idx])
                
                # Calculate Poynting vector components
                Sx = -Ez * Hy
                Sy = Ez * Hx
                
                # Calculate power magnitude
                power = Sx**2 + Sy**2
                power_over_time.append(power)
            
            # Convert to numpy array
            power_over_time = np.array(power_over_time)
            
            # Average power over time
            avg_power = np.mean(power_over_time, axis=0)
            
            # Calculate monitor length for x-axis
            monitor_length = np.sqrt((self.end[0] - self.start[0])**2 + 
                                  (self.end[1] - self.start[1])**2)
            
            # Create position array
            num_points = len(avg_power)
            positions = np.linspace(0, monitor_length, num_points)
            
            # Calculate appropriate spatial unit
            if monitor_length < 1e-6:  # nanometers
                position_scaled = positions * 1e9
                position_unit = 'nm'
            elif monitor_length < 1e-3:  # micrometers
                position_scaled = positions * 1e6
                position_unit = 'µm'
            else:  # millimeters
                position_scaled = positions * 1e3
                position_unit = 'mm'
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Apply log scale if requested
            if db_scale and np.max(avg_power) > 0:
                # Convert to dB
                min_power = np.max(avg_power) * 1e-10  # -100 dB floor
                power_db = 10 * np.log10(np.maximum(avg_power, min_power) / np.max(avg_power))
                ax.plot(position_scaled, power_db)
                ax.set_ylabel('Power (dB)')
            elif log_scale and np.min(avg_power) > 0:
                ax.semilogy(position_scaled, avg_power)
                ax.set_ylabel('Power (log scale)')
            else:
                ax.plot(position_scaled, avg_power)
                ax.set_ylabel('Power')
                
            # Add labels and title
            ax.set_xlabel(f'Position along monitor ({position_unit})')
            ax.set_title(f'Average Power at Monitor: {self.name}')
            ax.grid(True)
            
            return fig, ax
        else:
            print(f"No field data recorded for monitor {self.name}")
            return None, None
    
    def add_to_plot(self, ax):
        ax.plot((self.start[0], self.end[0]), (self.start[1], self.end[1]), '-', lw=4, color="navy", label='Monitor')
        ax.plot((self.start[0], self.end[0]), (self.start[1], self.end[1]), '-', lw=2, color="black", label='Monitor')