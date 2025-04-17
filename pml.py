    def _setup_pml(self):
        """Set up a stable PML implementation using a simple lossy layer approach."""
        # PML parameters
        self.pml_width = max(20, int(min(self.nx, self.ny) * 0.1))  # Min 20 cells or 10% of domain
        print(f"PML width: {self.pml_width} cells")
        
        # Create arrays to hold PML coefficients
        self.pml_x = np.ones((self.nx, self.ny))  # X-direction PML
        self.pml_y = np.ones((self.nx, self.ny))  # Y-direction PML
        
        # Initialize pml profile
        pml_profile = np.ones((self.pml_width))
        
        # Create polynomial grading (cubic) - proven to be stable
        for i in range(self.pml_width):
            d = i / self.pml_width
            pml_profile[i] = 1.0 - d**3
        
        # Calculate max conductivity based on grid and material
        eps_avg = np.mean(self.epsilon_r)
        # Calculate the impedance
        eta = np.sqrt(MU_0 / (EPS_0 * eps_avg))
        
        # Set max conductivity - conservative value for stability
        sigma_max = 0.5 / (eta * self.dx)
        
        # Apply PML in x-direction (left and right boundaries)
        for i in range(self.pml_width):
            sigma = sigma_max * pml_profile[i]
            self.pml_x[i, :] = 1.0 / (1.0 + sigma * self.dt / (2.0 * EPS_0 * self.epsilon_r[i, :]))
            self.pml_x[-(i+1), :] = 1.0 / (1.0 + sigma * self.dt / (2.0 * EPS_0 * self.epsilon_r[-(i+1), :]))
        
        # Apply PML in y-direction (top and bottom boundaries)
        for j in range(self.pml_width):
            sigma = sigma_max * pml_profile[j]
            self.pml_y[:, j] = 1.0 / (1.0 + sigma * self.dt / (2.0 * EPS_0 * self.epsilon_r[:, j]))
            self.pml_y[:, -(j+1)] = 1.0 / (1.0 + sigma * self.dt / (2.0 * EPS_0 * self.epsilon_r[:, -(j+1)]))
        
        print(f"PML configured: max sigma = {sigma_max:.2e}")