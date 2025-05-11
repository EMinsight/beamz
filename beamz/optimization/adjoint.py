import numpy as np
from beamz.simulation.fdtd import FDTD
from typing import Callable, Any, Optional, Dict, List, Union
import time

class AdjointOptimizer:
    def __init__(
        self, 
        simulation: FDTD, 
        objective_fn: Callable[[Dict[Any, Any]], float], 
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        filter_radius: float = None,
        projection_strength: float = None,
        callback: Optional[Callable[[int, float, np.ndarray], None]] = None
    ):
        """
        Initialize the adjoint optimizer.
        
        Args:
            simulation: FDTD simulation and design objects
            objective_fn: Function that calculates the figure of merit from simulation results
            learning_rate: Step size for gradient updates
            momentum: Momentum coefficient for gradient updates
            filter_radius: Radius for density filtering (smoothing)
            projection_strength: Strength of projection function (for binary designs)
            callback: Optional callback function called after each iteration
        """
        self.simulation = simulation
        self.objective_fn = objective_fn
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.filter_radius = filter_radius
        self.projection_strength = projection_strength
        self.callback = callback
        # Get design parameters from the simulation
        self._extract_design_parameters()
        # Initialize velocity for momentum updates
        self.velocity = np.zeros_like(self.design_parameters)
        
    def _extract_design_parameters(self):
        """Extract optimizable design parameters from the simulation."""
        # Find all design structures that are marked as optimizable
        self.design_parameters = []
        self.design_structures = []
        # Find all optimizable structures
        for structure in self.simulation.design.structures:
            if hasattr(structure, "optimize") and structure.optimize:
                #if hasattr(structure, "material") and hasattr(structure.material, "value"):
                self.design_structures.append(structure)
                print("Found optimizable structure:", structure)
        # Convert design parameters to numpy array
        self.design_parameters = np.array(self.design_parameters)
        # Store the grid positions of design parameters for spatial filtering
        self.design_positions = []
        for structure in self.design_structures:
            if hasattr(structure, "position"):
                self.design_positions.append(structure.position)
        self.design_positions = np.array(self.design_positions)
        
    def _forward_simulation(self):
        """Run the forward simulation and calculate objective value."""
        # Run the forward simulation
        sim_result = self.simulation.run(live=False, store_fields=True)
        # Calculate the objective function
        obj_value = self.objective_fn(sim_result)
        return sim_result, obj_value
    
    def _calculate_adjoint_source(self, sim_result):
        """Calculate the adjoint source based on the objective function derivative."""
        adjoint_sources = []
        for key, result in sim_result.items():
            if hasattr(result, "is_monitor") and result.is_monitor:
                # Create adjoint source at monitor position
                adjoint_source = {
                    "position": result.position,
                    "size": result.size,
                    "field_profile": result.E_field,  # Use recorded E-field
                    "time_profile": np.flip(result.time_points),  # Time-reversed
                }
                adjoint_sources.append(adjoint_source)
        
        return adjoint_sources
    
    def _adjoint_simulation(self, sim_result, forward_fields):
        # Calculate adjoint sources
        adjoint_sources = self._calculate_adjoint_source(sim_result)
        # Configure simulation for adjoint run
        adjoint_sim = self.simulation.copy()
        adjoint_sim.reset_sources()
        # Add adjoint sources to the simulation
        for source in adjoint_sources:
            adjoint_sim.add_adjoint_source(
                position=source["position"],
                size=source["size"],
                field_profile=source["field_profile"],
                time_profile=source["time_profile"]
            )
        # Run adjoint simulation
        adjoint_result = adjoint_sim.run(live=False, store_fields=True)
        adjoint_fields = adjoint_result.get_fields()
        # Calculate gradient using the overlap integral between forward and adjoint fields
        # The formula is: ∇F = Re{ ∫dt ∫dr λ*(r,T-t) · ∂L/∂ε · E(r,t) }
        gradients = np.zeros_like(self.design_parameters)
        # For each design element, compute the gradient
        for i, element in enumerate(self.design_elements):
            # Get spatial domain of this element
            element_region = self._get_element_region(element)
            # Compute the gradient for this element
            gradients[i] = self._compute_gradient(
                forward_fields, 
                adjoint_fields, 
                element_region, 
                element.material.value
            )
        
        return gradients
    
    def _get_element_region(self, element):
        """Get the spatial region of a design element on the simulation grid."""
        # This is a simplified implementation that would need to be adapted
        # to match the actual simulation grid structure
        x_min, y_min = element.position[0] - element.width/2, element.position[1] - element.height/2
        x_max, y_max = element.position[0] + element.width/2, element.position[1] + element.height/2
        # Get grid indices corresponding to this region
        grid = self.simulation.get_grid()
        x_indices = np.where((grid.x >= x_min) & (grid.x <= x_max))[0]
        y_indices = np.where((grid.y >= y_min) & (grid.y <= y_max))[0]
        
        return {"x": x_indices, "y": y_indices}
    
    def _compute_gradient(self, forward_fields, adjoint_fields, element_region, permittivity):
        """Compute gradient for a specific design element using the overlap integral."""
        # Extract fields in the element region
        E_forward = forward_fields.E[:, element_region["x"], element_region["y"]]
        E_adjoint = adjoint_fields.E[:, element_region["x"], element_region["y"]]
        # Time points must be reversed for adjoint fields
        E_adjoint = np.flip(E_adjoint, axis=0)
        # Compute the overlap integral. The gradient is proportional to: -∫dt E_forward(t) · ∂E_adjoint/∂t(T-t) / ε²
        dt = self.simulation.dt
        gradient = 0
        for t in range(len(forward_fields.time_points) - 1):
            # Calculate time derivative of adjoint field
            dE_adjoint_dt = (E_adjoint[t+1] - E_adjoint[t]) / dt
            # Contribution to gradient from this time step
            gradient += -np.sum(E_forward[t] * dE_adjoint_dt) * dt / (permittivity**2)
        
        return np.real(gradient)  # Ensure the result is real
    
    def _apply_filters(self, gradients):
        """Apply spatial filtering to the gradients if filter_radius is specified."""
        if self.filter_radius is None: return gradients
        # Apply spatial filtering to the gradients
        filtered_gradients = np.zeros_like(gradients)
        # Apply a Gaussian filter to each parameter based on spatial positions
        for i, pos_i in enumerate(self.design_positions):
            weights = np.zeros(len(self.design_positions))
            for j, pos_j in enumerate(self.design_positions):
                # Calculate distance between elements
                distance = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
                # Apply Gaussian weighting
                if distance <= self.filter_radius:
                    weights[j] = np.exp(-(distance**2) / (2 * self.filter_radius**2))
            # Normalize weights
            if np.sum(weights) > 0:
                weights /= np.sum(weights)
            # Apply weighted averaging to gradients
            filtered_gradients[i] = np.sum(weights * gradients)
        
        return filtered_gradients
    
    def _update_design(self, gradients):
        """Update design parameters using gradient information."""
        # Apply spatial filtering
        filtered_gradients = self._apply_filters(gradients)
        # Apply momentum-based update
        self.velocity = self.momentum * self.velocity + self.learning_rate * filtered_gradients
        self.design_parameters += self.velocity
        # Apply projection if specified
        if self.projection_strength is not None:
            # Projection function: pushes values toward 0 or 1
            # formula: (tanh(β*η) + tanh(β*(η-1))) / (tanh(β) + tanh(β*1))
            # where β is projection_strength and η is design_parameters
            beta = self.projection_strength
            numerator = np.tanh(beta * self.design_parameters) + np.tanh(beta * (self.design_parameters - 1))
            denominator = np.tanh(beta) + np.tanh(beta * 1)
            self.design_parameters = numerator / denominator
        # Apply constraints
        if self.min_value is not None:
            self.design_parameters = np.maximum(self.design_parameters, self.min_value)
        if self.max_value is not None:
            self.design_parameters = np.minimum(self.design_parameters, self.max_value)
        # Update the design elements
        for i, element in enumerate(self.design_elements):
            element.material.value = self.design_parameters[i]
    
    def optimize(self, iterations: int = 50) -> List[float]:
        """Run optimization for the specified number of iterations."""
        objective_history = []
        
        print(f"Starting optimization with {len(self.design_parameters)} design parameters")
        start_time = time.time()
        
        for i in range(iterations):
            iter_start = time.time()
            # Forward simulation
            sim_result, obj_value = self._forward_simulation()
            objective_history.append(obj_value)
            # Get forward fields
            forward_fields = sim_result.get_fields()
            # Calculate gradients using adjoint method
            gradients = self._adjoint_simulation(sim_result, forward_fields)
            # Update design parameters
            self._update_design(gradients)
            iter_time = time.time() - iter_start
            print(f"Iteration {i+1}/{iterations}, Objective: {obj_value:.6f}, Time: {iter_time:.2f}s")
            # Call callback if provided
            if self.callback: self.callback(i, obj_value, self.design_parameters)
        
        total_time = time.time() - start_time
        print(f"Optimization completed in {total_time:.2f}s")
        print(f"Final objective value: {objective_history[-1]:.6f}")
        
        return objective_history
