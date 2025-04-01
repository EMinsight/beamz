"""
Example of saving and loading simulation configurations.
"""

from beamz.sim import Simulation

def main():
    # Create a simulation
    sim = Simulation(
        type="2D",
        size=(100, 100),
        cell_size=0.1,
        dt=0.1,
        time=1.0
    )
    
    # Add materials
    sim.add_material(
        name="air",
        permittivity=1.0,
        permeability=1.0,
        conductivity=0.0
    )
    sim.add_material(
        name="metal",
        permittivity=1.0,
        permeability=1.0,
        conductivity=1e6
    )
    
    # Add sources
    sim.add_source(
        type="gaussian",
        position=(50, 50),
        amplitude=1.0,
        frequency=1.0,
        width=5.0
    )
    
    # Add boundaries
    sim.add_boundary(
        type="pml",
        thickness=10,
        direction="all"
    )
    
    # Save the configuration
    sim.save_config("simulation_config.json")
    
    # Load the configuration into a new simulation
    new_sim = Simulation.load_config("simulation_config.json")
    
    # Verify the configurations match
    print("Original simulation:")
    print(f"Type: {sim.type}")
    print(f"Size: {sim.size}")
    print(f"Materials: {sim.materials}")
    print(f"Sources: {sim.sources}")
    print(f"Boundaries: {sim.boundaries}")
    
    print("\nLoaded simulation:")
    print(f"Type: {new_sim.type}")
    print(f"Size: {new_sim.size}")
    print(f"Materials: {new_sim.materials}")
    print(f"Sources: {new_sim.sources}")
    print(f"Boundaries: {new_sim.boundaries}")

if __name__ == "__main__":
    main() 