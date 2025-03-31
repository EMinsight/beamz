"""
Material library for BeamZ.
"""

class MaterialLibrary:
    """Library of materials and their properties."""
    
    def __init__(self):
        """Initialize the material library."""
        self.materials = {}
        
    def add_material(self, name, permittivity=1.0, permeability=1.0, conductivity=0.0):
        """Add a material to the library.
        
        Args:
            name (str): Name of the material
            permittivity (float): Relative permittivity
            permeability (float): Relative permeability
            conductivity (float): Electrical conductivity
        """
        self.materials[name] = {
            'permittivity': permittivity,
            'permeability': permeability,
            'conductivity': conductivity
        }
        
    def get_material(self, name):
        """Get material properties by name.
        
        Args:
            name (str): Name of the material
            
        Returns:
            dict: Material properties
            
        Raises:
            KeyError: If material not found
        """
        return self.materials[name]
        
    def __getitem__(self, name):
        """Allow dictionary-style access to materials."""
        return self.get_material(name)
        
    def __contains__(self, name):
        """Check if material exists in library."""
        return name in self.materials 
    

class Material():
    def __init__(self, name, permittivity, permeability, conductivity, color):
        self.name = name
        self.permittivity = permittivity
        self.permeability = permeability
        self.conductivity = conductivity
        self.color = color


# Medium: Dispersionless medium.
class Medium(Material):
    def __init__(self, name, permittivity, permeability, conductivity, color):
        super().__init__(name, permittivity, permeability, conductivity, color) 

# CustomMedium: Medium with user-supplied permittivity distribution.

# ================================

# PoleResidue: A dispersive medium described by the pole-residue pair model.

# Lorentz: A dispersive medium described by the Lorentz model.

# Sellmeier: A dispersive medium described by the Sellmeier model.

# Drude: A dispersive medium described by the Drude model.

# Debye: A dispersive medium described by the Debye model.

