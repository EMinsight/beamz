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

