# Medium: Dispersionless medium.
class Material:
    def __init__(self, permittivity=1.0, permeability=1.0, conductivity=0.0):
        self.permittivity = permittivity
        self.permeability = permeability
        self.conductivity = conductivity

class VariableMaterial:
    def __init__(self, permittivity:list=[1,1], permeability:list=[1,1], conductivity:list=[0,0]):
        self.permittivity = permittivity
        self.permeability = permeability
        self.conductivity = conductivity

# CustomMedium: Medium with user-supplied permittivity distribution.

# ================================

# PoleResidue: A dispersive medium described by the pole-residue pair model.

# Lorentz: A dispersive medium described by the Lorentz model.

# Sellmeier: A dispersive medium described by the Sellmeier model.

# Drude: A dispersive medium described by the Drude model.

# Debye: A dispersive medium described by the Debye model.


# ================================

# Material Library

# Vacuum

# Air

# SiN

# SiO2

# Si3N4

# Gold

# Aluminum

# Copper

