# Medium: Dispersionless medium.
class Material:
    def __init__(self, permittivity=1.0, permeability=1.0, conductivity=0.0):
        self.permittivity = permittivity
        self.permeability = permeability
        self.conductivity = conductivity

class VariableMaterial:
    def __init__(self, permittivity_min=1.0, permittivity_max=1.0, permeability_min=1.0,
                 permeability_max=1.0, conductivity_min=0.0, conductivity_max=0.0):
        self.permittivity_min = permittivity_min
        self.permittivity_max = permittivity_max
        self.permeability_min = permeability_min
        self.permeability_max = permeability_max
        self.conductivity_min = conductivity_min
        self.conductivity_max = conductivity_max

# CustomMedium: Medium with user-supplied permittivity distribution.

# ================================

# PoleResidue: A dispersive medium described by the pole-residue pair model.

# Lorentz: A dispersive medium described by the Lorentz model.

# Sellmeier: A dispersive medium described by the Sellmeier model.

# Drude: A dispersive medium described by the Drude model.

# Debye: A dispersive medium described by the Debye model.