# Medium: Dispersionless medium.
class Material:
    def __init__(self, permittivity, permeability, conductivity):
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