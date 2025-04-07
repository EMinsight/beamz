class RegularGrid:
    """Takes in a design and resolution and returns a rasterized grid of that design."""
    def __init__(self, design, resolution):
        self.design = design
        self.resolution = resolution
        self.grid = None

    def rasterize(self):
        self.grid = self.design.rasterize(self.resolution)

    def aliazing(self):
        pass

    def show(self):
        pass



# RectilinearGrid

# ================================ UnstructuredGrid