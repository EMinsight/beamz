# import a gds file
from beamz.design import Design, Material, Rectangle, Polygon
from beamz.design.io import import_gds_as_polygons
polygons = import_gds_as_polygons("test.gds")
print(polygons)

# Create a design
design = Design(pml_size=0)

## Add the polygons to the design
#for polygon in polygons:
#    design += Polygon(polygon, material=Material(1.444**2))

# Show the design
design.show()