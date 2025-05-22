# import a gds file
from beamz.design import Design, Material, Polygon
from beamz.design.io import import_gds_as_polygons
from beamz.const import µm

# Create a design
design = Design(width=5*µm, height=5*µm, material=Material(1), pml_size=0)
# Import gds-file
gds_polygons = import_gds_as_polygons("test.gds")
## Add the polygons to the design
for p in gds_polygons: design += Polygon(vertices=p.vertices, material=Material(1.444**2)).scale(1*µm)
# Show the design
design.show()