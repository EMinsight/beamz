import gdspy
import os

# Ensure the target directory exists
output_dir = "tests/test_data"
os.makedirs(output_dir, exist_ok=True)
output_gds_file = os.path.join(output_dir, "test_import.gds")

print(f"Creating GDS file: {output_gds_file}")

# Create a new GDS library
lib = gdspy.GdsLibrary(unit=1e-6, precision=1e-9) # Unit: 1 micrometer, Precision: 1 nanometer

# Create a cell
main_cell = lib.new_cell("MAIN_CELL")

# Define polygon 1 for layer 1
poly1_vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
poly1 = gdspy.Polygon(poly1_vertices, layer=1)
main_cell.add(poly1)
print(f"Added polygon with vertices {poly1_vertices} to layer 1")

# Define polygon 2 for layer 2
poly2_vertices = [(2, 2), (3, 2), (3, 3), (2, 3)]
poly2 = gdspy.Polygon(poly2_vertices, layer=2)
main_cell.add(poly2)
print(f"Added polygon with vertices {poly2_vertices} to layer 2")

# Write the GDS file
lib.write_gds(output_gds_file)
print(f"Successfully wrote GDS file to {output_gds_file}")

# Minimal verification
print("\nCell info after adding polygons:")
print(main_cell)

# Verify file creation by listing directory contents (optional, for debugging)
if os.path.exists(output_gds_file):
    print(f"File {output_gds_file} confirmed to exist.")
    print(f"File size: {os.path.getsize(output_gds_file)} bytes.")
else:
    print(f"ERROR: File {output_gds_file} was NOT created.")

print("\nContents of tests/test_data:")
for item in os.listdir(output_dir):
    print(f"  - {item}")
print(f"Full path of GDS file: {os.path.abspath(output_gds_file)}")

print("Script finished.")
