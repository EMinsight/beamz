import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MatplotlibRectangle, PathPatch, Circle as MatplotlibCircle
from matplotlib.path import Path
import random
import numpy as np
from beamz.design.materials import Material
from beamz import viz as viz
from beamz.const import µm, EPS_0, MU_0
from beamz.design.sources import ModeSource, GaussianSource
from beamz.design.monitors import Monitor
from beamz.helpers import display_status, tree_view, get_si_scale_and_label
import colorsys
import gdspy

class Design:
    def __init__(self, width=5*µm, height=5*µm, depth=0, material=None, color=None, border_color="black", auto_pml=True,
                        pml_size=None):
        if material is None: material = Material(permittivity=1.0, permeability=1.0, conductivity=0.0)
        self.structures = [Rectangle(position=(0,0,0), width=width, height=height, depth=depth, material=material, color=color)]
        self.sources = []
        self.monitors = []
        self.boundaries = []
        self.width = width
        self.height = height
        self.depth = depth
        self.border_color = border_color
        self.time = 0
        self.is_3d = False if depth is None else True
        self.layers: dict[int, list[Polygon]] = {}
        self.is_3d = False if depth is None or depth == 0 else True
        if auto_pml: self.init_boundaries(pml_size)
        display_status(f"Created design with size: {self.width:.2e} x {self.height:.2e} m")

    def import_gds(gds_file: str, default_depth=1e-6):
        """Import a GDS file and return polygon and layer data.
        
        Args:
            gds_file (str): Path to the GDS file
            default_depth (float): Default depth/thickness for imported structures in meters
        """
        gds_lib = gdspy.GdsLibrary(infile=gds_file)
        design = Design()  # Create Design instance
        cells = gds_lib.cells  # Get all cells from the library
        total_polygons_imported = 0
        
        for _cell_name, cell in cells.items():
            # Get polygons by spec, which returns a dict: {(layer, datatype): [poly1_points, poly2_points,...]}
            gdspy_polygons_by_spec = cell.get_polygons(by_spec=True)
            for (layer_num, _datatype), list_of_polygon_points in gdspy_polygons_by_spec.items():
                if layer_num not in design.layers: design.layers[layer_num] = []
                for polygon_points in list_of_polygon_points:
                    # Convert points from microns to meters and ensure CCW ordering
                    vertices_2d = [(point[0] * 1e-6, point[1] * 1e-6) for point in polygon_points]
                    # Create polygon with appropriate depth
                    beamz_polygon = Polygon(vertices=vertices_2d, depth=default_depth)
                    design.layers[layer_num].append(beamz_polygon)
                    design.structures.append(beamz_polygon)
                    total_polygons_imported += 1
        
        # Set 3D flag if we have depth
        if default_depth > 0:
            design.is_3d = True
            design.depth = default_depth
            
        print(f"Imported {total_polygons_imported} polygons from '{gds_file}' into Design object.")
        if design.is_3d: print(f"3D design with depth: {design.depth:.2e} m")
        return design
    
    def export_gds(self, output_file):
        """Export a BEAMZ design (including only the structures, not sources or monitors) to a GDS file.
        
        For 3D designs, structures with the same material that touch (in 3D) will be placed in the same layer.
        """
        # Create library with micron units (1e-6) and nanometer precision (1e-9)
        lib = gdspy.GdsLibrary(unit=1e-6, precision=1e-9)
        cell = lib.new_cell("main")
        
        # First, we unify the polygons given their material and if they touch
        self.unify_polygons()
        
        # Scale factor to convert from meters to microns
        scale = 1e6  # 1 meter = 1e6 microns
        
        # Group structures by material properties
        material_groups = {}
        for structure in self.structures:
            # Skip PML visualizations, sources, monitors
            if hasattr(structure, 'is_pml') and structure.is_pml: continue
            if isinstance(structure, (ModeSource, GaussianSource, Monitor)): continue
            
            # Create material key based on material properties
            material = getattr(structure, 'material', None)
            if material is None: continue
                
            material_key = (
                getattr(material, 'permittivity', 1.0),
                getattr(material, 'permeability', 1.0),
                getattr(material, 'conductivity', 0.0)
            )
            
            if material_key not in material_groups: material_groups[material_key] = []
            material_groups[material_key].append(structure)
        
        # Export each material group as a separate layer
        for layer_num, (material_key, structures) in enumerate(material_groups.items()):
            for structure in structures:
                # Get vertices based on structure type
                if isinstance(structure, Polygon):
                    vertices = structure.vertices
                    interiors = structure.interiors if hasattr(structure, 'interiors') else []
                elif isinstance(structure, Rectangle):
                    x, y = structure.position[0:2]  # Take only x,y from position
                    w, h = structure.width, structure.height
                    vertices = [(x, y, 0), (x + w, y, 0), (x + w, y + h, 0), (x, y + h, 0)]
                    interiors = []
                elif isinstance(structure, (Circle, Ring, CircularBend, Taper)):
                    if hasattr(structure, 'to_polygon'):
                        poly = structure.to_polygon()
                        vertices = poly.vertices
                        interiors = getattr(poly, 'interiors', [])
                    else: continue
                else: continue
                
                # Project vertices to 2D and scale to microns
                vertices_2d = [(x * scale, y * scale) for x, y, _ in vertices]
                if not vertices_2d: continue
                
                # Scale and project interiors if they exist
                interior_2d = []
                if interiors: 
                    for interior in interiors:
                        interior_2d.append([(x * scale, y * scale) for x, y, _ in interior])
                
                try:
                    # Create gdspy polygon for this layer
                    if interior_2d: gdspy_poly = gdspy.Polygon(vertices_2d, layer=layer_num, holes=interior_2d)
                    else: gdspy_poly = gdspy.Polygon(vertices_2d, layer=layer_num)
                    cell.add(gdspy_poly)
                except Exception as e:
                    print(f"Warning: Failed to create GDS polygon: {e}")
                    continue
        
        # Write the GDS file
        lib.write_gds(output_file)
        print(f"GDS file saved as '{output_file}' with {len(material_groups)} material-based layers")
        # Print material information for each layer
        for layer_num, (material_key, structures) in enumerate(material_groups.items()):
            print(f"Layer {layer_num}: εᵣ={material_key[0]:.1f}, μᵣ={material_key[1]:.1f}, σ={material_key[2]:.2e} S/m")
        display_status(f"Created design with size: {self.width:.2e} x {self.height:.2e} x {self.depth:.2e} m")
    
    def add(self, structure):
        """Core add function for adding structures on top of the design."""
        if isinstance(structure, ModeSource):
            self.sources.append(structure)
            self.structures.append(structure)
        elif isinstance(structure, GaussianSource):
            self.sources.append(structure)
            self.structures.append(structure)
        elif isinstance(structure, Monitor):
            self.monitors.append(structure)
            self.structures.append(structure)
        else: self.structures.append(structure)
        
        # Check for 3D structures - improved detection
        if hasattr(structure, 'depth') and structure.depth != 0: self.is_3d = True
        elif hasattr(structure, 'position') and len(structure.position) > 2 and structure.position[2] != 0: self.is_3d = True
        elif hasattr(structure, 'vertices') and structure.vertices:
            # Check if any vertex has non-zero z coordinate
            for vertex in structure.vertices:
                if len(vertex) > 2 and vertex[2] != 0:
                    self.is_3d = True
                    break

    def __iadd__(self, structure):
        """Implement += operator for adding structures."""
        self.add(structure)
        return self
    
    def unify_polygons(self):
        """If polygons are the same material and overlap spatially, unify them into a single, simplified polygon."""
        try:
            from shapely.geometry import Polygon as ShapelyPolygon
            from shapely.ops import unary_union
        except ImportError:
            display_status("Shapely library is required for polygon unification. \
                            Please install with: pip install shapely", "error")
            return False
            
        # Group structures by material properties
        material_groups = {}
        non_polygon_structures = []
        # Track which structures to remove later
        structures_to_remove = []
        # First pass: group polygons by material
        for structure in self.structures:
            # Skip PML visualizations, sources, monitors
            if hasattr(structure, 'is_pml') and structure.is_pml:
                non_polygon_structures.append(structure)
                continue
            # Import at function level to avoid circular imports
            from beamz.design.sources import ModeSource, GaussianSource
            from beamz.design.monitors import Monitor
            if isinstance(structure, (ModeSource, GaussianSource, Monitor)):
                non_polygon_structures.append(structure)
                continue
            # Only process polygon-like structures with vertices
            if not hasattr(structure, 'vertices') or not hasattr(structure, 'material'):
                non_polygon_structures.append(structure)
                continue
                
            # Create a material key based on material properties
            material = structure.material
            if not material:
                non_polygon_structures.append(structure)
                continue
            material_key = (
                getattr(material, 'permittivity', None),
                getattr(material, 'permeability', None),
                getattr(material, 'conductivity', None)
            )
            
            # Add to the appropriate group
            if material_key not in material_groups: material_groups[material_key] = []
            # Convert to Shapely polygon
            try:
                # Handle polygons that might have interiors defined
                if hasattr(structure, 'interiors') and structure.interiors:
                    # Ensure vertices are not empty, and interiors are lists of coordinates
                    valid_interiors = [list(i_path) for i_path in structure.interiors if i_path]
                    if structure.vertices and valid_interiors:
                         shapely_polygon = ShapelyPolygon(shell=structure.vertices, holes=valid_interiors)
                    elif structure.vertices: # Only exterior is valid
                         shapely_polygon = ShapelyPolygon(shell=structure.vertices)
                    else: # No valid exterior
                        display_status(f"Skipping structure with no valid exterior vertices: {structure}", "warning")
                        non_polygon_structures.append(structure)
                        continue
                elif hasattr(structure, 'vertices') and structure.vertices: # Simple polygon with only exterior vertices
                    shapely_polygon = ShapelyPolygon(shell=structure.vertices)
                else: # Not a valid polygon structure
                    non_polygon_structures.append(structure)
                    continue # Skip if no vertices

                if shapely_polygon.is_valid:
                    material_groups[material_key].append((structure, shapely_polygon))
                    structures_to_remove.append(structure)
                else:
                    display_status(f"Skipping invalid polygon: {structure}", "warning")
                    non_polygon_structures.append(structure)
            except Exception as e:
                display_status(f"Error converting structure to Shapely polygon: {e}", "warning")
                non_polygon_structures.append(structure)
        
        # Second pass: Check for Ring objects that don't touch other objects
        rings_to_preserve = []
        for material_key, structure_group in material_groups.items():
            if len(structure_group) <= 1:
                continue  # Skip material groups with only one structure
            rings_in_group = [(idx, s) for idx, s in enumerate(structure_group) if isinstance(s[0], Ring)]
            if not rings_in_group:
                continue  # No rings in this group
            # For each Ring, preserve it unconditionally to maintain hole structure
            for ring_idx, (ring, ring_shapely) in rings_in_group:
                rings_to_preserve.append(ring)
                # Remove it from the material group to prevent it from being unified
                if ring in structures_to_remove:
                    structures_to_remove.remove(ring)
        
        # Third pass: unify polygons within each material group
        new_structures = []
        for material_key, structure_group in material_groups.items():
            # Exclude preserved rings from union to avoid duplicate drawing (simplified + original)
            filtered_group = [s for s in structure_group if s[0] not in rings_to_preserve]
            if len(filtered_group) <= 1:
                # Zero or one non-ring structure left to merge; keep originals (non-rings)
                new_structures.extend([s[0] for s in filtered_group])
                for s in filtered_group:
                    if s[0] in structures_to_remove:
                        structures_to_remove.remove(s[0])
                # Rings are already preserved and excluded from removal above
                continue
                
            # Extract shapely polygons for merging
            shapely_polygons = [p[1] for p in filtered_group]
            # Get the material from the first structure in the group
            material = filtered_group[0][0].material
            try:
                # Unify the polygons
                merged = unary_union(shapely_polygons)
                # The result could be a single polygon or a multipolygon
                if merged.geom_type == 'Polygon':
                    # Simplify the unified polygon to reduce complexity for 3D triangulation
                    simplified = self._simplify_polygon_for_3d(merged)
                    # Don't slice off the last vertex - our add_to_plot method needs complete vertices
                    exterior_coords = list(simplified.exterior.coords[:-1])  # Keep [:-1] to remove duplicate closing vertex from Shapely
                    interior_coords_lists = [list(interior.coords[:-1]) for interior in simplified.interiors]
                    if exterior_coords and len(exterior_coords) >= 3:
                        # Add depth and z information from the first structure
                        first_structure = filtered_group[0][0]
                        depth = getattr(first_structure, 'depth', 0)
                        z = getattr(first_structure, 'z', 0)
                        
                        new_poly = Polygon(vertices=exterior_coords, interiors=interior_coords_lists, 
                                         material=material, depth=depth, z=z)
                        new_structures.append(new_poly)
                        display_status(f"Unified {len(structure_group)} polygons with permittivity={material_key[0]} \
                                        (simplified to {len(exterior_coords)} vertices)", "success")
                    else:
                        display_status(f"Failed to convert merged polygon for material {material_key[0]} \
                            (no exterior or too few vertices), keeping original {len(structure_group)} structures.", "warning")
                        new_structures.extend([s[0] for s in structure_group])
                        for s_tuple in structure_group: # Ensure these are not removed
                            if s_tuple[0] in structures_to_remove:
                                structures_to_remove.remove(s_tuple[0])

                elif merged.geom_type == 'MultiPolygon':
                    all_geoms_converted_successfully = True
                    temp_new_polys_for_multipolygon = []
                    for geom in merged.geoms:
                        # Simplify each polygon in the multipolygon
                        simplified_geom = self._simplify_polygon_for_3d(geom)
                        # Keep [:-1] to remove duplicate closing vertex from Shapely (our add_to_plot will add it back)
                        exterior_coords = list(simplified_geom.exterior.coords[:-1])
                        interior_coords_lists = [list(interior.coords[:-1]) for interior in simplified_geom.interiors]
                        if exterior_coords and len(exterior_coords) >= 3:
                            # Add depth and z information from the first structure
                            first_structure = filtered_group[0][0]
                            depth = getattr(first_structure, 'depth', 0)
                            z = getattr(first_structure, 'z', 0)
                            new_poly = Polygon(vertices=exterior_coords, interiors=interior_coords_lists, 
                                             material=material, depth=depth, z=z)
                            temp_new_polys_for_multipolygon.append(new_poly)
                        else: # A sub-geometry had no exterior or too few vertices
                            all_geoms_converted_successfully = False
                            display_status(f"Failed to convert a geometry (no exterior or too few vertices) \
                                        within MultiPolygon for material {material_key[0]}.", "warning")
                            break 
                    
                    if all_geoms_converted_successfully:
                        new_structures.extend(temp_new_polys_for_multipolygon)
                        total_vertices = sum(len(p.vertices) for p in temp_new_polys_for_multipolygon)
                        display_status(f"Unified into {len(merged.geoms)} separate polygons with \
                                permittivity={material_key[0]} (simplified to {total_vertices} total vertices)", "success")
                    else:
                        display_status(f"Reverting unification for material {material_key[0]} due to conversion error \
                            in MultiPolygon, keeping original {len(structure_group)} structures.", "warning")
                        new_structures.extend([s[0] for s in structure_group])
                        for s_tuple in structure_group: # Ensure these are not removed
                            if s_tuple[0] in structures_to_remove:
                                structures_to_remove.remove(s_tuple[0])
                else:
                    # If the result is something unexpected, keep the original structures
                    display_status(f"Unexpected geometry type after union: {merged.geom_type} for material {material_key[0]}, \
                        keeping original structures", "warning")
                    new_structures.extend([s[0] for s in structure_group])
                    for s in structure_group:
                        if s[0] in structures_to_remove:
                            structures_to_remove.remove(s[0])
            except Exception as e:
                display_status(f"Error unifying polygons: {e}", "error")
                # Keep original structures if unification fails
                new_structures.extend([s[0] for s in structure_group])
                for s in structure_group:
                    if s[0] in structures_to_remove:
                        structures_to_remove.remove(s[0])
        
        # Rebuild structures list preserving original z-order
        # Create a mapping from material groups to their unified replacements
        material_replacements = {}
        
        # Match unified structures to their material groups by material properties
        for new_struct in new_structures:
            if hasattr(new_struct, 'material') and new_struct.material:
                new_material_key = (
                    getattr(new_struct.material, 'permittivity', None),
                    getattr(new_struct.material, 'permeability', None),
                    getattr(new_struct.material, 'conductivity', None)
                )
                # Find the material group that this unified structure belongs to
                for material_key, structure_group in material_groups.items():
                    if len(structure_group) > 1 and material_key == new_material_key:
                        if material_key not in material_replacements:
                            material_replacements[material_key] = []
                        material_replacements[material_key].append(new_struct)
                        break
        
        # Rebuild the structures list in original order
        rebuilt_structures = []
        material_groups_used = set()
        
        for structure in self.structures:
            if structure in structures_to_remove:
                # Find which material group this structure belongs to
                structure_material_key = None
                if hasattr(structure, 'material') and structure.material:
                    structure_material_key = (
                        getattr(structure.material, 'permittivity', None),
                        getattr(structure.material, 'permeability', None),
                        getattr(structure.material, 'conductivity', None)
                    )
                
                # Add the unified replacement(s) for this material group (only once)
                if structure_material_key and structure_material_key not in material_groups_used:
                    if structure_material_key in material_replacements:
                        rebuilt_structures.extend(material_replacements[structure_material_key])
                        material_groups_used.add(structure_material_key)
            else:
                # Keep non-unified structures (includes preserved rings and non-polygon structures)
                rebuilt_structures.append(structure)
        
        # Replace the structures list
        self.structures = rebuilt_structures
        
        # Final report
        display_status(f"Polygon unification complete: {len(structures_to_remove)} structures merged \
            into {len(new_structures)} unified shapes, {len(rings_to_preserve)} isolated rings preserved", "success")
        return True
    
    def _simplify_polygon_for_3d(self, shapely_polygon, tolerance_factor=0.01):
        """Simplify a Shapely polygon to reduce vertex count for better 3D triangulation."""
        try:
            # Calculate appropriate tolerance based on polygon size
            bounds = shapely_polygon.bounds
            size = max(bounds[2] - bounds[0], bounds[3] - bounds[1])  # max of width, height
            tolerance = size * tolerance_factor
            # Apply simplification
            simplified = shapely_polygon.simplify(tolerance, preserve_topology=True)
            # Check if simplification was successful and polygon is still valid
            if simplified.is_valid and not simplified.is_empty:
                # Check if we still have a reasonable number of vertices
                if simplified.geom_type == 'Polygon':
                    exterior_coords = len(list(simplified.exterior.coords))
                    if 3 <= exterior_coords <= 100:  # Reasonable range for 3D visualization
                        return simplified
                    elif exterior_coords > 100:
                        # Try more aggressive simplification
                        more_simplified = shapely_polygon.simplify(tolerance * 2.0, preserve_topology=True)
                        if more_simplified.is_valid and not more_simplified.is_empty:
                            return more_simplified
            
            # If simplification failed or created invalid geometry, return original
            return shapely_polygon
            
        except Exception as e:
            print(f"Warning: Polygon simplification failed ({e}), using original")
            return shapely_polygon

    def scatter(self, structure, n=1000, xyrange=(-5*µm, 5*µm), scale_range=(0.05, 1)):
        """Randomly distribute a given object over the design domain."""
        display_status(f"Scattering {n} instances of {structure.__class__.__name__}", "info")
        for _ in range(n):
            new_structure = structure.copy()
            new_structure.shift(random.uniform(xyrange[0], xyrange[1]), random.uniform(xyrange[0], xyrange[1]))
            new_structure.rotate(random.uniform(0, 360))
            new_structure.scale(random.uniform(scale_range[0], scale_range[1]))
            self.add(new_structure)
        display_status(f"Completed scattering {n} structures", "success")

    def init_boundaries(self, pml_size=None):
        """Add boundary conditions to the design area (using PML)."""
        # Calculate PML size more intelligently if not specified
        if pml_size is None:
            # Find max permittivity in design for wavelength calculation
            max_permittivity = 1.0
            for structure in self.structures:
                if hasattr(structure, 'material') and hasattr(structure.material, 'permittivity'):
                    max_permittivity = max(max_permittivity, structure.material.permittivity)
            # Estimate minimum wavelength
            wavelength_estimate = 1.55e-6 / np.sqrt(max_permittivity)
            # Make PML thicker to allow for more gradual absorption
            min_size = 1.5 * wavelength_estimate  # Increased from 1.0
            max_size = min(self.width, self.height) * 0.3  # Increased thickness for gradual absorption
            pml_size = max(min_size, min(max_size, min(self.width, self.height) / 3))
            display_status(f"Auto-selected PML size: {pml_size:.2e} m (~{pml_size/wavelength_estimate:.1f} wavelengths)", "info")
        
        # Create transparent material for PML outlines (for visualization only)
        pml_material = Material(permittivity=1.0, permeability=1.0, conductivity=0.0)
        
        # Create unified PML regions with optimized parameters for gradual absorption
        # Rectangular edge PMLs
        self.boundaries.append(PML("rect", (0, 0), (pml_size, self.height), "left"))
        self.boundaries.append(PML("rect", (self.width - pml_size, 0), (pml_size, self.height), "right"))
        self.boundaries.append(PML("rect", (0, self.height - pml_size), (self.width, pml_size), "top"))
        self.boundaries.append(PML("rect", (0, 0), (self.width, pml_size), "bottom"))
        # Corner PMLs
        self.boundaries.append(PML("corner", (0, 0), pml_size, "bottom-left"))
        self.boundaries.append(PML("corner", (self.width, 0), pml_size, "bottom-right"))
        self.boundaries.append(PML("corner", (0, self.height), pml_size, "top-left"))
        self.boundaries.append(PML("corner", (self.width, self.height), pml_size, "top-right"))
        
        # Add visual representations of PML regions to the structures list (for display only)
        # These are just visualization helpers and don't affect the actual simulation
        left_pml = Rectangle(position=(0, 0), width=pml_size,
                                height=self.height, material=pml_material, color='none', is_pml=True)
        self.structures.append(left_pml)
        # Right PML region
        right_pml = Rectangle(position=(self.width - pml_size, 0), width=pml_size, height=self.height,
                                material=pml_material, color='none', is_pml=True)
        self.structures.append(right_pml)
        # Bottom PML region
        bottom_pml = Rectangle(position=(0, 0), width=self.width, height=pml_size,
                                material=pml_material, color='none', is_pml=True)
        self.structures.append(bottom_pml)
        # Top PML region
        top_pml = Rectangle(position=(0, self.height - pml_size),width=self.width, height=pml_size, material=pml_material,
            color='none', is_pml=True)
        self.structures.append(top_pml)

    def show(self, unify_structures=True):
        """Display the design visually using 2D matplotlib or 3D plotly (delegated to beamz.viz)."""
        return viz.show_design(self, unify_structures)
    
    def _determine_if_3d(self):
        """Determine if the design should be visualized in 3D (delegated to beamz.viz)."""
        return viz.determine_if_3d(self)
    
    def show_2d(self, unify_structures=True):
        """Display the design using 2D matplotlib (delegated to beamz.viz)."""
        return viz.show_design_2d(self, unify_structures)
    
    def show_3d(self, unify_structures=True, max_vertices_for_unification=50):
        """Display the design using 3D plotly (delegated to beamz.viz)."""
        return viz.show_design_3d(self, unify_structures, max_vertices_for_unification)

    def __str__(self):
        return f"Design with {len(self.structures)} structures ({'3D' if self.is_3d else '2D'})"

    def get_material_value(self, x, y, z=0, dx=None, dt=None):
        """Return the material value at a given (x, y, z) coordinate, prioritizing the topmost structure."""
        # First get material values from underlying structures
        # Start with default background material 
        epsilon = 1.0
        mu = 1.0
        sigma_base = 0.0
        # Find the material values from the structures (outside PML calculation)
        for structure in reversed(self.structures):
            if isinstance(structure, Polygon):
                if structure.point_in_polygon(x, y, z):
                    epsilon, mu, sigma_base = structure.material.get_sample()
                    break
            elif isinstance(structure, Rectangle):
                if structure.is_pml:
                    # Skip visual PML structures - they're just for display
                    continue
                if structure.point_in_polygon(x, y, z):
                    epsilon, mu, sigma_base = structure.material.get_sample()
                    break
            elif isinstance(structure, Circle):
                if np.hypot(x - structure.position[0], y - structure.position[1]) <= structure.radius:
                    epsilon, mu, sigma_base = structure.material.get_sample()
                    break
            elif isinstance(structure, Ring):
                distance = np.hypot(x - structure.position[0], y - structure.position[1])
                if structure.inner_radius <= distance <= structure.outer_radius:
                    epsilon, mu, sigma_base = structure.material.get_sample()
                    break
            elif isinstance(structure, CircularBend):
                distance = np.hypot(x - structure.position[0], y - structure.position[1])
                if structure.inner_radius <= distance <= structure.outer_radius:
                    epsilon, mu, sigma_base = structure.material.get_sample()
                    break
        # Calculate PML conductivity based on the UNDERLYING material
        # This is crucial for proper absorption without reflection
        pml_conductivity = 0.0
        if dx is not None:
            eps_avg = epsilon  # Use the actual permittivity at this point
            # Apply all PML boundaries
            for boundary in self.boundaries:
                pml_conductivity += boundary.get_conductivity(x, y, dx=dx, dt=dt, eps_avg=eps_avg)
        
        # Return with the permittivity of the underlying structure plus PML conductivity
        return [epsilon, mu, sigma_base + pml_conductivity]

    def _point_in_polygon(self, x, y, vertices):
        """Check if a point is inside a polygon using the ray-casting algorithm.
        This method is kept for backwards compatibility but now uses the Polygon.point_in_polygon method."""
        # Create a temporary polygon to use the new point_in_polygon method
        temp_polygon = Polygon(vertices=vertices)
        return temp_polygon.point_in_polygon(x, y)

    def copy(self):
        """Create a deep copy of the design."""
        # Get background material from the first structure (background rectangle)
        background_material = None
        if self.structures and hasattr(self.structures[0], 'material'):
            background_material = self.structures[0].material
        
        # Create new design without auto_pml since we'll copy boundaries manually
        new_design = Design(width=self.width, height=self.height, material=background_material, 
                           auto_pml=False)
        
        # Clear the automatically created structures and rebuild with deep copies
        new_design.structures = []
        new_design.sources = []
        new_design.monitors = []
        new_design.boundaries = []
        
        # Deep copy all structures in the correct order
        for structure in self.structures:
            if hasattr(structure, 'copy'):
                copied_structure = structure.copy()
                # Deep copy materials if they have a copy method (like CustomMaterial)
                if hasattr(copied_structure, 'material') and copied_structure.material is not None:
                    if hasattr(copied_structure.material, 'copy'): copied_structure.material = copied_structure.material.copy()
                # Update design reference for sources and monitors
                if hasattr(copied_structure, 'design'): copied_structure.design = new_design
                new_design.structures.append(copied_structure)
                # Also add to appropriate lists
                if isinstance(copied_structure, (ModeSource, GaussianSource)):
                    new_design.sources.append(copied_structure)
                elif hasattr(structure, '__class__') and 'Monitor' in structure.__class__.__name__:
                    new_design.monitors.append(copied_structure)
            else:
                # Fallback for structures without copy method
                new_design.structures.append(structure)
        
        # Deep copy boundaries
        for boundary in self.boundaries:
            if hasattr(boundary, 'copy'): new_design.boundaries.append(boundary.copy())
            else: new_design.boundaries.append(boundary)
        
        # Copy other attributes
        new_design.is_3d = self.is_3d
        new_design.depth = self.depth
        new_design.border_color = self.border_color
        new_design.time = self.time
        new_design.layers = self.layers.copy() if hasattr(self, 'layers') else {}
        
        return new_design





class Polygon:
    def __init__(self, vertices=None, material=None, color=None, optimize=False, interiors=None, depth=0, z=0):
        # First ensure vertices are properly formatted and ordered
        self.vertices = self._process_vertices(vertices if vertices is not None else [], z)
        # Preserve interior path orientation (needed for proper hole rendering with nonzero rule)
        self.interiors = [self._process_vertices_preserve_orientation(interior, z) for interior in (interiors if interiors is not None else [])]
        self.material = material
        self.optimize = optimize
        self.color = color if color is not None else self.get_random_color_consistent()
        self.depth = depth if depth is not None else 0
        self.z = z if z is not None else 0
    
    def _process_vertices(self, vertices, z=0):
        """Process vertices to ensure they are 3D and properly ordered counterclockwise."""
        if not vertices: return []   
        # First ensure all vertices are 3D
        vertices_3d = self._ensure_3d_vertices(vertices)
        # Project to 2D for CCW check
        vertices_2d = [(v[0], v[1]) for v in vertices_3d]
        # Ensure counterclockwise ordering
        if len(vertices_2d) >= 3:
            vertices_2d = self._ensure_ccw_vertices(vertices_2d)
            # Convert back to 3D maintaining original z-coordinates or using provided z
            vertices_3d = [(x, y, vertices_3d[i][2] if len(vertices_3d[i]) > 2 else z) 
                          for i, (x, y) in enumerate(vertices_2d)]
        return vertices_3d

    def _process_vertices_preserve_orientation(self, vertices, z=0):
        """Ensure 3D coordinates but preserve original vertex order (for holes)."""
        if not vertices: return []
        vertices_3d = self._ensure_3d_vertices(vertices)
        # Do not alter orientation; only ensure z coordinate presence
        return [(v[0], v[1], v[2] if len(v) > 2 else z) for v in vertices_3d]
    
    def _ensure_ccw_vertices(self, vertices_2d):
        """Ensure vertices are ordered counterclockwise by computing signed area."""
        if len(vertices_2d) < 3: return vertices_2d
        # Calculate signed area
        area = 0
        for i in range(len(vertices_2d)):
            j = (i + 1) % len(vertices_2d)
            area += vertices_2d[i][0] * vertices_2d[j][1]
            area -= vertices_2d[j][0] * vertices_2d[i][1]
        # If area is positive, vertices are already CCW
        # If area is negative, reverse the vertices
        if area < 0: return vertices_2d[::-1]
        return vertices_2d
    
    def _ensure_3d_vertices(self, vertices):
        """Convert 2D vertices (x,y) to 3D vertices (x,y,z) with z=0 if not provided."""
        if not vertices: return []
        result = []
        for v in vertices:
            if len(v) == 2: result.append((v[0], v[1], 0.0))  # Add z=0 for 2D vertices
            elif len(v) == 3: result.append(v)  # Keep 3D vertices as-is
            else: raise ValueError(f"Vertex must have 2 or 3 coordinates, got {len(v)}")
        return result
    
    def _vertices_2d(self, vertices=None):
        """Get 2D projection of vertices for plotting (x,y only)."""
        if vertices is None: vertices = self.vertices
        return [(v[0], v[1]) for v in vertices]
    
    def get_random_color_consistent(self, saturation=0.6, value=0.7):
        """Generate a random color with consistent perceived brightness and saturation."""
        hue = random.random() # Generate random hue (0-1)
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
    
    def shift(self, x, y, z=0):
        """Shift the polygon by (x,y,z) and return self for method chaining. z is optional for 2D compatibility."""
        if self.vertices: self.vertices = [(v[0] + x, v[1] + y, v[2] + z) for v in self.vertices]
        new_interiors_paths = []
        for interior_path in self.interiors:
            if interior_path: new_interiors_paths.append([(v[0] + x, v[1] + y, v[2] + z) for v in interior_path])
        self.interiors = new_interiors_paths
        return self
    
    def scale(self, s_x, s_y=None, s_z=None):
        """Scale the polygon around its center of mass and return self for method chaining.
        If s_y and s_z are None, uniform scaling is applied to all dimensions."""
        if s_y is None: s_y = s_x  # Uniform scaling in xy if only one parameter given
        if s_z is None: s_z = 1.0 if s_y != s_x else s_x  # Keep z unchanged for 2D, or uniform for 3D
        
        if self.vertices:
            # Calculate center of mass of the exterior
            x_center = sum(v[0] for v in self.vertices) / len(self.vertices)
            y_center = sum(v[1] for v in self.vertices) / len(self.vertices)
            z_center = sum(v[2] for v in self.vertices) / len(self.vertices)
            # Scale exterior
            self.vertices = [(x_center + (v[0] - x_center) * s_x,
                              y_center + (v[1] - y_center) * s_y,
                              z_center + (v[2] - z_center) * s_z)
                              for v in self.vertices]
            # Scale interiors
            new_interiors_paths = []
            for interior_path in self.interiors:
                if interior_path:
                    new_interiors_paths.append([(x_center + (v[0] - x_center) * s_x,
                                                 y_center + (v[1] - y_center) * s_y,
                                                 z_center + (v[2] - z_center) * s_z)
                                                 for v in interior_path])
            self.interiors = new_interiors_paths
        return self
    
    def rotate(self, angle, axis='z', point=None):
        """Rotate the polygon around its center of mass or specified point.
        angle: Rotation angle in degrees
        axis: Rotation axis ('x', 'y', or 'z'). Default 'z' for 2D compatibility
        point: Optional (x,y,z) point to rotate around. If None, rotates around center of exterior.
        """
        if self.vertices:
            angle_rad = np.radians(angle)
            if point is None:
                # Calculate center of mass of the exterior
                x_center = sum(v[0] for v in self.vertices) / len(self.vertices)
                y_center = sum(v[1] for v in self.vertices) / len(self.vertices)
                z_center = sum(v[2] for v in self.vertices) / len(self.vertices)
            else: x_center, y_center, z_center = (point[0], point[1], point[2] if len(point) > 2 else 0)

            # Define rotation matrices for each axis
            if axis == 'z':  # 2D rotation in xy plane (most common)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                self.vertices = [
                    (x_center + (v[0] - x_center) * cos_a - (v[1] - y_center) * sin_a,
                     y_center + (v[0] - x_center) * sin_a + (v[1] - y_center) * cos_a,
                     v[2])
                    for v in self.vertices
                ]
            elif axis == 'x':  # Rotation around x-axis (yz plane)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                self.vertices = [
                    (v[0],
                     y_center + (v[1] - y_center) * cos_a - (v[2] - z_center) * sin_a,
                     z_center + (v[1] - y_center) * sin_a + (v[2] - z_center) * cos_a)
                    for v in self.vertices
                ]
            elif axis == 'y':  # Rotation around y-axis (xz plane)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                self.vertices = [
                    (x_center + (v[0] - x_center) * cos_a + (v[2] - z_center) * sin_a,
                     v[1],
                     z_center - (v[0] - x_center) * sin_a + (v[2] - z_center) * cos_a)
                    for v in self.vertices
                ]
            else:
                raise ValueError(f"Invalid rotation axis '{axis}'. Must be 'x', 'y', or 'z'.")
            
            # Rotate interiors
            new_interiors_paths = []
            for interior_path in self.interiors:
                if interior_path:
                    if axis == 'z':
                        new_interiors_paths.append([
                            (x_center + (v[0] - x_center) * cos_a - (v[1] - y_center) * sin_a,
                             y_center + (v[0] - x_center) * sin_a + (v[1] - y_center) * cos_a,
                             v[2])
                            for v in interior_path
                        ])
                    elif axis == 'x':
                        new_interiors_paths.append([
                            (v[0],
                             y_center + (v[1] - y_center) * cos_a - (v[2] - z_center) * sin_a,
                             z_center + (v[1] - y_center) * sin_a + (v[2] - z_center) * cos_a)
                            for v in interior_path
                        ])
                    elif axis == 'y':
                        new_interiors_paths.append([
                            (x_center + (v[0] - x_center) * cos_a + (v[2] - z_center) * sin_a,
                             v[1],
                             z_center - (v[0] - x_center) * sin_a + (v[2] - z_center) * cos_a)
                            for v in interior_path
                        ])
            self.interiors = new_interiors_paths
        return self

    def add_to_plot(self, ax, facecolor=None, edgecolor="black", alpha=None, linestyle=None):
        """Delegate polygon drawing to beamz.viz.draw_polygon."""
        return viz.draw_polygon(ax, self, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linestyle=linestyle)

    def copy(self):
        # Ensure interiors are copied as lists of tuples/lists, not as references
        copied_interiors = [list(path) for path in self.interiors if path] if self.interiors else []
        return Polygon(vertices=list(self.vertices) if self.vertices else [], 
                       interiors=copied_interiors, 
                       material=self.material, # Material can be shared
                       color=self.color, 
                       optimize=self.optimize,
                       depth=self.depth,
                       z=self.z)
        
    def get_bounding_box(self):
        """Get the bounding box of the polygon as (min_x, min_y, min_z, max_x, max_y, max_z)"""
        if not self.vertices or len(self.vertices) == 0:
            return (0, 0, 0, 0, 0, 0)
        # Extract x, y, and z coordinates
        x_coords = [v[0] for v in self.vertices]
        y_coords = [v[1] for v in self.vertices]
        z_coords = [v[2] for v in self.vertices]
        # Calculate min and max
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        min_z, max_z = min(z_coords), max(z_coords)
        return (min_x, min_y, min_z, max_x, max_y, max_z)
        
    def _point_in_polygon_single_path(self, x, y, path_vertices):
        """Check if a point is inside a single, simple polygon path using ray-casting.
        Uses 2D projection (xy plane) for 3D vertices."""
        if not path_vertices: return False
        # Project to 2D for point-in-polygon test
        path_2d = self._vertices_2d(path_vertices)
        n = len(path_2d)
        inside = False
        p1x, p1y = path_2d[0]
        for i in range(n + 1):
            p2x, p2y = path_2d[i % n] # Ensure closure for ray casting
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y: # Avoid division by zero for horizontal lines
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        else: # Edge is horizontal
                            xinters = p1x # Doesn't matter, will compare x <= max(p1x, p2x)
                        if p1x == p2x or x <= xinters: # For vertical edge or point left of intersection
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def point_in_polygon(self, x, y, z=None):
        """Check if a point (x,y,z) is inside this polygon (which may have holes).
        For 3D, currently uses 2D projection in xy plane. z parameter is for future 3D containment."""
        # Use self.vertices for exterior and self.interiors for holes
        exterior_path = self.vertices
        interior_paths = self.interiors
        if not exterior_path: return False
        # Check if point is in the exterior boundary (2D projection)
        if not self._point_in_polygon_single_path(x, y, exterior_path):
            return False # Not in exterior, so definitely not in polygon
        # If in exterior, check if it's in any of the holes
        for interior_path_pts in interior_paths:
            if interior_path_pts and self._point_in_polygon_single_path(x, y, interior_path_pts):
                return False # Point is in a hole, so not in polygon
        return True # In exterior and not in any hole

class Rectangle(Polygon):
    def __init__(self, position=(0,0,0), width=1, height=1, depth=1, material=None, color=None, is_pml=False, optimize=False, z=None):
        # Handle z parameter for backward compatibility
        if z is not None:
            if len(position) == 2:
                position = (position[0], position[1], z)
            elif len(position) == 3:
                position = (position[0], position[1], z)  # Override z from position
        # Handle 2D position input (x,y) by adding z=0
        if len(position) == 2: position = (position[0], position[1], 0.0)
        elif len(position) == 3: position = position
        else: raise ValueError("Position must be (x,y) or (x,y,z)")
        # Calculate vertices for the rectangle in 3D
        x, y, z_pos = position
        vertices = [(x, y, z_pos),  # Bottom left
                    (x + width, y, z_pos),  # Bottom right
                    (x + width, y + height, z_pos),  # Top right
                    (x, y + height, z_pos)]
        super().__init__(vertices=vertices, material=material, color=color, optimize=optimize, depth=depth, z=z_pos)
        self.position = position
        self.width = width
        self.height = height
        self.depth = depth
        self.is_pml = is_pml
        
    def get_bounding_box(self):
        """Get the axis-aligned bounding box for this rectangle."""
        # For non-rotated rectangles, this is straightforward
        if not hasattr(self, 'vertices') or len(self.vertices) == 0:
            x, y, z = self.position
            return (x, y, z, x + self.width, y + self.height, z + self.depth)
        # For potentially rotated rectangles, use the vertices
        return super().get_bounding_box()

    def shift(self, x, y, z=0):
        """Shift the rectangle by (x,y,z) and return self for method chaining."""
        self.position = (self.position[0] + x, self.position[1] + y, self.position[2] + z)
        super().shift(x, y, z)
        return self

    def rotate(self, angle, axis='z', point=None):
        """Rotate the rectangle around its center of mass or specified point."""        
        # Use parent class rotation method
        super().rotate(angle, axis, point)
        # Calculate new bounding box after rotation
        min_x = min(v[0] for v in self.vertices)
        min_y = min(v[1] for v in self.vertices)
        min_z = min(v[2] for v in self.vertices)
        max_x = max(v[0] for v in self.vertices)
        max_y = max(v[1] for v in self.vertices)
        max_z = max(v[2] for v in self.vertices)
        # Update position to be the bottom-left corner
        self.position = (min_x, min_y, min_z)
        self.width = max_x - min_x
        self.height = max_y - min_y
        self.depth = max_z - min_z
        return self

    def scale(self, s_x, s_y=None, s_z=None):
        """Scale the rectangle around its center of mass and return self for method chaining."""
        if s_y is None: s_y = s_x  # Uniform scaling in xy if only one parameter given
        if s_z is None: s_z = 1.0 if s_y != s_x else s_x  # Keep z unchanged for 2D, or uniform for 3D
        super().scale(s_x, s_y, s_z)
        self.width *= s_x
        self.height *= s_y
        self.depth *= s_z
        return self
    
    def copy(self):
        """Create a copy of this rectangle with the same attributes and vertices."""
        new_rect = Rectangle(self.position, self.width, self.height, self.depth, 
                            self.material, self.color, self.is_pml, self.optimize)
        # Ensure vertices are copied exactly as they are (important for rotated rectangles)
        new_rect.vertices = [(x, y, z) for x, y, z in self.vertices]
        return new_rect

class Circle(Polygon):
    def __init__(self, position=(0,0), radius=1, points=32, material=None, color=None, optimize=False, depth=0, z=0):
        # Handle 2D position input (x,y) by adding z=0
        if len(position) == 2: position = (position[0], position[1], 0.0)
        elif len(position) == 3: position = position
        else: raise ValueError("Position must be (x,y) or (x,y,z)")
        theta = np.linspace(0, 2*np.pi, points, endpoint=False)
        vertices = [(position[0] + radius * np.cos(t), position[1] + radius * np.sin(t), position[2]) for t in theta]
        super().__init__(vertices=vertices, material=material, color=color, optimize=optimize, depth=depth, z=z)
        self.position = position
        self.radius = radius
        self.points = points
    
    def shift(self, x, y, z=0):
        """Shift the circle by (x,y,z) and return self for method chaining."""
        self.position = (self.position[0] + x, self.position[1] + y, self.position[2] + z)
        super().shift(x, y, z)
        return self
    
    def scale(self, s_x, s_y=None, s_z=None):
        """Scale the circle radius and return self for method chaining.
        For circles, s_x is used as the radius scaling factor. s_y and s_z affect shape if different."""
        if s_y is None: s_y = s_x  # Uniform scaling in xy if only one parameter given
        if s_z is None: s_z = 1.0   # Don't scale z for circles by default
        self.radius *= s_x
        # Regenerate vertices with new radius
        theta = np.linspace(0, 2*np.pi, self.points, endpoint=False)
        self.vertices = [(self.position[0] + self.radius * np.cos(t), 
                         self.position[1] + self.radius * np.sin(t),
                         self.position[2]) for t in theta]
        return self
    
    def copy(self):
        return Circle(position=self.position, radius=self.radius, points=self.points, 
                     material=self.material, color=self.color, optimize=self.optimize, 
                     depth=self.depth, z=self.z)

class Ring(Polygon):
    def __init__(self, position=(0,0), inner_radius=1, outer_radius=2, material=None, color=None, optimize=False, points=256, depth=0, z=None):
        # Handle z parameter for backward compatibility
        if z is not None:
            if len(position) == 2: position = (position[0], position[1], z)
            elif len(position) == 3: position = (position[0], position[1], z)  # Override z from position
        
        # Handle 2D position input (x,y) by adding z=0
        if len(position) == 2: position = (position[0], position[1], 0.0)
        elif len(position) == 3: position = position
        else: raise ValueError("Position must be (x,y) or (x,y,z)")
        theta = np.linspace(0, 2*np.pi, points, endpoint=False) # CCW, N points
        # Exterior path (CCW, N points)
        # These are unclosed paths, as expected by Polygon.add_to_plot's Path logic
        outer_ext_vertices = [(position[0] + outer_radius * np.cos(t), 
                               position[1] + outer_radius * np.sin(t),
                               position[2]) for t in theta]
        # Interior path (should be CW for Matplotlib Path hole convention if exterior is CCW)
        # Generate points CW by reversing theta or using reversed(theta)
        inner_int_vertices_cw = [(position[0] + inner_radius * np.cos(t), 
                                  position[1] + inner_radius * np.sin(t),
                                  position[2]) for t in reversed(theta)]
        super().__init__(vertices=outer_ext_vertices, 
                         interiors=[inner_int_vertices_cw] if inner_int_vertices_cw else [], 
                         material=material, color=color, optimize=optimize, depth=depth, z=position[2])
        self.points = points # Store for potential regeneration or other logic if needed
        self.position = position
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
    
    def shift(self, x, y, z=0):
        """Shift the ring by (x,y,z) and return self for method chaining."""
        self.position = (self.position[0] + x, self.position[1] + y, self.position[2] + z)
        super().shift(x, y, z)
        return self
    
    def scale(self, s_x, s_y=None, s_z=None):
        """Scale the ring radii and return self for method chaining."""
        if s_y is None: s_y = s_x  # Uniform scaling in xy if only one parameter given
        if s_z is None: s_z = 1.0   # Don't scale z for rings by default
        self.inner_radius *= s_x; self.outer_radius *= s_x
        # Regenerate vertices with new radii
        theta = np.linspace(0, 2*np.pi, self.points, endpoint=False)
        # Outer circle points (exterior) and inner circle points (interior hole)
        outer_vertices = [(self.position[0] + self.outer_radius * np.cos(t), 
                          self.position[1] + self.outer_radius * np.sin(t),
                          self.position[2]) for t in theta]
        inner_vertices = [(self.position[0] + self.inner_radius * np.cos(t), 
                          self.position[1] + self.inner_radius * np.sin(t),
                          self.position[2]) for t in reversed(theta)]
        self.vertices = outer_vertices
        self.interiors = [inner_vertices]
        return self
    
    def add_to_plot(self, ax, facecolor=None, edgecolor="black", alpha=None, linestyle=None):
        if facecolor is None: facecolor = self.color
        if alpha is None: alpha = 1
        if linestyle is None: linestyle = '-'
        # Use the generic Polygon.add_to_plot which now handles holes via self.vertices and self.interiors
        # Ring.__init__ now populates self.vertices (exterior) and self.interiors correctly.
        super().add_to_plot(ax, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linestyle=linestyle)

    def copy(self):
        # Ring's __init__ now correctly sets up vertices and interiors for Polygon base class
        return Ring(position=self.position, 
                    inner_radius=self.inner_radius, 
                    outer_radius=self.outer_radius, 
                    material=self.material, # Material can be shared
                    color=self.color, 
                    optimize=self.optimize,
                    points=self.points,
                    depth=self.depth,
                    z=self.z)

class CircularBend(Polygon):
    def __init__(self, position=(0,0), inner_radius=1, outer_radius=2, angle=90, rotation=0, material=None, 
                 facecolor=None, optimize=False, points=64, depth=0, z=0):
        # Handle 2D position input (x,y) by adding z=0
        if len(position) == 2: position = (position[0], position[1], 0.0)
        elif len(position) == 3: position = position
        else: raise ValueError("Position must be (x,y) or (x,y,z)")
        self.points = points
        theta = np.linspace(0, np.radians(angle), points)
        rotation_rad = np.radians(rotation)
        outer_vertices = [(position[0] + outer_radius * np.cos(t + rotation_rad),
                          position[1] + outer_radius * np.sin(t + rotation_rad),
                          position[2]) for t in theta]
        inner_vertices = [(position[0] + inner_radius * np.cos(t + rotation_rad),
                          position[1] + inner_radius * np.sin(t + rotation_rad),
                          position[2]) for t in reversed(theta)]
        vertices = outer_vertices + inner_vertices
        super().__init__(vertices=vertices, material=material, color=facecolor, optimize=optimize, depth=depth, z=z)
        self.position = position
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.angle = angle
        self.rotation = rotation
    
    def shift(self, x, y, z=0):
        """Shift the bend by (x,y,z) and return self for method chaining."""
        self.position = (self.position[0] + x, self.position[1] + y, self.position[2] + z)
        super().shift(x, y, z)
        return self
    
    def rotate(self, angle, axis='z', point=None):
        """Rotate the bend around its center or specified point."""
        if axis == 'z':  # Only z-rotation makes sense for circular bends in xy plane
            self.rotation = (self.rotation + angle) % 360
        super().rotate(angle, axis, point or self.position)
        return self
    
    def scale(self, s_x, s_y=None, s_z=None):
        """Scale the bend radii and return self for method chaining."""
        if s_y is None: s_y = s_x  # Uniform scaling in xy if only one parameter given
        if s_z is None: s_z = 1.0   # Don't scale z for bends by default
        
        self.inner_radius *= s_x; self.outer_radius *= s_x
        theta = np.linspace(0, np.radians(self.angle), self.points)
        rotation_rad = np.radians(self.rotation)
        outer_vertices = [(self.position[0] + self.outer_radius * np.cos(t + rotation_rad),
                          self.position[1] + self.outer_radius * np.sin(t + rotation_rad),
                          self.position[2]) for t in theta]
        inner_vertices = [(self.position[0] + self.inner_radius * np.cos(t + rotation_rad),
                          self.position[1] + self.inner_radius * np.sin(t + rotation_rad),
                          self.position[2]) for t in reversed(theta)]
        self.vertices = outer_vertices + inner_vertices
        return self
    
    def add_to_plot(self, ax, facecolor=None, edgecolor="black", alpha=None, linestyle=None):
        if facecolor is None: facecolor = self.color
        if alpha is None: alpha = 1
        if linestyle is None: linestyle = '-'
        # Convert angles to radians
        angle_rad = np.radians(self.angle)
        rotation_rad = np.radians(self.rotation)
        theta = np.linspace(rotation_rad, rotation_rad + angle_rad, self.points, endpoint=True)
        # Outer and inner arc points (project 3D to 2D for plotting)
        x_outer = self.position[0] + self.outer_radius * np.cos(theta)
        y_outer = self.position[1] + self.outer_radius * np.sin(theta)
        x_inner = self.position[0] + self.inner_radius * np.cos(theta)
        y_inner = self.position[1] + self.inner_radius * np.sin(theta)
        # Create a closed path by combining points and adding connecting lines
        vertices = np.vstack([
            [x_outer[0], y_outer[0]],
            *np.column_stack([x_outer[1:], y_outer[1:]]),
            [x_inner[-1], y_inner[-1]],
            *np.column_stack([x_inner[-2::-1], y_inner[-2::-1]]),
            [x_outer[0], y_outer[0]]
        ])
        # Define path codes for a single continuous path
        codes = [Path.MOVETO] + \
                [Path.LINETO] * (len(vertices) - 2) + \
                [Path.CLOSEPOLY]
        # Create the path and patch
        path = Path(vertices, codes)
        bend_patch = PathPatch(path, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor, linestyle=linestyle)
        ax.add_patch(bend_patch)
        
    def copy(self):
        return CircularBend(self.position, self.inner_radius, self.outer_radius, 
                            self.angle, self.rotation, self.material, self.color, self.optimize, 
                            self.points, self.depth, self.z)

class Taper(Polygon):
    """Taper is a structure that tapers from a width to a height."""
    def __init__(self, position=(0,0), input_width=1, output_width=0.5, length=1, material=None, color=None, optimize=False, depth=0, z=0):
        # Handle 2D position input (x,y) by adding z=0
        if len(position) == 2: position = (position[0], position[1], 0.0)
        elif len(position) == 3: position = position
        else: raise ValueError("Position must be (x,y) or (x,y,z)")
        # Calculate vertices for the trapezoid shape in 3D
        x, y, z = position
        vertices = [(x, y - input_width/2, z),  # Bottom left
                    (x + length, y - output_width/2, z),  # Bottom right
                    (x + length, y + output_width/2, z),  # Top right
                    (x, y + input_width/2, z)] # Top left
        super().__init__(vertices=vertices, material=material, color=color, optimize=optimize, depth=depth, z=z)
        self.position = position
        self.input_width = input_width
        self.output_width = output_width
        self.length = length
        self.optimize = optimize

    def rotate(self, angle, axis='z', point=None):
        """Rotate the taper around its center of mass or specified point."""
        # Use parent class rotation method
        super().rotate(angle, axis, point)
        # Calculate new bounding box after rotation
        min_x = min(v[0] for v in self.vertices)
        min_y = min(v[1] for v in self.vertices)
        min_z = min(v[2] for v in self.vertices)
        max_x = max(v[0] for v in self.vertices)
        max_y = max(v[1] for v in self.vertices)
        max_z = max(v[2] for v in self.vertices)
        # Update position to left bottom corner and update length
        self.position = (min_x, min_y, min_z)
        self.length = max_x - min_x
        return self

    def copy(self):
        """Create a copy of this taper with the same attributes and vertices."""
        new_taper = Taper(self.position, self.input_width, self.output_width, 
                          self.length, self.material, self.color, self.optimize, self.depth, self.z)
        # Ensure vertices are copied exactly as they are (important for rotated tapers)
        new_taper.vertices = [(x, y, z) for x, y, z in self.vertices]
        return new_taper

class PML:
    """Unified PML (Perfectly Matched Layer) class for absorbing boundary conditions."""
    def __init__(self, region_type, position, size, orientation, polynomial_order=2.0, sigma_factor=1.0, alpha_max=0.1):
        self.region_type = region_type  # "rect" or "corner"
        self.position = position
        self.orientation = orientation
        self.polynomial_order = polynomial_order  # Reduced to allow smoother transition
        self.sigma_factor = sigma_factor  # Reduced to allow waves to enter
        self.alpha_max = alpha_max  # Reduced frequency-shifting for smoother transition
        if region_type == "rect": self.width, self.height = size
        else: self.radius = size

    def add_to_plot(self, ax, facecolor='none', edgecolor="black", alpha=0.5, linestyle='--'):
        """Delegate PML drawing to beamz.viz.draw_pml."""
        return viz.draw_pml(ax, self, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linestyle=linestyle)

    def get_profile(self, normalized_distance):
        """Calculate PML absorption profile using gradual grading."""
        # Ensure distance is within [0,1]
        d = min(max(normalized_distance, 0.0), 1.0)
        # Create a smooth transition from 0 at the interface
        # Start with nearly zero conductivity at the interface and gradually increase
        if d < 0.05: sigma = 0.01 * (d/0.05)**2
        else: sigma = ((d - 0.05) / 0.95)**self.polynomial_order
        # Smooth frequency-shifting profile
        alpha = self.alpha_max * d**2  # Quadratic profile for smooth transition
        return sigma, alpha
    
    def get_conductivity(self, x, y, dx=None, dt=None, eps_avg=None):
        """Calculate PML conductivity at a point using smooth-transition PML."""
        # Calculate theoretical optimal conductivity based on impedance matching
        if dx is not None and eps_avg is not None:
            # Calculate impedance 
            eta = np.sqrt(MU_0 / (EPS_0 * eps_avg))
            # Optimal conductivity for minimal reflection at interface
            # Reduced from 1.2 to 0.8 for smoother transition
            sigma_max = 1.2 / (eta * dx)
            sigma_max *= self.sigma_factor  # Apply gentler factor
        else: sigma_max = 1.0  # Lower default conductivity
        
        # Get normalized distance based on region type and orientation
        if self.region_type == "rect":
            # Check if point is within rectangular PML region
            if not (self.position[0] <= x <= self.position[0] + self.width and
                    self.position[1] <= y <= self.position[1] + self.height):
                return 0.0
            # Calculate normalized distance from boundary based on orientation
            # Distance should be 0 at inner boundary and 1 at outer boundary
            if self.orientation == "left": distance = 1.0 - (x - self.position[0]) / self.width
            elif self.orientation == "right": distance = (x - self.position[0]) / self.width
            elif self.orientation == "top": distance = (y - self.position[1]) / self.height
            elif self.orientation == "bottom": distance = 1.0 - (y - self.position[1]) / self.height
            else: return 0.0
        
        else: # corner PML
            # Calculate distance from corner to point
            distance_from_corner = np.hypot(x - self.position[0], y - self.position[1])
            # Outside the PML region
            if distance_from_corner > self.radius: return 0.0
            # Check if in correct quadrant
            dx_from_corner = x - self.position[0]
            dy_from_corner = y - self.position[1]
            if self.orientation == "top-left" and (dx_from_corner > 0 or dy_from_corner < 0): return 0.0
            elif self.orientation == "top-right" and (dx_from_corner < 0 or dy_from_corner < 0): return 0.0
            elif self.orientation == "bottom-left" and (dx_from_corner > 0 or dy_from_corner > 0): return 0.0
            elif self.orientation == "bottom-right" and (dx_from_corner < 0 or dy_from_corner > 0): return 0.0
            # Normalize distance (0 at inner edge, 1 at corner)
            distance = distance_from_corner / self.radius
        
        # Get optimized profile values
        sigma_profile, alpha_profile = self.get_profile(distance)
        # Apply stretched-coordinate PML with gradual absorption
        conductivity = sigma_max * sigma_profile
        # The material-dependent scaling might have been causing excessive reflection
        # We'll use a gentler approach that smoothly transitions at the boundary
        if dt is not None:
            # Apply frequency-shifting with reduced effect near boundary
            frequency_factor = 1.0 / (1.0 + alpha_profile)
            conductivity *= frequency_factor
            
        return conductivity
    
    def copy(self):
        """Create a deep copy of the PML boundary."""
        return PML(
            region_type=self.region_type,
            position=self.position,
            size=(self.width, self.height) if self.region_type == "rect" else self.radius,
            orientation=self.orientation,
            polynomial_order=self.polynomial_order,
            sigma_factor=self.sigma_factor,
            alpha_max=self.alpha_max
        )