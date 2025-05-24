import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MatplotlibRectangle, PathPatch, Circle as MatplotlibCircle
from matplotlib.path import Path
import random
import numpy as np
from beamz.design.materials import Material
from beamz.const import µm, EPS_0, MU_0
from beamz.design.sources import ModeSource, GaussianSource
from beamz.design.monitors import Monitor
from beamz.design.helpers import get_si_scale_and_label
from beamz.helpers import display_header, display_status, tree_view, console
import colorsys

class Design:
    def __init__(self, width=5*µm, height=5*µm, depth=0, material=None, color=None, border_color="black", auto_pml=True, pml_size=None):
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
        self.is_3d = False if depth is None or depth == 0 else True
        if auto_pml: self.init_boundaries(pml_size)
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
        if hasattr(structure, 'depth') and structure.depth != 0:
            self.is_3d = True
        elif hasattr(structure, 'position') and len(structure.position) > 2 and structure.position[2] != 0:
            self.is_3d = True
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
            display_status("Shapely library is required for polygon unification. Please install with: pip install shapely", "error")
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
            if isinstance(structure, ModeSource) or isinstance(structure, GaussianSource) or isinstance(structure, Monitor):
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
            if material_key not in material_groups:
                material_groups[material_key] = []
            
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
            if len(structure_group) <= 1:
                # Only one structure with this material, no merging needed
                new_structures.extend([s[0] for s in structure_group])
                for s in structure_group:
                    if s[0] in structures_to_remove:
                        structures_to_remove.remove(s[0])
                continue
                
            # Extract shapely polygons for merging
            shapely_polygons = [p[1] for p in structure_group]
            
            # Get the material from the first structure in the group
            material = structure_group[0][0].material
            
            try:
                # Unify the polygons
                merged = unary_union(shapely_polygons)
                
                # The result could be a single polygon or a multipolygon
                if merged.geom_type == 'Polygon':
                    # Don't slice off the last vertex - our add_to_plot method needs complete vertices
                    exterior_coords = list(merged.exterior.coords[:-1])  # Keep [:-1] to remove duplicate closing vertex from Shapely
                    interior_coords_lists = [list(interior.coords[:-1]) for interior in merged.interiors]
                    
                    if exterior_coords:
                        new_poly = Polygon(vertices=exterior_coords, interiors=interior_coords_lists, material=material)
                        new_structures.append(new_poly)
                        display_status(f"Unified {len(structure_group)} polygons with permittivity={material_key[0]}", "success")
                    else:
                        display_status(f"Failed to convert merged polygon for material {material_key[0]} (no exterior), keeping original {len(structure_group)} structures.", "warning")
                        new_structures.extend([s[0] for s in structure_group])
                        for s_tuple in structure_group: # Ensure these are not removed
                            if s_tuple[0] in structures_to_remove:
                                structures_to_remove.remove(s_tuple[0])

                elif merged.geom_type == 'MultiPolygon':
                    all_geoms_converted_successfully = True
                    temp_new_polys_for_multipolygon = []
                    for geom in merged.geoms:
                        # Keep [:-1] to remove duplicate closing vertex from Shapely (our add_to_plot will add it back)
                        exterior_coords = list(geom.exterior.coords[:-1])
                        interior_coords_lists = [list(interior.coords[:-1]) for interior in geom.interiors]

                        if exterior_coords:
                            new_poly = Polygon(vertices=exterior_coords, interiors=interior_coords_lists, material=material)
                            temp_new_polys_for_multipolygon.append(new_poly)
                        else: # A sub-geometry had no exterior
                            all_geoms_converted_successfully = False
                            display_status(f"Failed to convert a geometry (no exterior) within MultiPolygon for material {material_key[0]}.", "warning")
                            break 
                    
                    if all_geoms_converted_successfully:
                        new_structures.extend(temp_new_polys_for_multipolygon)
                        display_status(f"Unified into {len(merged.geoms)} separate polygons with permittivity={material_key[0]}", "success")
                    else:
                        display_status(f"Reverting unification for material {material_key[0]} due to conversion error in MultiPolygon, keeping original {len(structure_group)} structures.", "warning")
                        new_structures.extend([s[0] for s in structure_group])
                        for s_tuple in structure_group: # Ensure these are not removed
                            if s_tuple[0] in structures_to_remove:
                                structures_to_remove.remove(s_tuple[0])
                else:
                    # If the result is something unexpected, keep the original structures
                    display_status(f"Unexpected geometry type after union: {merged.geom_type} for material {material_key[0]}, keeping original structures", "warning")
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
        
        # Remove the structures that were unified
        for structure in structures_to_remove:
            if structure in self.structures:
                self.structures.remove(structure)
        
        # Add the unified structures, non-polygon structures, and preserved rings back
        self.structures.extend(new_structures)
        self.structures.extend(rings_to_preserve)
        
        # Final report
        display_status(f"Polygon unification complete: {len(structures_to_remove)} structures merged into {len(new_structures)} unified shapes, {len(rings_to_preserve)} isolated rings preserved", "success")
        return True

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
        left_pml = Rectangle(
            position=(0, 0),
            width=pml_size,
            height=self.height,
            material=pml_material,
            color='none',
            is_pml=True  # Flag to identify it as a visual PML marker
        )
        self.structures.append(left_pml)
        # Right PML region
        right_pml = Rectangle(
            position=(self.width - pml_size, 0),
            width=pml_size,
            height=self.height,
            material=pml_material,
            color='none',
            is_pml=True
        )
        self.structures.append(right_pml)
        # Bottom PML region
        bottom_pml = Rectangle(
            position=(0, 0),
            width=self.width,
            height=pml_size,
            material=pml_material,
            color='none',
            is_pml=True
        )
        self.structures.append(bottom_pml)
        # Top PML region
        top_pml = Rectangle(
            position=(0, self.height - pml_size),
            width=self.width,
            height=pml_size,
            material=pml_material,
            color='none',
            is_pml=True
        )
        self.structures.append(top_pml)

    def show(self, unify_structures=True):
        """Display the design visually using 2D matplotlib or 3D plotly."""
        if self.is_3d: self.show_3d(unify_structures)
        else: self.show_2d(unify_structures)
    
    def show_2d(self, unify_structures=True):
        """Display the design using 2D matplotlib visualization."""
        # Determine appropriate SI unit and scale
        max_dim = max(self.width, self.height)
        scale, unit = get_si_scale_and_label(max_dim)
        # Calculate figure size based on domain dimensions
        aspect_ratio = self.width / self.height
        base_size = 5
        if aspect_ratio > 1: figsize = (base_size * aspect_ratio, base_size)
        else: figsize = (base_size, base_size / aspect_ratio)
        # Do we want to show the individual structures or a unified shape?
        if unify_structures: self.unify_polygons()
        # Create a single figure for all structures
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')
        # Now plot each structure
        for structure in self.structures:
            # Use dashed lines for PML regions
            if hasattr(structure, 'is_pml') and structure.is_pml:
                structure.add_to_plot(ax, edgecolor=self.border_color, linestyle='--', facecolor='none', alpha=0.5)
            else: structure.add_to_plot(ax)
        
        # Plot PML boundaries explicitly with dashed lines
        for boundary in self.boundaries:
            if hasattr(boundary, 'add_to_plot'):
                boundary.add_to_plot(ax, edgecolor=self.border_color, linestyle='--', facecolor='none', alpha=0.5)
        
        # Set proper limits, title and label, and ensure the full design is visible
        ax.set_title('Design Layout')
        ax.set_xlabel(f'X ({unit})')
        ax.set_ylabel(f'Y ({unit})')
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        # Update tick labels with scaled values
        ax.xaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
        ax.yaxis.set_major_formatter(lambda x, pos: f'{x*scale:.1f}')
        # Adjust layout for clean appearance
        plt.tight_layout()
        plt.show()
    
    def show_3d(self, unify_structures=True):
        """Display the design using 3D plotly visualization."""
        try:
            import plotly.graph_objects as go
            import plotly.figure_factory as ff
            from plotly.subplots import make_subplots
        except ImportError:
            display_status("Plotly is required for 3D visualization. Install with: pip install plotly", "error")
            display_status("Falling back to 2D visualization...", "warning")
            self.show_2d(unify_structures)
            return
        
        # Determine appropriate SI unit and scale
        max_dim = max(self.width, self.height, self.depth if self.depth else 0)
        scale, unit = get_si_scale_and_label(max_dim)
        # Do we want to show the individual structures or a unified shape?
        if unify_structures: self.unify_polygons()
        # Create 3D figure with modern styling
        fig = go.Figure()
        # Default depth for 2D structures in 3D view
        default_depth = self.depth if self.depth else min(self.width, self.height) * 0.1
        
        # Modern color palette for different materials
        modern_colors = [
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange  
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#bcbd22',  # Olive
            '#17becf'   # Cyan
        ]
        
        # Track materials for consistent coloring
        material_colors = {}
        color_index = 0
        # Process each structure
        for structure in self.structures:
            # Skip PML structures for now (they would clutter the 3D view)
            if hasattr(structure, 'is_pml') and structure.is_pml:
                continue
                
            # Get structure depth
            struct_depth = getattr(structure, 'depth', default_depth)
            struct_z = getattr(structure, 'z', 0)
            # Add 3D representation based on structure type
            mesh_data = self._structure_to_3d_mesh(structure, struct_depth, struct_z)
            if mesh_data:
                x, y, z = mesh_data['vertices']
                i, j, k = mesh_data['faces']
                # Assign color based on material properties for consistency
                material_key = None
                if hasattr(structure, 'material') and structure.material:
                    material_key = (
                        getattr(structure.material, 'permittivity', 1.0),
                        getattr(structure.material, 'permeability', 1.0),
                        getattr(structure.material, 'conductivity', 0.0)
                    )
                
                if material_key not in material_colors:
                    if hasattr(structure, 'color') and structure.color and structure.color != 'none':
                        material_colors[material_key] = structure.color
                    else:
                        material_colors[material_key] = modern_colors[color_index % len(modern_colors)]
                        color_index += 1
                
                color = material_colors[material_key]
                
                # Convert hex color to rgba if needed for transparency
                if color.startswith('#'):
                    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                    face_color = f"rgba({r},{g},{b},0.8)"
                    edge_color = "rgba(0,0,0,0.8)"  # Black edges
                else:
                    face_color = color
                    edge_color = "black"
                
                # Add material name to hover info if available
                hovertext = f"{structure.__class__.__name__}"
                if hasattr(structure, 'material') and structure.material:
                    if hasattr(structure.material, 'name'):
                        hovertext += f"<br>Material: {structure.material.name}"
                    if hasattr(structure.material, 'permittivity'):
                        hovertext += f"<br>εᵣ = {structure.material.permittivity:.1f}"
                    if hasattr(structure.material, 'permeability') and structure.material.permeability != 1.0:
                        hovertext += f"<br>μᵣ = {structure.material.permeability:.1f}"
                    if hasattr(structure.material, 'conductivity') and structure.material.conductivity != 0.0:
                        hovertext += f"<br>σ = {structure.material.conductivity:.2e} S/m"
                
                fig.add_trace(go.Mesh3d(
                    x=x, y=y, z=z,
                    i=i, j=j, k=k,
                    color=face_color,  # Use 'color' for uniform coloring
                    opacity=1.0,  # Slightly higher opacity for better visibility
                    name=hovertext,
                    showscale=False,
                    hovertemplate=hovertext + "<extra></extra>",
                    # Prominent black outlines for all shapes
                    contour=dict(
                        show=True,
                        color="black",  # Always black outlines
                        width=5  # Thicker lines for better visibility
                    ),
                    # Flat shading - minimal lighting for clean appearance
                    lighting=dict(
                        ambient=0.8,    # High ambient for flat appearance
                        diffuse=0.2,    # Low diffuse for minimal shadows
                        fresnel=0.0,    # No fresnel effects
                        specular=0.5,   # No specular highlights  
                        roughness=1.0   # Maximum roughness for flat appearance
                    ),
                    # Simple light position
                    lightposition=dict(
                        x=0, y=50, z=100  # Far away light for even illumination
                    ),
                    # Force flat shading
                    flatshading=True
                ))
        
        # Modern layout with better styling
        fig.update_layout(
            title=dict(
                text='3D Design Layout',
                x=0.5,
                font=dict(size=20, color='#2c3e50', family="Arial Black")
            ),
            scene=dict(
                xaxis=dict(
                    title=dict(text=f'X ({unit})', font=dict(size=14, color='#34495e')),
                    range=[0, self.width],
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.3)',
                    showbackground=True,
                    backgroundcolor='rgba(248,249,250,0.8)',
                    tickmode='array',
                    tickvals=np.linspace(0, self.width, 6),
                    ticktext=[f'{val*scale:.1f}' for val in np.linspace(0, self.width, 6)],
                    tickfont=dict(size=11, color='#34495e')
                ),
                yaxis=dict(
                    title=dict(text=f'Y ({unit})', font=dict(size=14, color='#34495e')),
                    range=[0, self.height],
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.3)',
                    showbackground=True,
                    backgroundcolor='rgba(248,249,250,0.8)',
                    tickmode='array',
                    tickvals=np.linspace(0, self.height, 6),
                    ticktext=[f'{val*scale:.1f}' for val in np.linspace(0, self.height, 6)],
                    tickfont=dict(size=11, color='#34495e')
                ),
                zaxis=dict(
                    title=dict(text=f'Z ({unit})', font=dict(size=14, color='#34495e')),
                    range=[0, self.depth if self.depth else default_depth],
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.3)',
                    showbackground=True,
                    backgroundcolor='rgba(248,249,250,0.8)',
                    tickmode='array',
                    tickvals=np.linspace(0, self.depth if self.depth else default_depth, 6),
                    ticktext=[f'{val*scale:.1f}' for val in np.linspace(0, self.depth if self.depth else default_depth, 6)],
                    tickfont=dict(size=11, color='#34495e')
                ),
                aspectmode='manual',
                aspectratio=dict(
                    x=1,
                    y=self.height/self.width if self.width > 0 else 1,
                    z=(self.depth if self.depth else default_depth)/self.width if self.width > 0 else 1
                ),
                # Modern camera angle for better initial view
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            # Modern layout styling
            width=900,
            height=700,
            margin=dict(l=60, r=60, t=80, b=60),
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color='#2c3e50'),
            # Enhanced legend
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left", 
                x=1.02,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
                font=dict(size=10)
            )
        )
        
        # Add subtle shadow effect with a ground plane if structures are elevated
        if any(getattr(s, 'z', 0) > 0 for s in self.structures if not (hasattr(s, 'is_pml') and s.is_pml)):
            # Create a subtle ground plane
            ground_x = [0, self.width, self.width, 0]
            ground_y = [0, 0, self.height, self.height]
            ground_z = [0, 0, 0, 0]
            
            fig.add_trace(go.Mesh3d(
                x=ground_x + ground_x,  # Duplicate for thickness
                y=ground_y + ground_y,
                z=ground_z + [-default_depth*0.05]*4,  # Thin ground plane
                i=[0, 0, 4, 4, 0, 1, 2, 3],
                j=[1, 3, 5, 7, 4, 5, 6, 7], 
                k=[2, 2, 6, 6, 1, 2, 3, 0],
                color='rgba(220,220,220,0.3)',  # Use 'color' instead of 'facecolor'
                name="Ground Plane",
                showlegend=False,
                hoverinfo='skip',
                # Consistent flat shading for ground plane
                lighting=dict(
                    ambient=0.8,
                    diffuse=0.2,
                    fresnel=0.0,
                    specular=0.0,
                    roughness=1.0
                ),
                flatshading=True,
                # No outline for ground plane (to keep it subtle)
                contour=dict(show=False)
            ))
        
        fig.show()
    
    def _structure_to_3d_mesh(self, structure, depth, z_offset=0):
        """Convert a 2D structure to 3D mesh data for plotly with improved triangulation."""
        if not hasattr(structure, 'vertices') or not structure.vertices: return None
        # Ensure depth has a valid value
        if depth is None: depth = 0.1 * min(self.width, self.height)
        vertices_2d = structure.vertices
        n_vertices = len(vertices_2d)
        if n_vertices < 3: return None  # Need at least 3 vertices for a face
        # Handle polygons with holes (like Ring structures)
        interior_paths = getattr(structure, 'interiors', [])
        # For complex polygons with holes, use a different approach
        if interior_paths and len(interior_paths) > 0:
            return self._triangulate_polygon_with_holes(vertices_2d, interior_paths, depth, z_offset)
    
        # Simple polygon triangulation using corrected ear clipping
        def simple_triangulation(vertices):
            """Simple triangulation for convex/simple polygons."""
            if len(vertices) < 3: return []
            if len(vertices) == 3: return [(0, 1, 2)]
            if len(vertices) == 4: return [(0, 1, 2), (0, 2, 3)] # split quads into two triangles
            # For more complex polygons, use fan triangulation from centroid
            triangles = []
            for i in range(len(vertices) - 2): triangles.append((0, i + 1, i + 2))
            return triangles
        
        # Create 3D vertices by extruding the 2D shape
        vertices_3d = []
        # Bottom face vertices (z = z_offset)
        for x, y in vertices_2d: vertices_3d.append([x, y, z_offset])
        # Top face vertices (z = z_offset + depth)
        for x, y in vertices_2d: vertices_3d.append([x, y, z_offset + depth])
        # Extract coordinates
        x_coords = [v[0] for v in vertices_3d]
        y_coords = [v[1] for v in vertices_3d]
        z_coords = [v[2] for v in vertices_3d]
        # Create triangular faces for the mesh
        faces_i, faces_j, faces_k = [], [], []
        # Bottom face triangulation (CCW when viewed from above = normal pointing down)
        bottom_triangles = simple_triangulation(vertices_2d)
        for tri in bottom_triangles:
            faces_i.append(tri[0])
            faces_j.append(tri[1]) 
            faces_k.append(tri[2])
        # Top face triangulation (CW when viewed from above = normal pointing up)
        for tri in bottom_triangles:
            faces_i.append(tri[0] + n_vertices)
            faces_j.append(tri[2] + n_vertices)  # Swap j,k for opposite winding
            faces_k.append(tri[1] + n_vertices)
        # Side faces (rectangles split into two triangles each)
        for i in range(n_vertices):
            next_i = (i + 1) % n_vertices
            # Each side face is a rectangle with 4 vertices:
            # Bottom: i, next_i
            # Top: i + n_vertices, next_i + n_vertices
            # Triangle 1: bottom-left, bottom-right, top-left (CCW from outside)
            faces_i.append(i)
            faces_j.append(next_i)
            faces_k.append(i + n_vertices)
            # Triangle 2: bottom-right, top-right, top-left (CCW from outside)
            faces_i.append(next_i)
            faces_j.append(next_i + n_vertices)
            faces_k.append(i + n_vertices)
        
        return {
            'vertices': (x_coords, y_coords, z_coords),
            'faces': (faces_i, faces_j, faces_k)
        }
    
    def _triangulate_polygon_with_holes(self, exterior_vertices, interior_paths, depth, z_offset):
        """Handle polygons with holes (like Ring structures) using proper triangulation."""
        try:
            # Try to use a proper constrained triangulation library if available
            import scipy.spatial
            from matplotlib.path import Path
            # For now, use a simpler approach for Ring-like structures
            # Create separate meshes for exterior and interior, then combine
            n_ext = len(exterior_vertices)
            total_vertices = n_ext
            all_vertices_2d = list(exterior_vertices)
            # Add interior vertices
            interior_starts = []
            for interior in interior_paths:
                interior_starts.append(total_vertices)
                all_vertices_2d.extend(interior)
                total_vertices += len(interior)
            
            # Create 3D vertices
            vertices_3d = []
            # Bottom face vertices
            for x, y in all_vertices_2d:
                vertices_3d.append([x, y, z_offset])
            # Top face vertices
            for x, y in all_vertices_2d:
                vertices_3d.append([x, y, z_offset + depth])
            
            # Extract coordinates
            x_coords = [v[0] for v in vertices_3d]
            y_coords = [v[1] for v in vertices_3d]
            z_coords = [v[2] for v in vertices_3d]
            faces_i, faces_j, faces_k = [], [], []
            
            # For Ring structures, create triangular strip between inner and outer
            if len(interior_paths) == 1 and len(interior_paths[0]) == len(exterior_vertices):
                # This is likely a Ring structure with equal number of points
                inner_start = interior_starts[0]
                # Create triangular strips connecting outer to inner
                for i in range(n_ext):
                    next_i = (i + 1) % n_ext
                    # Bottom face triangles (outer ring)
                    outer_i = i
                    outer_next = next_i
                    inner_i = inner_start + i
                    inner_next = inner_start + next_i
                    # Triangle 1: outer_i -> outer_next -> inner_i
                    faces_i.append(outer_i)
                    faces_j.append(outer_next)
                    faces_k.append(inner_i)
                    # Triangle 2: outer_next -> inner_next -> inner_i  
                    faces_i.append(outer_next)
                    faces_j.append(inner_next)
                    faces_k.append(inner_i)
                    # Top face triangles (same pattern but offset by total_vertices)
                    top_offset = total_vertices
                    # Triangle 1: outer_i -> inner_i -> outer_next (reversed winding)
                    faces_i.append(outer_i + top_offset)
                    faces_j.append(inner_i + top_offset)
                    faces_k.append(outer_next + top_offset)
                    # Triangle 2: outer_next -> inner_i -> inner_next (reversed winding)
                    faces_i.append(outer_next + top_offset)
                    faces_j.append(inner_i + top_offset)
                    faces_k.append(inner_next + top_offset)
                
                # Side faces for outer ring
                for i in range(n_ext):
                    next_i = (i + 1) % n_ext
                    # Outer side face
                    faces_i.append(i)
                    faces_j.append(next_i)
                    faces_k.append(i + total_vertices)
                    faces_i.append(next_i)
                    faces_j.append(next_i + total_vertices)
                    faces_k.append(i + total_vertices)
                
                # Side faces for inner ring (reversed winding for inward-facing)
                for i in range(len(interior_paths[0])):
                    next_i = (i + 1) % len(interior_paths[0])
                    inner_i = inner_start + i
                    inner_next = inner_start + next_i
                    # Inner side face (reversed winding)
                    faces_i.append(inner_i + total_vertices)
                    faces_j.append(inner_next + total_vertices)
                    faces_k.append(inner_i)
                    faces_i.append(inner_i)
                    faces_j.append(inner_next + total_vertices)
                    faces_k.append(inner_next)
            
            return {
                'vertices': (x_coords, y_coords, z_coords),
                'faces': (faces_i, faces_j, faces_k)
            }
            
        except Exception as e:
            # Fallback to simple approach if triangulation fails
            print(f"Warning: Complex triangulation failed, using simple approach: {e}")
            return self._structure_to_3d_mesh_simple(exterior_vertices, depth, z_offset)
    
    def _structure_to_3d_mesh_simple(self, vertices_2d, depth, z_offset):
        """Fallback simple mesh generation."""
        n_vertices = len(vertices_2d)
        
        # Create 3D vertices
        vertices_3d = []
        for x, y in vertices_2d: vertices_3d.append([x, y, z_offset])
        for x, y in vertices_2d: vertices_3d.append([x, y, z_offset + depth])
        
        x_coords = [v[0] for v in vertices_3d]
        y_coords = [v[1] for v in vertices_3d]
        z_coords = [v[2] for v in vertices_3d]
        faces_i, faces_j, faces_k = [], [], []
        
        # Simple fan triangulation for bottom
        for i in range(1, n_vertices - 1):
            faces_i.append(0)
            faces_j.append(i)
            faces_k.append(i + 1)
        # Simple fan triangulation for top (reversed)
        for i in range(1, n_vertices - 1):
            faces_i.append(n_vertices)
            faces_j.append(n_vertices + i + 1)
            faces_k.append(n_vertices + i)
        # Side faces
        for i in range(n_vertices):
            next_i = (i + 1) % n_vertices
            faces_i.append(i)
            faces_j.append(next_i)
            faces_k.append(i + n_vertices)
            
            faces_i.append(next_i)
            faces_j.append(next_i + n_vertices)
            faces_k.append(i + n_vertices)
        
        return {
            'vertices': (x_coords, y_coords, z_coords),
            'faces': (faces_i, faces_j, faces_k)
        }
    
    def _point_in_triangle(self, point, a, b, c):
        """Check if a point is inside a triangle using barycentric coordinates."""
        x, y = point
        x1, y1 = a
        x2, y2 = b
        x3, y3 = c
        denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(denominator) < 1e-10: return False
        a_coord = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator
        b_coord = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator
        c_coord = 1 - a_coord - b_coord
        return a_coord >= 0 and b_coord >= 0 and c_coord >= 0

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
            if isinstance(structure, Rectangle):
                if structure.is_pml:
                    # Skip visual PML structures - they're just for display
                    continue
                if structure.point_in_polygon(x, y, z):
                    epsilon = structure.material.permittivity
                    mu = structure.material.permeability
                    sigma_base = structure.material.conductivity
                    break
            elif isinstance(structure, Circle):
                if np.hypot(x - structure.position[0], y - structure.position[1]) <= structure.radius:
                    epsilon = structure.material.permittivity
                    mu = structure.material.permeability
                    sigma_base = structure.material.conductivity
                    break
            elif isinstance(structure, Ring):
                distance = np.hypot(x - structure.position[0], y - structure.position[1])
                if structure.inner_radius <= distance <= structure.outer_radius:
                    epsilon = structure.material.permittivity
                    mu = structure.material.permeability
                    sigma_base = structure.material.conductivity
                    break
            elif isinstance(structure, CircularBend):
                distance = np.hypot(x - structure.position[0], y - structure.position[1])
                if structure.inner_radius <= distance <= structure.outer_radius:
                    epsilon = structure.material.permittivity
                    mu = structure.material.permeability
                    sigma_base = structure.material.conductivity
                    break
            elif isinstance(structure, Polygon):
                if structure.point_in_polygon(x, y, z):
                    epsilon = structure.material.permittivity
                    mu = structure.material.permeability
                    sigma_base = structure.material.conductivity
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

    def get_tree_view(self):
        """Return a structured view of the design as a tree"""
        design_data = {
            "Properties": {
                "Width": self.width,
                "Height": self.height,
                "Depth": self.depth,
                "Dimension": "3D" if self.is_3d else "2D"
            },
            "Structures": {},
            "Sources": {},
            "Monitors": {}
        }
        
        # Add structure data
        for idx, structure in enumerate(self.structures):
            if isinstance(structure, ModeSource) or isinstance(structure, GaussianSource) or isinstance(structure, Monitor):
                continue
                
            struct_type = structure.__class__.__name__
            if struct_type not in design_data["Structures"]:
                design_data["Structures"][struct_type] = []
                
            struct_info = {"position": getattr(structure, "position", None)}
            if hasattr(structure, "material"):
                mat = structure.material
                struct_info["material"] = {
                    "permittivity": getattr(mat, "permittivity", None),
                    "permeability": getattr(mat, "permeability", None),
                    "conductivity": getattr(mat, "conductivity", None)
                }
            design_data["Structures"][struct_type].append(struct_info)
        
        # Add source data
        for idx, source in enumerate(self.sources):
            source_type = source.__class__.__name__
            if source_type not in design_data["Sources"]:
                design_data["Sources"][source_type] = []
            
            source_info = {
                "position": source.position,
                "wavelength": getattr(source, "wavelength", None)
            }
            design_data["Sources"][source_type].append(source_info)
        
        # Add monitor data
        for idx, monitor in enumerate(self.monitors):
            monitor_type = monitor.__class__.__name__
            if monitor_type not in design_data["Monitors"]:
                design_data["Monitors"][monitor_type] = []
            
            monitor_info = {
                "position": monitor.position,
                "size": getattr(monitor, "size", None)
            }
            design_data["Monitors"][monitor_type].append(monitor_info)
            
        return design_data
        
    def display_tree(self):
        """Display the design as a hierarchical tree"""
        design_data = self.get_tree_view()
        tree_view(design_data, "Design Structure")

class Polygon:
    def __init__(self, vertices=None, material=None, color=None, optimize=False, interiors=None):
        self.vertices = self._ensure_3d_vertices(vertices if vertices is not None else []) # Exterior path
        self.interiors = [self._ensure_3d_vertices(interior) for interior in (interiors if interiors is not None else [])] # List of interior paths
        self.material = material
        self.optimize = optimize
        self.color = color if color is not None else self.get_random_color_consistent()
    
    def _ensure_3d_vertices(self, vertices):
        """Convert 2D vertices (x,y) to 3D vertices (x,y,z) with z=0 if not provided."""
        if not vertices:
            return []
        result = []
        for v in vertices:
            if len(v) == 2:
                result.append((v[0], v[1], 0.0))  # Add z=0 for 2D vertices
            elif len(v) == 3:
                result.append(v)  # Keep 3D vertices as-is
            else:
                raise ValueError(f"Vertex must have 2 or 3 coordinates, got {len(v)}")
        return result
    
    def _vertices_2d(self, vertices=None):
        """Get 2D projection of vertices for plotting (x,y only)."""
        if vertices is None:
            vertices = self.vertices
        return [(v[0], v[1]) for v in vertices]
    
    def get_random_color_consistent(self, saturation=0.6, value=0.7):
        """Generate a random color with consistent perceived brightness and saturation."""
        hue = random.random() # Generate random hue (0-1)
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
    
    def shift(self, x, y, z=0):
        """Shift the polygon by (x,y,z) and return self for method chaining. z is optional for 2D compatibility."""
        if self.vertices: 
            self.vertices = [(v[0] + x, v[1] + y, v[2] + z) for v in self.vertices]
        new_interiors_paths = []
        for interior_path in self.interiors:
            if interior_path: # Ensure path is not empty
                new_interiors_paths.append([(v[0] + x, v[1] + y, v[2] + z) for v in interior_path])
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
            else:
                x_center, y_center, z_center = (point[0], point[1], point[2] if len(point) > 2 else 0)

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
        """Add the polygon as a patch to the axis, handling holes correctly.
        For 3D vertices, project to 2D (xy plane) for plotting."""
        if facecolor is None: facecolor = self.color
        if alpha is None: alpha = 1.0 # Default alpha to 1.0 for visibility
        if linestyle is None: linestyle = '-'
        
        if not self.vertices: # No exterior to draw
            return

        # Path components: first is exterior, subsequent are interiors
        all_path_coords = []
        all_path_codes = []

        # Exterior path - project 3D to 2D for plotting
        vertices_2d = self._vertices_2d(self.vertices)
        if len(vertices_2d) > 0:
            # Add all vertices
            all_path_coords.extend(vertices_2d)
            # Add the first vertex again to close the path visually
            all_path_coords.append(vertices_2d[0])
            # Set codes: MOVETO for first vertex, LINETO for middle vertices, CLOSEPOLY for last
            all_path_codes.append(Path.MOVETO)
            if len(vertices_2d) > 1:
                all_path_codes.extend([Path.LINETO] * (len(vertices_2d) - 1))
            all_path_codes.append(Path.CLOSEPOLY)

        # Interior paths - project 3D to 2D for plotting
        for interior_v_list in self.interiors:
            if interior_v_list and len(interior_v_list) > 0:
                interior_2d = self._vertices_2d(interior_v_list)
                # Add all interior vertices
                all_path_coords.extend(interior_2d)
                # Add the first vertex again to close the interior path
                all_path_coords.append(interior_2d[0])
                # Set codes: MOVETO for first vertex, LINETO for middle vertices, CLOSEPOLY for last
                all_path_codes.append(Path.MOVETO)
                if len(interior_2d) > 1:
                    all_path_codes.extend([Path.LINETO] * (len(interior_2d) - 1))
                all_path_codes.append(Path.CLOSEPOLY)
        
        if not all_path_coords or not all_path_codes: # Nothing to draw
            return
            
        path = Path(np.array(all_path_coords), np.array(all_path_codes))
        patch = PathPatch(path, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor, linestyle=linestyle)
        ax.add_patch(patch)

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
    def __init__(self, position=(0,0,0), width=1, height=1, depth=1, material=None, color=None, is_pml=False, optimize=False):
        # Handle 2D position input (x,y) by adding z=0
        if len(position) == 2: position = (position[0], position[1], 0.0)
        elif len(position) == 3: position = position
        else: raise ValueError("Position must be (x,y) or (x,y,z)")
        # Calculate vertices for the rectangle in 3D
        x, y, z = position
        vertices = [(x, y, z),  # Bottom left
                    (x + width, y, z),  # Bottom right
                    (x + width, y + height, z),  # Top right
                    (x, y + height, z)]
        super().__init__(vertices=vertices, material=material, color=color, optimize=optimize)
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
    def __init__(self, position=(0,0), radius=1, points=32, material=None, color=None, optimize=False):
        # Handle 2D position input (x,y) by adding z=0
        if len(position) == 2: position = (position[0], position[1], 0.0)
        elif len(position) == 3: position = position
        else: raise ValueError("Position must be (x,y) or (x,y,z)")
        theta = np.linspace(0, 2*np.pi, points, endpoint=False)
        vertices = [(position[0] + radius * np.cos(t), position[1] + radius * np.sin(t), position[2]) for t in theta]
        super().__init__(vertices=vertices, material=material, color=color, optimize=optimize)
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
                     material=self.material, color=self.color, optimize=self.optimize)

class Ring(Polygon):
    def __init__(self, position=(0,0), inner_radius=1, outer_radius=2, material=None, color=None, optimize=False, points=256):
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
                         material=material, color=color, optimize=optimize, depth=depth, z=z)
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
                 facecolor=None, optimize=False, points=64):
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
    def __init__(self, position=(0,0), input_width=1, output_width=0.5, length=1, material=None, color=None, optimize=False):
        # Handle 2D position input (x,y) by adding z=0
        if len(position) == 2:
            position = (position[0], position[1], 0.0)
        elif len(position) == 3:
            position = position
        else:
            raise ValueError("Position must be (x,y) or (x,y,z)")
            
        # Calculate vertices for the trapezoid shape in 3D
        x, y, z = position
        vertices = [(x, y - input_width/2, z),  # Bottom left
                    (x + length, y - output_width/2, z),  # Bottom right
                    (x + length, y + output_width/2, z),  # Top right
                    (x, y + input_width/2, z)] # Top left
        super().__init__(vertices=vertices, material=material, color=color)
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
        """Add the PML boundary to a plot with dashed lines."""
        if self.region_type == "rect":
            # Create a rectangle patch for rectangular PML regions
            rect_patch = MatplotlibRectangle(
                (self.position[0], self.position[1]),
                self.width, self.height,
                fill=False, 
                edgecolor=edgecolor,
                linestyle=linestyle,
                alpha=alpha
            )
            ax.add_patch(rect_patch)
        elif self.region_type == "corner":
            # Use a rectangle for corner PML regions as well
            # Position and size depend on orientation
            if self.orientation == "bottom-left":
                rect_patch = MatplotlibRectangle(
                    (self.position[0] - self.radius, self.position[1] - self.radius),
                    self.radius, self.radius,
                    fill=False,
                    edgecolor=edgecolor,
                    linestyle=linestyle,
                    alpha=alpha
                )
            elif self.orientation == "bottom-right":
                rect_patch = MatplotlibRectangle(
                    (self.position[0], self.position[1] - self.radius),
                    self.radius, self.radius,
                    fill=False,
                    edgecolor=edgecolor,
                    linestyle=linestyle,
                    alpha=alpha
                )
            elif self.orientation == "top-right":
                rect_patch = MatplotlibRectangle(
                    (self.position[0], self.position[1]),
                    self.radius, self.radius,
                    fill=False,
                    edgecolor=edgecolor,
                    linestyle=linestyle,
                    alpha=alpha
                )
            elif self.orientation == "top-left":
                rect_patch = MatplotlibRectangle(
                    (self.position[0] - self.radius, self.position[1]),
                    self.radius, self.radius,
                    fill=False,
                    edgecolor=edgecolor,
                    linestyle=linestyle,
                    alpha=alpha
                )
            ax.add_patch(rect_patch)

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