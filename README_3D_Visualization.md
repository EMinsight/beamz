# Beamz 3D Visualization

This document describes the new 3D visualization capabilities in the beamz design module.

## Overview

The beamz design module now supports both 2D and 3D visualization:

- **2D designs** are automatically visualized using **matplotlib** (fast, static plots)
- **3D designs** are automatically visualized using **plotly** (interactive, web-based)

## Key Features

### ✅ Automatic Detection
- The system automatically detects whether a design contains 3D structures
- Structures are considered 3D if they have:
  - A `depth` parameter (not None)
  - A `z` parameter (not equal to 0)

### ✅ No Window Closing Issues
- Uses plotly for 3D visualization (web-based, no PyVista window issues)
- Opens visualizations in your default web browser
- Clean shutdown and interaction

### ✅ Modern 3D Styling
- **Black outlines**: All 3D objects have crisp black edges for clear definition
- **Flat shading**: Clean, even lighting without confusing shadows or reflections
- **Fixed polygon meshing**: Proper triangulation for complex shapes including rings and tapers
- **Material-based coloring**: Consistent colors for structures with the same material properties
- **Modern UI**: Clean typography, professional layout, and intuitive controls
- **Ground plane**: Subtle ground reference for elevated structures
- **Interactive hover**: Rich tooltips showing structure type and material properties

### ✅ 3D Structure Support
All structure types now support 3D parameters:
- `depth`: Extrusion depth in the Z direction
- `z`: Z-position (height above base plane)

Supported structures:
- `Rectangle`
- `Circle` 
- `Ring`
- `CircularBend`
- `Taper`
- `Polygon` (base class)

## Installation Requirements

For 3D visualization, you need plotly:

```bash
pip install plotly
```

If plotly is not installed, the system will fall back to 2D matplotlib visualization with a warning message.

## Usage Examples

### 2D Design (matplotlib)
```python
from beamz.design.structures import Design, Rectangle
from beamz.design.materials import Material
from beamz.const import µm

# Create 2D design
design = Design(width=10*µm, height=8*µm)
silicon = Material(permittivity=12.0)

# Add 2D structures (no depth specified)
design.add(Rectangle(position=(2*µm, 2*µm), width=3*µm, height=2*µm, material=silicon))

# Automatically uses matplotlib
design.show()  # Opens matplotlib window
```

### 3D Design (plotly)
```python
# Create 3D design
design = Design(width=10*µm, height=8*µm, depth=2*µm)

# Add 3D structures with depth and z-positioning
design.add(Rectangle(
    position=(2*µm, 2*µm), 
    width=3*µm, height=2*µm, 
    depth=1*µm, z=0.5*µm,  # 3D parameters
    material=silicon
))

# Automatically uses plotly
design.show()  # Opens in web browser
```

### Mixed 2D/3D Structures
```python
design = Design(width=10*µm, height=8*µm)

# Add 2D structure first
design.add(Rectangle(position=(1*µm, 1*µm), width=2*µm, height=2*µm, material=silicon))
print(design.is_3d)  # False

# Add 3D structure - design becomes 3D
design.add(Rectangle(position=(4*µm, 4*µm), width=2*µm, height=2*µm, depth=1*µm, material=silicon))
print(design.is_3d)  # True

design.show()  # Uses plotly
```

## 3D Structure Parameters

### Rectangle
```python
Rectangle(
    position=(x, y),      # 2D position
    width=w, height=h,    # 2D dimensions
    depth=d,              # Extrusion depth (None for 2D)
    z=z_pos,              # Z position (0 for base plane)
    material=mat
)
```

### Circle
```python
Circle(
    position=(x, y),      # Center position
    radius=r,             # Radius
    depth=d,              # Extrusion depth
    z=z_pos,              # Z position
    material=mat
)
```

### Ring
```python
Ring(
    position=(x, y),      # Center position
    inner_radius=r1,      # Inner radius
    outer_radius=r2,      # Outer radius
    depth=d,              # Extrusion depth
    z=z_pos,              # Z position
    material=mat
)
```

## Visualization Controls

### show() Method
```python
design.show(unify_structures=True)  # Auto-detects 2D vs 3D
design.show_2d()                   # Force 2D matplotlib
design.show_3d()                   # Force 3D plotly
```

### Plotly 3D Features
- **Interactive rotation**: Click and drag to rotate
- **Zoom**: Mouse wheel or zoom controls
- **Pan**: Shift + drag
- **Black outlines**: Crisp black edges on all 3D objects for clear structure definition
- **Enhanced lighting**: Professional lighting system with multiple light sources
- **Material-based coloring**: Consistent colors for materials with same properties
- **Rich hover information**: Detailed tooltips showing structure type and material properties
- **Modern styling**: Professional typography, clean layout, and intuitive UI
- **Ground plane**: Subtle reference plane for elevated structures
- **Proper scaling**: Automatic unit conversion and axis labels
- **Optimized camera**: Perfect initial viewing angle for each design

## Technical Details

### 3D Mesh Generation
- 2D structures are extruded to create 3D meshes
- **Ear clipping triangulation** for complex polygons (replaces simple fan triangulation)
- Handles complex polygons with holes (via interiors)
- Proper triangulation for Ring, Taper, and CircularBend structures
- **Flat shading** with optimized lighting for clear geometry visibility
- Consistent black outlines (width=3) on all 3D objects

### Performance
- **2D visualization**: Fast matplotlib rendering
- **3D visualization**: Web-based plotly (may be slower for complex structures)
- **Memory**: 3D meshes require more memory than 2D plots

### Fallback Behavior
- If plotly is not available: Falls back to 2D matplotlib with warning
- If invalid 3D structure: Skips structure with warning
- If mesh generation fails: Continues with other structures

## Demo Scripts

Run the included demo scripts to see the features in action:

```bash
# Basic test of 2D/3D detection and visualization
python test_3d_viz.py

# Comprehensive demo with photonic devices
python demo_3d_viz.py

# Enhanced modern 3D styling showcase
python demo_modern_3d.py

# Quick test of enhanced features
python test_enhanced_3d.py

# Test polygon meshing fixes and flat shading
python test_fixed_3d.py
```

## Troubleshooting

### Plotly Not Installed
```
Error: Plotly is required for 3D visualization. Install with: pip install plotly
Falling back to 2D visualization...
```
Solution: Install plotly with `pip install plotly`

### Browser Doesn't Open
- Plotly tries to open in your default browser
- If it doesn't open automatically, look for the local URL in the console
- Copy the URL to your preferred browser

### Performance Issues
- For complex 3D structures with many elements, rendering may be slow
- Consider reducing the number of structures or using 2D visualization for design iteration
- Use `unify_structures=True` to merge similar structures

## Future Enhancements

Potential future improvements:
- Support for more complex 3D geometries
- Volume rendering for materials
- Cross-sectional views
- Animation support for time-dependent simulations
- Export to common 3D formats (STL, OBJ)

## Comparison with PyVista

| Feature | PyVista | Plotly |
|---------|---------|--------|
| Window closing bug | ❌ Yes | ✅ No |
| Web-based | ❌ No | ✅ Yes |
| Interactive | ✅ Yes | ✅ Yes |
| Installation | Complex | Simple |
| Performance | Fast | Moderate |
| Compatibility | Limited | Excellent |

The plotly-based solution resolves the critical window closing issues while providing excellent cross-platform compatibility and ease of installation. 