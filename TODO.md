=== BEAMZ Project Analysis Report ===

1. Project File Structure:
  beamz:
    __init__.py (0.64KB, 23 lines)
    helpers.py (0.35KB, 9 lines)
    const.py (0.21KB, 10 lines)

  beamz/design:
    signals.py (2.30KB, 63 lines)
    materials.py (1.03KB, 48 lines)
    mode.py (7.96KB, 227 lines)
    sources.py (6.72KB, 142 lines)
    structures.py (32.18KB, 712 lines)
    monitors.py (12.20KB, 316 lines)
    helpers.py (2.37KB, 66 lines)

  beamz/optimization:
    adjoint.py (10.81KB, 224 lines)

  beamz/simulation:
    fdtd.py (56.36KB, 1179 lines)
    meshing.py (4.95KB, 106 lines)

  beamz/simulation/backends:
    torch_backend.py (17.57KB, 448 lines)
    numpy_backend.py (3.34KB, 82 lines)
    __init__.py (5.58KB, 166 lines)
    base.py (1.17KB, 44 lines)


3. Number of Files:
Total files: 64
Core package files: 17

4. Functions and Classes:
Total functions: 161
Total classes: 20
Core package functions: 144
Core package classes: 20

5. Lines of Code:
By folder:
  beamz.egg-info: 134 lines
  beamz: 42 lines
  beamz/design: 1574 lines
  beamz/optimization: 224 lines
  beamz/simulation: 1285 lines
  beamz/simulation/backends: 740 lines

Core package total: 3865 lines
Project total: 7881 lines

---

## TODO
+ [ ] Simplify the code as much as possible! Goal: Core 2800 lines of core package code & <500 lines / module
    + [X] Make all structures polygon objects
    + [ ] Convert all structure plotting calls into a method of the polygon object
        + [X] Design.show()
        + [ ] FDTD.show(live=True) Core package total: 3309 lines
    + [ ] Store material values within a structure
    + [ ] Implement the method to get the material from a structure (e.g. in Design)

+ [ ] Speed up the rasterization!!!


+ [ ] Manually implement the adjoint inverse design solver without any complex classes directly in an example!
+ [ ] Manually implement the shape optimization solver without any complex classes directly in an example!
+ [ ] Create the adjoint optimizer module.
+ [ ] Test the backend in PyTorch on a GPU.



GOAL: First inverse design by Friday 16, 2024