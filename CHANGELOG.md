# Changelog

## v0.4.3 - 2026-06-26

### Changed
- Updated cosine crossing and modal source example notebooks.

### Fixed
- Fixed mode-source flux calculation and expanded related mode-source and tidy API coverage.

## v0.4.2 - 2026-06-22

### Changed
- Improved matplotlib field plotting with permittivity overlays for field-frame and DFT field views.
- Refined material overlay styling so real fields use darker structure overlays and power fields use lighter overlays.

### Fixed
- Expanded visualization coverage for real-field and power-field overlay behavior.

## v0.4.1 - 2026-06-22

### Changed
- Improved matplotlib field plotting aliases, marker orientation handling, and DFT field component extraction.
- Clarified README terminology around sub-pixel averaging.

### Fixed
- Preserved `SimulationResults.plot_field` behavior while forwarding monitor and field aliases through keyword arguments.
- Added clearer errors for missing DFT field components and expanded visualization coverage.

## v0.4.0 - 2026-06-20

### Added
- Added TFSF support for mode sources.
- Added CPML waveguide benchmarking documentation and scripts.
- Added expanded tests for mode sources, PML/CPML behavior, animation, visualization, and curl kernels.

### Changed
- Improved CPML behavior to reduce waveguide reflections.
- Updated examples and notebooks for modal source and monitor workflows.
- Refined license labeling and documentation around recommended and experimental examples.

## v0.3.2 - 2026-06-13

### Changed
- Reduced compiled-engine memory use and simulation compile overhead for larger 3D runs.
- Simplified boundary and compiled simulation internals while preserving the public API.
- Streamlined examples, benchmark scripts, and development tooling to reduce repository size.
- Updated project licensing metadata to Apache-2.0.

### Fixed
- Improved 3D permittivity handling and memory estimation in simulation setup.
- Fixed mode profile data handling for visualization workflows.
- Adjusted tests and CI coverage around compiled-engine, boundary, and 3D constitutive behavior.

## v0.3.1 - 2026-05-31

### Added
- Added a compact demo example for the demux workflow.

### Changed
- Reduced memory load in boundary and compiled simulation paths.

### Fixed
- Fixed the demux example.

## v0.3.0 - 2026-05-26

### Added
- Added modal port workflows with `Port` and `ModeMonitor` support.
- Added xarray-backed result accessors and plotting conveniences for simulation data.
- Added matplotlib visualization helpers for snapshots, fields, layouts, mode fields, and Tidy3D-style DFT views.
- Added UBC PDK support and improved gdsfactory component handling.
- Added broader 2D/3D physics, monitor, source, and engine-equivalence test coverage.

### Changed
- Replaced the Tidy3D mode-solver dependency path with `micromode`.
- Improved 3D mode-source normalization, source quadrature handling, Yee phase-plane calibration, and monitor DFT/modal projection behavior.
- Refactored CPML/PML handling, including sponge-style absorbing layers and compatibility through `AbsorbingLayer`.
- Improved material sampling on Yee components, full-PEC 3D sampling, and source scattering/S-parameter calculations.
- Promoted the Design material API and added simulation-domain/depth convenience aliases.
- Updated development tooling around `uv`, `ruff`, `vulture`, and Makefile audit commands.

### Fixed
- Fixed 2D and 3D mode-source handedness, power normalization, and source-plane phase-referenced test expectations.
- Fixed cropped Yee profile interpolation and modal monitor projection alignment.
- Fixed 3D CPML compiled-engine behavior and related monitor/source reconstruction cases.
- Restored the dipole example and reverted an incompatible material-handling refactor.
