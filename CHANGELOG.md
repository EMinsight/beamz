# Changelog

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
