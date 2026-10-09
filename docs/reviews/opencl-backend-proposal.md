# Draft issue: OpenCL execution with CUDA scheduler parity

This is a local issue draft; it has not been posted to GitHub.

An OpenCL backend could provide an option for non-NVIDIA GPUs. Its implementation
should follow the established CUDA scheduler and `cuda_mstep` contract rather
than evolve a separate solver. First stabilize CUDA multistep correctness,
workspace planning and measured selection criteria.

Keep geometry, material preparation, CPML slab descriptors, source batching,
DFT plans, absolute clocks and continuation in the shared Python compiler.
Retain one reference definition of Yee curls, coefficient arithmetic, CPML
recurrence and PEC constraints. Device-specific arithmetic must preserve the
same rounding/FMA policy within the accepted numerical tolerance. The compact
quadrant planner can be reused directly; CUDA graph capture and JAX FFI require
an OpenCL-specific execution/interop design.

Deliver the ordinary phase scheduler first, then the two-step column scheduler.
Each implementation must advance all six fields and both CPML recurrence banks,
inject sources at pre-E/H/E phases, sample both substeps for DFT monitors, and
return canonical state with identical continuation semantics. Unsupported
features must fail explicitly. Portability does not justify silently omitting
material loss, absorber terms or observations.

Acceptance checks:

- Run the same complete-state reference cases as CUDA: scalar/packed/dense/lossy
  materials, PEC and asymmetric CPML, all source phases, mode/slab sources,
  masked/windowed/interval DFTs, odd tails, ragged tiles and continuation.
- Test device memory and synchronization with the platform's validation tools.
- Record setup, compilation, first execution, warm throughput and peak memory
  for small, large and narrow domains on named devices and drivers.
- Share scheduler selection inputs and reporting; retain explicit schedule
  overrides. Establish an OpenCL selection policy from its own measurements.
- Require the common parity suite for changes to shared numerical operations.

Open design questions are JAX buffer ownership/interoperability, queue/event
management, compilation/cache keys, supported device capabilities and precision,
and whether a portable kernel representation can share numerical primitives
without coupling the CUDA implementation to OpenCL-specific constraints.

No default backend selection or performance promise is proposed by this issue.
