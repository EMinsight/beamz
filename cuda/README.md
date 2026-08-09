# BeamZ CUDA extension

This directory builds the optional `beamz-cuda` wheel. It registers typed JAX FFI
targets while the main `beamz` package remains usable with JAX alone.

The streamed backend replaces the six large 3D Yee/CPML array programs with fused
magnetic and electric CUDA launches. Bounded multi-step CUDA graphs own complete
3D runs, and one grouped program path covers arbitrary source batches and DFT
monitor schedules. JAX still owns tracing, buffers, and orchestration around that
small native interface, preserving BeamZ's public numerical semantics and JAX
fallback without duplicating configuration-specific FFI targets.

On SM90, the experimental `beamz_cuda_hopper` target uses the same ABI and arithmetic
but maps each component to `32 × 4 × 2` spatial tiles. Each derivative input stages
only its directional halo in shared memory, reusing values across neighboring
updates while keeping the x direction warp-contiguous. Backend selection only
exposes this target on compute capability 9.0 or newer. It remains explicit-only
until hardware parity and throughput gates justify promotion; `auto` and generic
`cuda` use the streamed path.

Build in a CUDA 12 development environment:

```console
python -m pip wheel ./cuda --no-deps
python -m pip install beamz_cuda-*.whl
```

Release builds use precise CUDA division and square-root behavior. Approximate CUDA
intrinsics are available only for controlled experiments with
`-DBEAMZ_CUDA_FAST_MATH=ON`; they must pass the same hardware parity suite before a
result can be used for promotion.

The wheel compiles SASS for SM80, SM86, SM89, and SM90. `backend="auto"` detects and
registers it lazily; `backend="cuda_streamed"` requests it explicitly and
`backend="cuda_hopper"` requests the tiled target. The first
release supports one GPU and float32 3D grids. Multi-GPU and 2D simulations retain
the JAX backend; only the explicitly selected CPML recurrence state may use BF16.

BeamZ validates the extension's explicit ABI version and complete streamed-target
manifest before registering any FFI handler. ABI v10 is distributed as
`beamz-cuda==0.10.0`; an older or partial wheel makes `auto` fall back to JAX and
causes explicit CUDA requests to fail with a compatibility diagnostic.

Regular-grid, lossless CPML simulations with packed source groups use two
XLA-owned field banks. Alternating frozen inputs and outputs removes in-place
read/write hazards and enables safe spatial fusion in the CPML-free core. Set
`BEAMZ_CUDA_DISABLE_CPML_TEMPORAL=1` only for paired diagnostics against the
legacy in-place schedule.

For memory-constrained CPML runs,
`BEAMZ_CUDA_CPML_PSI_PRECISION=bf16` stores only absorber recurrence state in
BF16 while fields and recurrence arithmetic remain FP32. The default is `fp32`;
use BF16 only after validating application-level accuracy. On GA102 the combined
queue uses a precision-specific `32 × 4` absorber tile and remaps the same 128
threads to a `64 × 2` recurrence-free core tile; FP32 retains its measured-optimal
`64 × 4` queue.

The exact cooperative persistent CPML scheduler remains available as an
experimental diagnostic with `BEAMZ_CUDA_ENABLE_PERSISTENT_CPML=1`. Its
warp-tiled resident grid covers every physical cell and bounds individual
launches to avoid starving an interactive display, but the CUDA-graph scheduler
currently sustains substantially higher bandwidth on GA102 and therefore
remains the default. `BEAMZ_CUDA_DISABLE_PERSISTENT_CPML=1` overrides the opt-in.

No CUDA result is promoted without all of the following on real hardware:

- compile with the oldest supported CUDA toolkit and import beside supported JAX;
- compare fields, CPML recurrence buffers, monitor accumulators, clocks, and
  continuation state with JAX over bare, lossy, PEC, CPML, source, and DFT cases;
- run Compute Sanitizer memcheck and racecheck;
- capture Nsight Compute memory throughput and the canonical H100 GCUPS records.

The CUDA workflow always compiles and imports the wheel in a CUDA development
container. Repositories with an H100 self-hosted runner can set the Actions variable
`BEAMZ_H100_RUNNER_ENABLED=true` and label that runner `h100` to additionally run the
32-step PEC/CPML parity envelope and publish canonical benchmark JSON artifacts.

The host FFI decoder deliberately has no CUDA-header dependency and can be checked
on developer machines with the JAX headers alone:

```console
clang++ -std=c++17 -DBEAMZ_CUDA_ABI_VERSION=10 \
  -I"$(python -c 'import jax; print(jax.ffi.include_dir())')" -Icuda/src \
  -fsyntax-only cuda/src/ffi_handler.cc
```
