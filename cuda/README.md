# BeamZ CUDA extension

This directory builds the optional `beamz-cuda` wheel. It registers typed JAX FFI
targets while the main `beamz` package remains usable with JAX alone.

The streamed backend replaces the six large 3D Yee/CPML array programs with three
fused magnetic and three fused electric CUDA launches per timestep. Source phases,
PEC restoration, monitor accumulation, continuation state, and the timestep loop
remain JAX transformations around those calls. This keeps the CUDA surface local
while preserving BeamZ's public numerical semantics.

Build in a CUDA 12 development environment:

```console
python -m pip wheel ./cuda --no-deps
python -m pip install beamz_cuda-*.whl
```

The wheel compiles SASS for SM80, SM89, and SM90. `backend="auto"` detects and
registers it lazily; `backend="cuda_streamed"` requests it explicitly. The first
release supports one GPU and float32 3D grids. Multi-GPU and 2D simulations retain
the JAX backend.

No CUDA result is promoted without all of the following on real hardware:

- compile with the oldest supported CUDA toolkit and import beside supported JAX;
- compare fields, CPML recurrence buffers, monitor accumulators, clocks, and
  continuation state with JAX over bare, lossy, PEC, CPML, source, and DFT cases;
- run Compute Sanitizer memcheck and racecheck;
- capture Nsight Compute memory throughput and the canonical H100 GCUPS records.

The host FFI decoder deliberately has no CUDA-header dependency and can be checked
on developer machines with the JAX headers alone:

```console
clang++ -std=c++17 -DBEAMZ_CUDA_ABI_VERSION=1 \
  -I"$(python -c 'import jax; print(jax.ffi.include_dir())')" -Icuda/src \
  -fsyntax-only cuda/src/ffi_handler.cc
```
