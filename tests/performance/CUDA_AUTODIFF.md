# CUDA FDTD and reverse-mode performance audit

The current branch preserves portable JAX imports while adding first-order CUDA
reverse mode. Fresh measurements after integrating `main` (`6836a6fe`) use
component 0.24.0 / ABI 24 on the same RTX 3090 and JAX/JAXLIB 0.9.0.

| Current workload | GPU JAX median | CUDA median | JAX / CUDA |
| --- | ---: | ---: | ---: |
| FDTD, CPML + heterogeneous guide + source + DFT, 64×128×192, 512 steps | 187.67 ms | 107.40 ms | 1.75× |
| Bend inverse objective, 100 nm | 50.65 ms | 39.08 ms | 1.30× |
| Bend inverse objective **and full gradient**, 100 nm | 255.49 ms | 255.58 ms | 1.00× |

Current GPU JAX is substantially faster than the earlier reference below. CUDA
has no demonstrated full-gradient speedup at 100 nm on this updated branch;
its measured difference is below 0.3%, with a bootstrap 95% interval of
0.998–1.001×. The objective-only interval is
1.278–1.302×. These describe within-run timing variation, not portability across
machines. Initial-design gradient relative L2 disagreement is 2.03×10⁻⁷; the
objective absolute difference is 4.19×10⁻⁹. Centered finite differences at steps
0.2, 0.1 and 0.05 agree with the analytic directional derivative within 0.1%.
The standard FDTD maximum relative state-leaf error is 8.05×10⁻⁵.

The fresh raw records are [main_fdtd_medium.json](cuda_autodiff_results/main_fdtd_medium.json)
and [main_inverse_100nm.json](cuda_autodiff_results/main_inverse_100nm.json).
They use synchronized, alternating samples, three warmups, device-resident
inputs, identical physics and checkpointing, and include source/binary checksums.
The final implementation retains current `main`'s forward kernels and scheduling.
Differentiable observation clocks use an initial origin plus integer step count,
avoiding accumulated float32 clock drift while preserving clock gradients.
Padding and paired temporal schedules are disabled inside the differentiable
scan; axis permutations and dispersive material binding are rejected explicitly.


The updated notebook was executed through all 25 cells and 160 optimization
updates. Guided transmission is 88.333% on the 100 nm training mesh and 85.989%
after thresholding. At 25 nm with a 400 fs window, the corresponding values are
80.032% and 78.604%. At 50 nm, forward and reversed excitation give matching
transmission (80.704% continuous, 79.898% binary). Final material transpose
asymmetry is 2.16×10⁻⁸; final CUDA/JAX gradient relative disagreement is 0.0009%.
These mesh-dependent results supersede the historical notebook scores below.
Current [verification rows](cuda_autodiff_results/main_notebook_verification.json)
and [optimized parameters](cuda_autodiff_results/main_optimized_parameters.npz)
are included for review and reproduction.


Current contributor checks passed full Ruff formatting/lint, changed-module
Pyright, dead-code checks, native CMake compilation, and wheel/sdist build and
distribution validation. Selected tests cover 246 unique CPU cases, 28 GPU
adjoint cases, and 11 passing ordinary-CUDA cases. Four additional ordinary
CUDA cases fail identically on `main`: two smooth-material comparisons exceed
an FP32 near-cancellation tolerance and two lossy cases assert an outdated
schedule capability. The baseline uses unchanged main Python and the identical
main forward kernels, adapting only its ABI number to load this ABI-24 build.
`make audit` stops at five existing GDSFactory attribute typing errors in
`beamz/design/gds.py`; its full-suite/coverage stage did not run. The current
standalone harness passes 72 cases under both initcheck (zero errors) and
racecheck (zero hazards). See [main_validation.json](cuda_autodiff_results/main_validation.json).

## Historical measurements before integrating current main

The following records describe the earlier ABI 14 implementation on base
`e4c2f3d4`; they are retained as optimization history, **not current performance
claims**. That implementation's gradient evaluation was 6.24–8.04 times faster
than its original CUDA autodiff path. These measurements do not establish a
speedup for every simulation or GPU, or for the updated JAX reference above.

### Historical matched measurements

RTX 3090, driver 610.43.03, Python 3.11.15, JAX/JAXLIB 0.9.0, native component
0.14.0 / ABI 14. Fields, coefficients, recurrence arithmetic, and CPML storage
are FP32. The native build uses precise math (`BEAMZ_CUDA_FAST_MATH=OFF`). Both
backends execute on the same GPU; JAX is not a CPU reference.

| Workload | GPU JAX median | CUDA median | JAX / CUDA |
| --- | ---: | ---: | ---: |
| FDTD, CPML + heterogeneous guide + source + DFT, 64×128×192, 512 steps | 188.35 ms | 102.01 ms | 1.85× |
| FDTD, same features, 128×256×384, 500 steps | 1496.79 ms | 725.80 ms | 2.06× |
| FDTD, medium grid with conductivity 120 S/m | 187.93 ms | 133.80 ms | 1.40× |
| FDTD, bare large grid with PEC boundaries | 898.60 ms | 529.54 ms | 1.70× |
| Bend inverse objective, 100 nm | 141.79 ms | 32.27 ms | 4.39× |
| Bend inverse objective **and full gradient**, 100 nm | 545.62 ms | 261.62 ms | 2.09× |
| Bend inverse objective, 50 nm | 597.75 ms | 258.14 ms | 2.32× |
| Bend inverse objective **and full gradient**, 50 nm | 3096.07 ms | 1661.61 ms | 1.86× |

The inverse problem is the [waveguide-bend notebook](../../examples/notebooks/cuda_autodiff_waveguide_bend.ipynb):
22×50×50 material cells / 1299 steps / 900 parameters at 100 nm, and
44×100×100 / 2597 steps / 3600 parameters at 50 nm. Both backends use the same
guided-mode objective, source profiles, material mapping, monitor interpolation,
CPML, and checkpoint interval 32. Full means differentiating every discrete
timestep and the objective, including CPML state, rather than inserting a
manually constructed adjoint source.

The comparison includes improvements to JAX source injection and continuous DFT
accumulation, which also accelerate the JAX reference. Against the *original*
CUDA implementation, gradient medians fell from 1633.41 to 261.62 ms at 100 nm,
and from 13355.96 to 1661.61 ms at 50 nm. Original JAX gradient medians were
747.74 and 3660.60 ms; those older JAX times are not used in the final speedup
column. Original measurements are retained separately as `baseline_inverse_*`.

All final [raw records](cuda_autodiff_results/) contain individual samples,
compile times, compiler-reported temporary storage, device/software identity,
native binary and runtime/kernel source checksums, and bootstrap 95% intervals
for ratios of medians. For gradient evaluation those intervals are approximately
2.085–2.086× at 100 nm and 1.857–1.867× at 50 nm. They describe variation within
this run, not uncertainty across machines, driver versions, or workloads.

## What was optimized

The initial reverse path spent more time moving and initializing arrays than
executing Yee transposes. A diagnostic 100 nm trace showed roughly 90,000 device
copies, including many tiny copies, plus 23,000 initialization operations. It
also differentiated fixed recurrence coefficients and retained primal fields
whose local Jacobians were independent of their values.

The changes address that overhead and the remaining native kernels:

- Symbolic-zero masks omit fixed-parameter gradients and their output buffers.
  Residual pruning saves only the fields and CPML memories actually needed by
  active coefficient derivatives. Lossless decay is explicitly constant.
- Source-field transposes gather adjacent cotangents with one writer per output.
  Full material-grid derivatives use direct stores; only broadcast reductions
  need atomics or initialization. Scalar and y/z profile reductions aggregate
  within a warp.
- Each phase uses one fused, spatially tiled adjoint kernel for all three
  components. Common field-only and electric-material VJPs have specialized
  kernels that eliminate unused coefficient arithmetic.
- Eligible forward timesteps use the existing native graph for Yee, sources,
  CPML, and vector DFTs. Backward linearization substitutes the already computed
  post-source fields, then applies the verified phase VJPs and JAX source/DFT
  transposes. Compiled HLO contains no streamed forward phase call in this
  backward replay. Recorder and unsupported monitor plans retain the phase path.
- Differentiable scans keep each monitor's DFT state as separate leaves, allowing
  unused monitor transposes to disappear. Continuous DFT updates avoid unnecessary
  conditional execution. Slab sources expose their identity field derivative
  through a scatter-add. The native forward program still evaluates every
  configured monitor.
- The CPML queue also handles full dynamic material coefficients, and its strict
  interior avoids recurrence and boundary checks. Fixed-material standard FDTD
  retains its exact lossless codebook and bounded native multi-step graph.

The final diagnostic trace reduced device-copy counts to about 28,000 and their
summed GPU duration from about 122 to 38 ms. Redundant gradient initialization
launches disappeared. Native adjoint execution was about 43 ms. These traced
durations are diagnostic, not the warm timing samples in the
table. Copies, coefficient-gradient accumulation, source/monitor transposes,
and per-timestep orchestration still limit inverse throughput. Ordinary FDTD's
dominant cost remains the forward field/CPML updates.

A 32×4 FP32 queue trial was about 2.2% slower than the retained 64×4 layout on the
large realistic workload. Explicit XLA command-buffer flag experiments that
included `WHILE` also regressed performance. Neither is enabled. The smaller
inverse mesh's large gain comes chiefly from eliminating launch and AD overhead;
it should not be extrapolated directly to large grids.

## Correctness and memory

Initial-design gradients differ from GPU JAX by 1.34×10⁻⁶ and 1.15×10⁻⁶ in
relative L2 norm at 100 and 50 nm. Standard FDTD comparisons include every state
leaf: fields, CPML buffers, DFT data/weights, and clocks. The largest relative
leaf difference is 1.09×10⁻⁴, in accumulated monitor data; the bare-grid maximum
is 3.59×10⁻⁶. Finite differences independently check material derivatives.
The initial bend objective's analytic directional derivative is −6.2538×10⁻⁵;
central differences at steps 0.2, 0.1, and 0.05 each agree within 0.1%.

The saved optimized layout still has approximately 92.6% guided transmission
on the 100 nm optimization mesh. Its CUDA/JAX objective difference is
1.79×10⁻⁷. Gradient disagreement is 5.32×10⁻⁴ relative near this nearly stationary
solution, whose gradient norm is only 7.86×10⁻⁴. Its directional finite-difference
check is limited by FP32 cancellation: the analytic derivative is 2.63×10⁻⁶,
and the estimate at step 0.05 is 2.98×10⁻⁶. This is not a claim of the same
transmission on finer meshes; the notebook separately checks mesh, time-window,
binary-layout, and reciprocity behavior.

Compiler-reported gradient temporary storage drops from 171.21 to 114.15 MB
at 100 nm, and from 1941.48 to 1491.31 MB at 50 nm. Final JAX uses 158.69 and
1831.76 MB respectively. These are executable workspace estimates, not total
process peak VRAM. Checkpointing still stores full states at chunk boundaries.

CUDA supports first-order reverse mode on one GPU with FP32 3D fields. Native
forward-mode JVPs, higher derivatives, BF16 autodiff, coupled full-tensor updates,
and sharded CUDA autodiff remain unsupported. Rebuild the native component for
ABI 24 before running this API. Historical records below use ABI 14.

## Reproduction

Build the native component, then run from the repository root in a CUDA-enabled
Python environment. Disable XLA preallocation. Run one benchmark process at a
time; compilation, mode solving, placement, and three warmups are excluded from
timings. Each timed call is synchronized. Backend order alternates across seven
inverse samples or nine ordinary FDTD samples. Calls preserve their input state.
No global XLA flag override is needed.

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python scripts/benchmark_cuda_inverse_design.py --dx-nm 100 --samples 7 \
  --finite-difference --output results/inverse_100nm.json
python scripts/benchmark_cuda_inverse_design.py --dx-nm 50 --samples 7 \
  --output results/inverse_50nm.json
python scripts/benchmark_cuda_inverse_design.py --dx-nm 100 --samples 7 \
  --beta 16 --finite-difference \
  --parameters-npz tests/performance/cuda_autodiff_results/optimized_parameters.npz \
  --output results/optimized_design_100nm.json
python scripts/benchmark_cuda_fdtd.py --shape 128 256 384 --steps 500 \
  --samples 9 --standard-only --output results/fdtd_large.json
python scripts/benchmark_cuda_fdtd.py --shape 64 128 192 --steps 512 \
  --samples 9 --standard-only --output results/fdtd_medium.json
python scripts/benchmark_cuda_fdtd.py --shape 64 128 192 --steps 512 \
  --samples 9 --standard-only --conductivity 120 --output results/fdtd_conductive.json
python scripts/benchmark_cuda_fdtd.py --shape 128 256 384 --steps 500 \
  --samples 9 --standard-only --bare --output results/fdtd_bare_large.json
```

The ordinary realistic workload uses six-cell CPML, a heterogeneous guide, an
electric source, and a three-frequency DFT plane. Add `--profile results/profile`
to either runner for a JAX/CUPTI diagnostic trace. The inverse runner also writes
compiled HLO. Omit `--standard-only` for the ordinary workload's differentiable
field/DFT objective; it is a different objective from the guided-mode bend.

Correctness checks:

```bash
python scripts/generate_cuda_abi.py --check
python -m pytest tests/hardware/test_cuda_autodiff.py tests/hardware/test_cuda_backends.py
JAX_PLATFORMS=cpu python -m pytest tests/unit/test_execution_backend.py \
  tests/unit/test_cuda_runtime_contract.py tests/unit/test_cuda_abi_schema.py \
  tests/integration/test_differentiable_simulation.py \
  tests/integration/test_differentiable_mode_projection.py \
  tests/integration/test_compiled_engine.py
compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest \
  tests/hardware/test_cuda_autodiff.py -k 'native_timestep or trainable_broadcast'
compute-sanitizer --tool initcheck --error-exitcode 99 \
  python scripts/check_cuda_adjoint_memory.py --cudart /path/to/libcudart.so
compute-sanitizer --tool racecheck --error-exitcode 99 \
  python scripts/check_cuda_adjoint_memory.py --cudart /path/to/libcudart.so
```

The integration memcheck uses JAX 0.10.2, avoiding CUDA symbol-probe diagnostics
from the older runtime: 11 focused tests passed with zero errors. Initcheck and
racecheck each passed 72 cases with zero errors/hazards through the standalone
harness, which loads the same native binary without JAX. It leaves gradient
allocations uninitialized, reads every output element, and independently checks
diagonal field and CPML-memory columns. Cases cover both phases, all three
adjoint specializations, scalar/profile/grid coefficients, bare/CPML grids, and
PEC intersections. The harness mirrors the private Linux C++ layout and must be
updated when that ABI changes.

The historical CPU suite passed 103 checks. Its hardware matrix passed 55 with two Hopper
skips on this GPU; two additional direct phase-state checks also passed. Their
summaries and the final native binary digest are saved in
[`validation.json`](cuda_autodiff_results/validation.json).

Unrestricted JAX initcheck/racecheck runs stalled during instrumentation and
were stopped. Filtering initcheck to native kernels alone leaves JAX producers
untracked and can produce false uninitialized-copy reports; the standalone
harness avoids that problem. Performance records use JAX 0.9.0 throughout.
