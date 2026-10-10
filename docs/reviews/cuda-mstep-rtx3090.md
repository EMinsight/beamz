# CUDA multistep: design and RTX3090 benchmarks

The revised `cuda_mstep` beats the Futhark temporal reference in nine of ten
matched RTX3090 cases and reaches practical parity in the remaining case.
The largest one-wall case is 0.24% behind Futhark, smaller than the observed
variation between rounds. The nine wins range from 4% to 27%.

This remains a scheduler inside the existing CUDA component, sharing material
compilation, numerical primitives, CPML, source injection and DFT operations.
It requires no Futhark installation. `auto` and established CUDA are unchanged.
The algorithm follows [PR #308](https://github.com/beamzorg/beamz/pull/308) by
Pepijn de Vos.

## Matched measurements

Both checkouts import the same simulation builder: binary dielectric waveguide,
float32 fields, isotropic 80-nm spacing, ModeSource, two field DFT planes at
three frequencies, 256 full timesteps, and 12-cell CPML where applicable.
Shapes are physical `(z,y,x)` cells. PEC cases have no CPML.

Every process uses three warmups and five timed executions. Two rounds reverse
backend order. All GPU work is serialized. Timing includes sources, boundaries
and DFTs; setup, compilation and first execution are recorded separately.
Input state is not donated, and output lifetimes match between implementations.
All three backends use the same 0.6 JAX allocator fraction with preallocation
disabled, leaving room for Futhark's separate CUDA arena. No clock or power
settings were changed.

The table divides physical cell count times timesteps by the median of ten
execution times. GCUPS counts full cell timesteps. Per-round samples, GPU
telemetry, source/library fingerprints, validation logs and selected core profiles
are preserved in [the evidence JSON](cuda-mstep-rtx3090.json.gz).

| Boundary | Shape | Established CUDA | CUDA multistep | Futhark temporal | Multistep / Futhark |
|---|---|---:|---:|---:|---:|
| all CPML | 64×96×128 | 4.711 | 3.487 | 2.876 | 1.213× |
| all CPML | 96×160×256 | 6.754 | 6.726 | 5.291 | 1.271× |
| all CPML | 128×256×512 | 7.607 | 9.067 | 7.292 | 1.243× |
| all CPML | 256×512×512 | 7.900 | 10.619 | 9.420 | 1.127× |
| all CPML | 512×256×64 | 6.435 | 3.895 | 3.218 | 1.210× |
| +x CPML | 64×96×128 | 4.900 | 3.803 | 3.279 | 1.160× |
| +x CPML | 128×256×512 | 7.263 | 11.272 | 10.064 | 1.120× |
| +x CPML | 256×512×512 | 6.811 | 11.875 | 11.904 | 0.998× |
| all PEC | 128×256×512 | 8.258 | 12.121 | 10.233 | 1.184× |
| all PEC | 256×512×512 | 8.117 | 12.608 | 12.129 | 1.040× |

For the large one-wall case, per-round medians are 11.809 / 11.909 GCUPS for
CUDA multistep and 11.892 / 11.918 for Futhark. This is a match within 1%,
rather than evidence of a CUDA speed advantage in that case. The other wins
repeat in both rounds. These measurements compare the explicit two-step
Futhark schedule, not automatic selection among Futhark schedules.

## Retained optimizations

The core keeps Futhark's 16×32 whole-column tiles, 8×16 eligibility quadrants,
720-thread plane mapping, compact first-step edge storage and shared stage
rings. It hoists scalar H coefficients and invariant plane coordinates.

The common packed-E/scalar-H case now lowers launch metadata to a compact
kernel view. Linear field/code addresses advance by a plane stride instead of
repeating coordinate products inside the column loop. A two-plane E0 ring
prefetches plane `p+2` with CUDA pipeline copies after E1 finishes reading `p`,
overlapping that copy with H2/E2. Each thread waits for its copy, then the block
synchronizes before the next wave reads neighbouring lanes. This specialization
uses 61,392 shared bytes and 56 registers per thread on SM86.

On the large all-wall case, diagnostic mean core time drops from 13.108 ms per
pair in the initial port to 5.564 ms in the selected implementation. The
reference core takes 5.056 ms in the earlier Futhark profile. CUDA's faster edge
work gives the whole-simulation results above. These are separate instrumented
runs with 640 core invocations each; profile throughput is excluded from the
matched measurement table.

Dense coefficients retain the general column core. Both cores use the shared
Yee/material arithmetic; edge work continues to use established CUDA updates.
Odd tails, source phases, DFT windows and absolute clocks retain their existing
semantics. Register codebook caching, shared codebook caching, H prefetching and
an earlier combined copy wait were measured and discarded when they slowed
execution.

The original PR measured 9.394 GCUPS for the temporal implementation versus
8.581 for CUDA on the large all-wall case. Those measurements motivated the
port; the matched matrix above records the final implementation. Earlier
prototype journals and reference build tooling have been removed.

## Choosing between CUDA schedules

Keep `cuda_mstep` explicit while extending the selection envelope. In this
workload family, established CUDA wins on the small and narrow domains; the
roughly four-million-cell case is tied. Multistep wins from the tested
16-million-cell domains upward, by 19–74% depending on the boundary layout.
The narrow eight-million-cell case demonstrates why cell count alone cannot
choose a schedule. Eligible column count, core coverage, edge work, source and
monitor coverage, coefficient representation and GPU occupancy all matter.

First-use costs also matter. At 256×512×512, median setup + compilation + first
execution takes the following seconds in fresh processes:

| Boundary | Established CUDA | CUDA multistep | Futhark |
|---|---:|---:|---:|
| all CPML | 12.07 | 15.66 | 12.66 |
| +x CPML | 11.54 | 13.13 | 11.48 |
| all PEC | 9.00 | 10.85 | 9.27 |

Long runs and executable reuse can amortize multistep's extra compilation.
These are fresh processes on a benchmarked machine, with existing driver/JIT
caches; they do not measure an empty-cache installation. Compact scratch also
does not guarantee lower total memory use. Native executable/live allocation
statistics and Futhark's total GPU memory snapshots are recorded separately.

The next unification should expose one CUDA backend with an explicit schedule
override and a selector based on measured costs. These ten cases establish the
performance result and useful crossover examples; they do not justify a
universal cell-count threshold. Future OpenCL work should share the schedule
contract and correctness tests; it is outside this CUDA implementation.

## Validation and reproduction

The selected implementation passes 36 multistep hardware cases, 20 existing
CUDA regression cases and 42 planner/ABI/runtime host cases. Comparisons cover
complete field/CPML/DFT state against CUDA and JAX, one-wall CPML, larger
material codebooks, dense/lossy coefficients, source phases, continuation,
donation, BF16 recurrence, windowed monitors, odd tails and native chunks.
ABI generation, Ruff and whitespace checks pass.

Memcheck reports zero errors for symmetric two-step and asymmetric odd-step
CPML. Racecheck reports zero hazards/errors/warnings for the packed short-column
core with ring wraparound. Its filter explicitly selects `MstepPackedColumnCore`.
`--report-api-errors no` suppresses JAX's recoverable constant-symbol probes;
memory and race instrumentation remain enabled. Exact commands/logs are in the
evidence JSON.

Measurements use RTX3090 24 GB, driver 610.43.03, CUDA 13.3.73, Release SM86,
JAX 0.9.0 and the existing 370-W power limit. CUDA 12.4 compilation and other
GPU architectures remain untested locally. Performance claims here apply to
the measured binary-material/ModeSource cases; correctness coverage is broader.

The historical Futhark library was the compiled PR checkout `55287d41`. Its temporal source
matches reviewed head `17ff1fa3` after normalizing the compiler annotation
spelling. The source and library fingerprints remain in the evidence for provenance.
This branch retains no Futhark runtime or reference benchmark driver; the
current profiling uses the shared schedule-space runner:

```console
python scripts/benchmark_cuda_schedule_space.py --cases small crossover medium large narrow_x --configs streamed mstep-16x32-whole --steps 256 --samples 5 --rounds 2 --isolated --output-dir .cache/mstep-comparison
python scripts/benchmark_cuda_schedule_space.py --cases right large_right pec large_pec --configs streamed mstep-16x32-whole --steps 256 --samples 5 --rounds 2 --isolated --output-dir .cache/mstep-boundaries
```

These commands measure the current native schedules. The historical Futhark
reference and its original method/fingerprints remain in the compressed evidence;
its implementation and experiment-only profiler are not needed in the PR.
The [current startup study](cuda-schedule-space.md) describes the later preparation
improvements and the explicit automatic selector.
