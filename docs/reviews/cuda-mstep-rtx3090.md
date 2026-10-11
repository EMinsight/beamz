# CUDA multistep: Futhark baseline on RTX3090

The CUDA port of [Pepijn de Vos's PR #308](https://github.com/beamzorg/beamz/pull/308)
won nine of ten matched cases and matched the largest one-wall case within 0.3%.
These are historical measurements of the explicit two-step schedules, not a
claim about every shape or today's default selector. The full samples,
fingerprints, telemetry and validation logs remain in
[cuda-mstep-rtx3090.json.gz](cuda-mstep-rtx3090.json.gz).

## Method and results

RTX3090 24 GB, driver 610.43.03, CUDA 13.3.73, Release SM86, JAX 0.9.0,
370-W power limit. Each workload has binary waveguide material, float32 fields,
80-nm cells, one mode source, two field DFT planes with three frequencies,
256 timesteps and 12-cell CPML where applicable. Shapes are physical (z, y, x).
Two fresh processes per backend reverse execution order; each uses three warmups
and five samples. Inputs are borrowed, output lifetimes match, GPU completion
is synchronized, and all backends use a 0.6 allocator fraction with preallocation
disabled. GCUPS includes boundaries, sources and DFTs but excludes setup.

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

The large one-wall result is a tie: CUDA round medians were 11.809 / 11.909
GCUPS and Futhark 11.892 / 11.918. Other wins repeated in both rounds.
Small and narrow shapes still favored streamed CUDA over mstep.

Median setup + compilation + first execution at 256×512×512, in seconds:

| Boundary | Established CUDA | CUDA multistep | Futhark |
|---|---:|---:|---:|
| all CPML | 12.07 | 15.66 | 12.66 |
| +x CPML | 11.54 | 13.13 | 11.48 |
| all PEC | 9.00 | 10.85 | 9.27 |

These first-use measurements retain existing driver/JIT caches. They predate
the later preparation improvements in the [schedule study](cuda-schedule-space.md).
Futhark used checkout `55287d41`; its temporal source matched `17ff1fa3` after
normalizing the compiler annotation. Its original large all-wall result,
9.394 GCUPS versus CUDA's 8.581, motivated the port.

## Ideas retained in CUDA

Whole-column 16×32 tiles use 8×16 ownership quadrants and shared plane rings.
Only boundary dependencies and DFT sample cells need first-step scratch.
Source columns and their halos stay on the established CUDA edge path.
The packed-E/scalar-H core hoists material metadata, advances linear addresses,
and prefetches two E input planes while computing the second timestep.
It uses 61,392 shared bytes and 56 registers per thread on SM86.
Dense materials use the general column core; both share Yee/CPML arithmetic,
source phases and monitor clocks with streamed CUDA.

An instrumented large all-wall core fell from 13.108 to 5.564 ms per pair;
the separate Futhark profile measured 5.056 ms. CUDA's cheaper edge work accounts
for the whole-simulation result. These profiles are excluded from timed trials.

The historical gates passed 36 mstep GPU cases, 20 established CUDA cases and
42 host cases, covering complete field/CPML/DFT state, odd tails, continuation,
donation, coefficient layouts and source phases. Memcheck found zero errors;
packed-core racecheck found zero hazards. Commands and logs are in the archive.

Current implementation, overrides and reproduction commands live in the
[CUDA README](../../cuda/README.md). Futhark code/tooling stays removed;
its ideas, credit and measurements remain. OpenCL is a future portability task
that should share this schedule and correctness contract.
