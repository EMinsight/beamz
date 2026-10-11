# CUDA tile, layout and startup study

The [raw evidence](cuda-schedule-space.json.gz) retains 2,234 trials across
26 included workloads, 214 fresh-process summaries, five owned-state checks,
and a separate main-compatibility study. Excluded pilots and failures remain
marked in the archive. Its original commit identifiers describe the measured
snapshots before this PR's history consolidation.

The conclusion is shape-dependent: small/thin domains often favor streamed;
large eligible columns can repay mstep preparation; narrow-x domains can favor
a streamed storage rotation. Warm throughput alone cannot choose the fastest
first result. The [CUDA README](../../cuda/README.md) defines the supported API,
memory checks and conservative automatic-selection envelope.

## Method

The runner exposes 72 configurations: 12 streamed choices and 4 mstep tiles ×
5 segment lengths × 3 cyclic storage orders. The normal survey screens 38;
rotated finalists and startup measurements run in isolated processes.
Shapes span 32×48×64 to 256×512×512, including cubes, thin/narrow axes and odd
extents. Variations include CPML4/12/20, all-PEC and one-wall boundaries,
binary/smooth/conductive materials, mode/Gaussian sources, sparse/dense DFT
monitoring and BF16 recurrence. Precision stays fixed within a comparison.

Rank using:

```
T = schedule preparation + first-execution overhead
    + expected_runs * steps * seconds_per_step
```

Material/source/monitor/state setup and imports are recorded separately.
Repeated common setup is excluded from ranking so the prepared baseline's cache
hit cannot bias it. Cached ranking excludes preparation/capture; new state
buffers may require capture again. Shared sweeps screen warm performance only;
use fresh-process measurements at the actual horizon to claim a cold-start win.
Differences below 3% are treated as ties, not statistical confidence bounds.

The analyzer separates hardware UUID/driver/power, workload, JAX, native binary,
Python implementation, planner and benchmark fingerprints, including per-worker
identities in resumed sweeps. It excludes historical pilots marked in the packet.
The permeability-only pilot did not create dense H banks. Earlier conductive
profiles also used obsolete magnetic-loss semantics; the current dense-H fixture
uses sponge damping and verifies its coefficient shapes before timing.

## Preparation and geometry findings

Plans now use cached executable input buffers and vectorized monitor dependency
construction. Material preparation is shared across schedules. Rotated monitor
arenas move component blocks rather than storing a per-value permutation.
No numerical kernel is duplicated for these choices.

Historical medium-case fresh-process medians at 256 steps (seconds):

| Configuration | Build scan, including plan | Plan construction | Lower | XLA compile | Schedule preparation, excluding common program setup | First execution | Warm execution |
|---|---:|---:|---:|---:|---:|---:|---:|
| Streamed | 0.3935 | 0 | 0.0170 | 0.1341 | 0.5446 | 0.6043 | 0.5649 |
| Mstep, original tile / whole | 0.5267 | 0.1284 | 0.0168 | 0.1239 | 0.6674 | 0.5180 | 0.4755 |
| Mstep, 8x16 / segment 32 | 0.5237 | 0.1277 | 0.0175 | 0.1249 | 0.6662 | 0.5114 | 0.4713 |

The 8×16 / segment-32 variant improved PEC/one-wall mstep by about 9–12% at both
tested sizes. Ordinary CPML tile differences were small. Rotation reduced warm
narrow-x execution but added about 1.15–1.88 s startup at the two measured sizes.
Current sponge dense-H favored streamed (6.28 versus 5.34 GCUPS).

Historical measured time after common setup, including preparation and first
execution (seconds):

| Case | Steps | Canonical streamed | Compared schedule | Compared seconds |
|---|---:|---:|---|---:|
| Medium | 16 | 0.611 | Mstep | 0.733 |
| Medium† | 1024 | 2.866 | Mstep | 2.614 |
| Medium | 4096 | 9.763 | Mstep | 8.408 |
| Large | 16 | 0.971 | Mstep | 1.430 |
| Large† | 1024 | 9.610 | Mstep | 7.755 |
| Large | 4096 | 36.487 | Mstep | 27.202 |
| Medium PEC | 16 | 0.555 | Mstep | 0.669 |
| Medium PEC | 1024 | 2.615 | Mstep | 1.952 |
| Medium one-wall | 16 | 0.648 | Mstep | 0.715 |
| Medium one-wall | 1024 | 3.005 | Mstep | 2.079 |
| Narrow 512x256x64 | 256 | 1.288 | Rotated streamed | 2.306 |
| Narrow 512x256x64 | 1024 | 2.736 | Rotated streamed | 3.438 |
| Narrow 512x256x64† | 4096 | 8.378 | Rotated streamed | 7.418 |
| Narrow 1024x256x64 | 256 | 2.119 | Rotated streamed | 3.757 |
| Narrow 1024x256x64 | 4096 | 16.109 | Rotated streamed | 14.299 |

† These rows cross implementation profiles or lack method/layout fingerprints.
They are individual observations, not matched comparisons or precise crossover
estimates. Other rows match the full recorded profile and requested horizon.
The finite RTX3090 search does not establish a universal optimum.

## Compatibility with current main

This comparison used main `3847a523`, before the later main merge in this PR.
Defaults retained streamed execution and geometry-based layout selection;
`schedule="auto"` enabled the new cost model explicitly. Both checkouts used
CUDA 13.3 Release SM86, binary materials and mode monitors. Two fresh processes
per checkout reversed order. Initial state, coefficients, sources and every
returned default-path state leaf matched bit for bit.

| Default case | Steps | Main warm GCUPS | Branch warm GCUPS | Main first result, s | Branch first result, s |
|---|---:|---:|---:|---:|---:|
| Tiny 32x48x64 | 64 | 1.628 | 1.577 | 7.738 | 7.619 |
| Small 64x96x128 | 256 | 5.272 | 5.199 | 7.875 | 7.876 |
| Medium 128x256x512 | 256 | 8.727 | 8.661 | 8.880 | 8.895 |
| Large 256x512x512 | 256 | 9.383 | 9.318 | 11.011 | 11.031 |
| Narrow 512x256x64 | 256 | 8.275 | 8.236 | 8.738 | 8.738 |
| Medium, longer run | 1024 | 8.744 | 8.669 | 10.359 | 10.358 |
| Medium, all PEC | 256 | 8.829 | 8.827 | 5.380 | 5.331 |

First result includes setup/preparation/execution but excludes imports and
validation. Tiny timing repeats ranged from 3.3% faster to 2.2% slower for the
branch; this is not a precise zero-overhead claim. Other default warm differences
were below 1.5%, first-result differences below 2%.

Opt-in auto reached 9.545 GCUPS / 10.218 s first result for medium/1024 steps
(main: 8.744 / 10.359), and 14.931 / 5.189 for medium PEC (8.829 / 5.380).
These mode-monitor checks are separate from the field-monitor calibration.
The comparison caught a streamed slowdown from compact address checks; those
checks are now compiled only into mstep updates. The selector was refitted using
fresh `main-anchors`, retaining the old measurements as history.

Owned-state checks also compared donation and public API execution; large
streamed/mstep allocator peaks were 11.95/12.86 GiB. These include setup and are
not just scratch sizes. Full-state parity, segmented/cyclic layouts, odd tails,
CPML, donation, and memory/race instrumentation remain covered by the hardware
tests and archive. See the [H100 study](cuda-h100-regression.md) for other GPUs.

## Reproduce and rank

```console
python scripts/benchmark_cuda_schedule_space.py --cases small medium large narrow_x --configs streamed mstep-16x32-whole --isolated --steps 256 --samples 5 --rounds 2 --output-dir .cache/cuda-study
python -m scripts.analyze_cuda_schedule_space .cache/cuda-study --fresh-only --steps 256 --expected-runs 1 --output .cache/cuda-ranking.json
python -m scripts.analyze_cuda_schedule_space docs/reviews/cuda-schedule-space.json.gz --fresh-only --steps 1024 --output .cache/cuda-replay.json
```

Use `--exhaustive-layouts` to extend a survey to all 72 candidates; `--cached`
ranks reusable executables. Rankings label extrapolated horizons. Match the full
workload/hardware identity before changing automatic defaults.
