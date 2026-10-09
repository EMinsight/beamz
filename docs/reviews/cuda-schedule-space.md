# CUDA tile, layout and startup study

This study optimizes time to the result: preparation, first execution and all
requested timesteps. Warm throughput alone is insufficient. The CUDA engine
retains streamed and column multistep schedules, with shared numerical operations,
source injection, boundaries and observation code.

## What changed

- Four column tile shapes: 16x32, 8x32, 8x16 and 16x16, written as y by x.
- Whole columns or segments of 32, 64, 128 or 256 planes. Segment boundaries read
  overlapping frozen inputs and write only their owned outputs.
- All three right-handed cyclic storage orders, using the existing transformation
  of fields, materials, boundaries, sources and monitor indices.
- Dependency plans are executable input buffers rather than embedded constants.
  A bounded host cache reuses unchanged plans across launch choices and horizons.
- Monitor dependency construction uses vector operations instead of inserting
  individual points in Python. Reference comparisons produce identical plans.
- Material preparation is shared across schedules and launch settings. Distinct
  execution settings retain distinct execution keys.
- Progress runs pass the known number of calls to each compiled chunk into the
  cost calculation. Ordinary single calls still assume one execution by default.

These are variations of the same CUDA implementation. They do not introduce
separate copies of the physics. One- and two-timestep algorithms are tested;
deeper temporal blocking is not part of this study.

## Method and limits

The search space contains 72 configurations: 12 existing streamed choices and
60 mstep combinations (four tiles, five segment lengths, three storage orders).
The normal sweep screens 38 representative configurations. A promising rotated
order is refined across all 20 tile/segment combinations. Large-domain and
finalist startup measurements use a fresh process for every configuration.

Shape cases range from 32x48x64 to 256x512x512, including cubes, narrow domains,
thin domains in each direction and odd extents. Workload variations cover all-face
CPML, one-wall CPML with five PEC faces, all PEC, CPML thicknesses 4/12/20,
binary/smooth/sparse-conductive/dense-conductive materials, mode and Gaussian sources, no monitors,
ordinary DFT monitoring, dense DFT monitoring and BF16 CPML recurrence. Precision
is fixed within each workload; BF16 results are not substituted for FP32 results.

The initial 64-step shared sweeps screen warm performance. They cannot establish
unbiased cold preparation costs: helper code and dependency plans become cached.
Finalists are measured independently at their actual horizons. Balanced warm
orders and repeated samples reduce order effects. Differences below 3% are
reported as ties; this margin is a policy, not a statistical confidence interval.

The cold objective is:

```
T = preparation + first-execution overhead + expected_runs * steps * seconds_per_step
```

A reused executable with reusable graph buffers excludes preparation and initial
capture. A new state or changed buffer addresses can require graph capture again.
Material/state setup and Python import are reported separately. They contribute to
complete first-use time but do not favor a kernel when setup is shared. CUDA C++
compilation builds the optional component ahead of use; per-program XLA compilation
is a different stage.

This is a finite measured search on one RTX3090, not proof of an optimum on every
shape, source footprint or GPU. Shape, source coverage, monitored points, coefficient
representation, boundary thickness, ownership and allocator headroom all matter.
Unsupported mstep features retain the established streamed/JAX capability paths.

The permeability-only pilot named `magnetic` did not exercise dense H coefficient
banks: the present coefficient compiler uses magnetic conductivity and MU0 for
that update. Its timings are excluded from the dense-coefficient conclusions and
final recommendations. The replacement `dense_conductive` case uses a spatial
conductivity gradient and checks that all H source coefficients are nonempty 3D
banks before timing. The ordinary lossy case also covers dense H coefficients.
Variable-permeability physics needs a separate review across the existing solvers;
this performance work does not change that numerical behavior.

## Startup improvement

Fresh-process medium-case medians at 256 steps, in seconds:

| Configuration | Build scan, including plan | Plan construction | Lower | XLA compile | Extra preparation after shared setup | First execution | Warm execution |
|---|---:|---:|---:|---:|---:|---:|---:|
| Streamed | 0.3935 | 0 | 0.0170 | 0.1341 | 0.8569 | 0.6043 | 0.5649 |
| Mstep, original tile / whole | 0.5267 | 0.1284 | 0.0168 | 0.1239 | 0.9803 | 0.5180 | 0.4755 |
| Mstep, 8x16 / segment 32 | 0.5237 | 0.1277 | 0.0175 | 0.1249 | 0.9762 | 0.5114 | 0.4713 |

Before vectorizing monitor planning, the corresponding original mstep plan took
0.4171 seconds and its extra preparation took 1.2727 seconds. Before passing plan
buffers as inputs, the earlier diagnostic measured 1.205 seconds of XLA compilation
for mstep, compared with 0.133 seconds for streamed. These are different revisions;
raw evidence preserves their identities. The native numerical algorithm is
unchanged by these preparation fixes.

The medium plan comparison, with two monitor planes, took 0.473 seconds using the
old Python loop and 0.133 seconds using vector operations. At the large size it
fell from 1.792 to 0.468 seconds. All five plan arrays match byte for byte. Those
isolated host timings support the algorithm change; the fresh GPU-process timings
above measure its effect on real preparation.

At 256 steps, the original mstep tile is about 19% faster during execution, but its
first result still arrives slightly later than streamed. The balance changes as
execution grows or preparation is reused. The final horizon measurements below
check that prediction rather than assuming a warm speedup always repays startup.

## Reproduction and configuration selection

The benchmark supports both a representative search and the full combination
space. Run measurements sequentially on an otherwise idle GPU:

```console
python scripts/benchmark_cuda_schedule_space.py --cases medium narrow_x --steps 64 --output-dir .cache/screen
python scripts/benchmark_cuda_schedule_space.py --cases medium --exhaustive-layouts --steps 64 --output-dir .cache/exhaustive
python scripts/benchmark_cuda_schedule_space.py --cases medium --configs streamed streamed-capacity mstep-8x16-32 --steps 256 --isolated --output-dir .cache/fresh
python scripts/analyze_cuda_schedule_space.py .cache/fresh --fresh-only --steps 256 --output .cache/recommendations.json
```

Each ranked configuration includes its backend and environment settings. The
analysis supports `--cached` for an existing executable and `--expected-runs N`
for repeated executions; `N` describes reuse, not an instruction to execute N
simulations. Ranking a different horizon is labeled as an extrapolation. Validate
close choices at the actual requested horizon. Keep the complete workload,
physical GPU, driver, power limit, implementation and numerical precision matched.

Do not run the full search on every short simulation: the search itself can cost
more than the useful work. Use measured profiles and cheap capability/geometry
checks first, then calibrate a small candidate set when sufficient future work can
repay it. Refresh profiles after implementation or hardware changes.

## Further opportunities

Smaller tiles reduce scratch per block and expose more blocks. They also repeat
more halo work. Segments increase available parallel work but repeat ring startup
and halo reads. A rotated layout changes both memory access and which columns
intersect sources. These effects explain why one tile or cell-count threshold
cannot select every case.

Current source exclusion covers an entire column, even when a source occupies
only part of it. Segment-specific eligibility could recover work outside the
source's z interval. It would require per-segment masks, dependency queues that
cover changes along z, and the same source-timing and complete-state tests. This
is a concrete next experiment; current measurements do not establish its benefit.

Deeper temporal blocking should follow evidence of a large remaining gain.
It expands halos, shared storage and register use, and may make small shapes
worse. Retaining one-step streamed execution as a candidate already lets the
shared engine handle those shapes without forcing the column algorithm to win.
