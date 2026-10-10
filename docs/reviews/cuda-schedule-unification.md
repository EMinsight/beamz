# One CUDA engine, two execution schedules

The CUDA schedules now share memory planning and source/monitor orchestration.
`backend="cuda"` can choose a schedule automatically or accept an explicit
choice. The `cuda_streamed` and `cuda_mstep` names still select their respective
schedules. All work remains local; nothing was pushed.

This follows the [backend review](cuda-mstep-code-review.md). The original
[RTX3090 comparison](cuda-mstep-rtx3090.md) retains the useful Futhark ideas and
reference timings. There is no Futhark runtime or benchmark backend in this
branch. The column kernels retain the implementation that reached nine wins
and one result within 0.3% of that reference across ten cases.

## What is shared now

Both schedules lower source groups, monitor buffers, complex phase workspace,
absolute monitor clocks and returned DFT state through the same Python helpers.
Native source injection and ordinary observation calls use common program
helpers. Graph buffer keys use the same serialization. The two schedules still
need different dependency plans and intermediate storage: those are part of
their algorithms.

The planner checks column geometry before building dependency queues. A
single-step invocation or a domain with no eligible column core uses the shared
in-place CUDA program. It allocates neither the mstep alternate bank nor compact
intermediate storage. This also applies to an explicit mstep request.

The shared memory estimate counts output state, possible alternate banks,
compact mstep intermediates, dependency arrays, DFT accumulators and retained
input ownership. Streamed estimates conservatively include possible graph
field banks, including temporal-pair storage. Capacity mode uses the existing
in-place schedule. Forced mstep rejects a known shortage before lowering its
workspace and rejects capacity-only policy when a pair workspace is needed.
Automatic selection falls back to streamed when the mstep estimate does not
fit the remaining allocator budget plus 64 MiB headroom, or when that budget is
unknown. Unknown headroom or an impossible minimum workspace now rejects the
candidate before constructing column geometry or a dependency plan. After
planning, selection checks the complete estimate against a fresh budget reading.
The minimum is only an early rejection check; it excludes plan and scratch bytes.
An estimate is not an allocator reservation.

## How to choose a schedule

```python
program = simulation.compile(
    num_steps=1024,
    backend="cuda",
    schedule="auto",
    expected_runs=1,
)
print(program.cuda_schedule_report())

# Explicit choices use the same engine.
run = simulation.advance(backend="cuda", schedule="mstep")
run = simulation.advance(backend="cuda", schedule="streamed")
```

`expected_runs` describes how often the caller expects to reuse that compiled
program. It defaults to one and never executes additional simulations. The
choice is frozen for each executable and donation policy. The report explains
the request, selection, reason and workspace; mstep candidates also report
column geometry and predicted costs. Reading a report runs no timing trial.
Schedule and reuse hints are execution-cache inputs, while material preparation
is shared. Progress runs automatically count reuse of each known compiled chunk.
Unknown automatic-termination duration does not count as promised future work.
Reuse hints must be positive integers; booleans are rejected consistently in
ordinary and progress execution. Layout calibration explicitly measures the
streamed schedule and remains opt-in.

Automatic mstep selection currently requires all of the following:

- A single RTX3090 and the optional mstep FFI target.
- Supported unsharded, canonical, isotropic float32 3D fields and diagonal
  materials, with packed E and scalar H coefficients.
- Default kernel and precision settings, with graph caching enabled and positive
  graph-cache capacity.
- At least 16 steps per invocation, at least 256 known total steps across expected
  reuse, and 16,777,216–67,108,864 physical cells.
- At least 164 active column tiles, a depth of 64 and 60% eligible core coverage.
- The measured boundary families: all six CPML faces at thickness 12, only the
  right CPML face at thickness 12 with five PEC faces, or all six PEC faces.
- At most two vector DFT monitors with at most three frequencies each, and a
  combined monitor point count no larger than 2% of the physical cell count.
- Known sufficient memory and predicted execution savings greater than 1.25
  times the additional mstep startup cost.

The cost model interpolates warm throughput and additional startup time
between the two measured sizes per boundary family in the
[current tile and startup study](cuda-schedule-space.md). Startup includes host
planning, lowering, XLA compilation and initial execution/capture overhead.
It estimates savings over the requested steps and expected reuse. These are
predictions, not a
claim that every admitted workload has been timed. Cases outside that envelope
stay streamed. Explicit mstep remains available on other supported GPUs and
for its broader feature set, including dense/lossy materials and unequal CPML.
The automatic tile is 8x16 / segment 32 for PEC and one-wall CPML, and the original
16x32 / whole-column tile for ordinary CPML, where differences were small.

This keeps the first execution conservative while letting repeated workloads
use the column schedule. A fixed cell-count threshold would miss compilation
cost, narrow column geometry, monitor work and memory requirements.

The shared engine also considers a rotated streamed layout. Geometry must predict
at least 25% less CPML queue work, and execution must repay additional startup
with a 25% margin. Its cost model covers the measured RTX3090 narrow-x family:
physical z=512–1024, y=256, x=63–65, lossless packed material, all-face CPML12,
small slab sources and at most two three-frequency DFT monitors. Calls need at
least 16 steps and 32 known total steps. Other shapes retain canonical storage;
explicit cyclic layout overrides remain available. It does not compile candidate
programs during default selection.

## Startup and measured results

CUDA C++ compilation builds the optional component ahead of use. The additional
per-simulation work was primarily host dependency planning and JAX/XLA compilation
of large embedded plan constants. Passing plans as input buffers reduced a
diagnostic mstep XLA compile from 1.205 to 0.124 seconds. Vectorized monitor
planning reduced isolated plan construction from 0.473 to 0.133 seconds at the
medium size, and from 1.792 to 0.468 seconds at the large size. A bounded host
cache retains at most two plans and 256 MiB; material preparation is shared.

Matched-profile fresh-process results illustrate why both schedules remain:

| Case | Steps | Streamed first result | Compared schedule | Compared first result |
| --- | ---: | ---: | --- | ---: |
| Medium, all-face CPML12 | 16 | 0.611 s | Mstep 8x16 / 32 | 0.733 s |
| Medium, all-face CPML12 | 4096 | 9.763 s | Mstep 8x16 / 32 | 8.408 s |
| Large, all-face CPML12 | 4096 | 36.487 s | Mstep 16x32 / 64 | 27.202 s |
| Medium, all PEC | 1024 | 2.615 s | Mstep 8x16 / 32 | 1.952 s |
| Narrow 512x256x64 | 1024 | 2.736 s | Rotated streamed 120 | 3.438 s |
| Narrow 1024x256x64 | 4096 | 16.109 s | Rotated streamed 120 | 14.299 s |

These times include schedule preparation and first execution after common setup.
They compare the tested variants, not a guarantee of the fastest possible
configuration. The automatic ordinary-CPML selector deliberately retains the
original tile when warm differences are small. Complete raw timings and
comparison fingerprints are in [the evidence packet](cuda-schedule-space.json).

## Historical selector check before startup improvements

The following measurements describe the earlier cost model. They explain the
original 32-run example; they do not describe the current crossover.

Fresh-process RTX3090 runs used 128×256×512 cells, 256 steps, all-face CPML12,
binary packed material, one ModeSource, two vector DFT monitors with three
frequencies, three warmups and five warm samples. Both schedules returned
complete state without donation. The allocator fraction was 0.6.

| Reuse hint | Selected schedule | Warm throughput | Reason |
| --- | --- | ---: | --- |
| 1 | streamed | 7.587 GCUPS | Extra compilation does not repay itself |
| 16 | streamed | 7.576 GCUPS | Predicted savings still below the margin |
| 32 | mstep | 9.065 GCUPS | Predicted savings exceed compilation and margin |

In that earlier experiment, the 32-run hint gave about 19.5% higher warm throughput
than the one-run selection. It did not make the first execution faster: mstep
required more compilation. For this geometry, the model predicted 2.910 seconds saved across
32 executions versus 1.538 seconds of additional compilation. The selected
mstep workspace estimate was 1,691,551,636 bytes. A reuse hint is workload-specific;
32 is an example, not a universal crossover.

The [raw evidence](cuda-schedule-unification.json) includes all three records,
samples, compilation timings, workspace summaries, source hashes and extension
hashes. The records identify the dirty working-tree source used for measurement
relative to `56141db6`. The later streamed bank accounting/report addition does
not change the execution kernels in these adequately provisioned cases.

## Direct intermediate publication was tested and discarded

The column kernel was changed experimentally to publish first-step H/E values
directly into existing compact scratch slots. Owned core values were then
removed from first-step edge queues. Both general and packed cores passed all
39 hardware cases available at that point, but total execution became slower:

| All-face CPML12 case | Existing kernel | Direct publication | Change |
| --- | ---: | ---: | ---: |
| 128×256×512 | 9.017 GCUPS | 8.446 GCUPS | −6.3% |
| 256×512×512 | 10.596 GCUPS | 10.087 GCUPS | −4.8% |

Less edge work did not offset the cost of extra publication in the column
kernel. The experiment was removed; the faster existing kernels remain. The
raw before/after records are retained alongside the selector evidence.

## Validation and remaining work

The initial unification validation on CUDA 13.3 and RTX3090 passed:

- 153 distinct host contract cases across backend selection, memory, runtime,
  mstep planning, schedule choice, layout tuning, calibration control and ABI.
- 32 CPU integration and automatic termination cases, using `JAX_PLATFORMS=cpu`.
- 41 mstep GPU cases, including unified API parity, automatic dispatch, odd
  tails, complete state, DFT clocks, continuation and unequal six-face CPML.
- 20 existing streamed CUDA GPU regressions covering the feature envelope,
  full-state parity, workspace/tails, source graphs and DFT continuation.
- Native build, Python lint/format, generated ABI consistency and diff checks.

The later tile/startup study passed all 129 mstep GPU cases and 197 unique CPU
cases, with selected memcheck/racecheck results reporting zero errors/hazards.
After the review fixes, 149 affected CPU cases and six focused GPU cases pass.
These counts overlap and are not added together. The review fixes change host
selection, public input validation and evidence analysis; native kernels are
unchanged. The current report and packet preserve commands and validation scope.

The H100 workflow now collects explicit schedule comparisons for small,
medium, large and narrow domains in fresh processes with reversed order across
two rounds. H100 remains outside the automatic model until actual measurements
support a rule. The modified CI workflow and CUDA 12.4 build have not been run
locally. No H100 performance result is claimed.

The current study records 2,198 trials across 26 included workloads and measures
actual horizons of 16, 256, 1024 and 4096 steps. The next useful work is to extend
matched profiles to other GPU models and workloads outside the admitted envelope,
then validate close decisions at their actual horizons. Those results can extend
the selector without creating another backend. Any OpenCL implementation should
share the schedule contract and correctness cases; it remains a separate
portability proposal rather than part of this local CUDA change.
