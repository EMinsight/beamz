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
unknown. An estimate is not an allocator reservation.

## How to choose a schedule

```python
program = simulation.compile(
    num_steps=256,
    backend="cuda",
    schedule="auto",
    expected_runs=32,
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
is shared. Layout calibration explicitly measures the streamed schedule.

Automatic mstep selection currently requires all of the following:

- A single RTX3090 and the optional mstep FFI target.
- Supported unsharded, canonical, isotropic float32 3D fields and diagonal
  materials, with packed E and scalar H coefficients.
- Default kernel and precision settings, apart from the graph cache toggle.
- At least 256 steps per invocation and 16,777,216–67,108,864 physical cells.
- At least 164 active column tiles, a depth of 64 and 60% eligible core coverage.
- The measured boundary families: all six CPML faces at thickness 12, only the
  right CPML face at thickness 12 with five PEC faces, or all six PEC faces.
- At most two vector DFT monitors with at most three frequencies each, and a
  combined monitor point count no larger than 2% of the physical cell count.
- Known sufficient memory and predicted execution savings greater than 1.25
  times the additional mstep compilation cost.

The cost model interpolates warm throughput and additional compilation time
between the two measured sizes in the original report. It estimates savings
over the requested steps and expected reuse. These are predictions, not a
claim that every admitted workload has been timed. Cases outside that envelope
stay streamed. Explicit mstep remains available on other supported GPUs and
for its broader feature set, including dense/lossy materials and unequal CPML.

This keeps the first execution conservative while letting repeated workloads
use the column schedule. A fixed cell-count threshold would miss compilation
cost, narrow column geometry, monitor work and memory requirements.

## Measured selector check

Fresh-process RTX3090 runs used 128×256×512 cells, 256 steps, all-face CPML12,
binary packed material, one ModeSource, two vector DFT monitors with three
frequencies, three warmups and five warm samples. Both schedules returned
complete state without donation. The allocator fraction was 0.6.

| Reuse hint | Selected schedule | Warm throughput | Reason |
| --- | --- | ---: | --- |
| 1 | streamed | 7.587 GCUPS | Extra compilation does not repay itself |
| 16 | streamed | 7.576 GCUPS | Predicted savings still below the margin |
| 32 | mstep | 9.065 GCUPS | Predicted savings exceed compilation and margin |

The 32-run hint gives about 19.5% higher warm throughput than the one-run
selection. It does not make the first execution faster: mstep requires more
compilation. For this geometry, the model predicts 2.910 seconds saved across
32 executions versus 1.538 seconds of additional compilation. The selected
mstep workspace estimate is 1,691,551,636 bytes. A reuse hint is workload-specific;
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

Local validation on CUDA 13.3 and RTX3090 passed:

- 153 distinct host contract cases across backend selection, memory, runtime,
  mstep planning, schedule choice, layout tuning, calibration control and ABI.
- 32 CPU integration and automatic termination cases, using `JAX_PLATFORMS=cpu`.
- 41 mstep GPU cases, including unified API parity, automatic dispatch, odd
  tails, complete state, DFT clocks, continuation and unequal six-face CPML.
- 20 existing streamed CUDA GPU regressions covering the feature envelope,
  full-state parity, workspace/tails, source graphs and DFT continuation.
- Native build, Python lint/format, generated ABI consistency and diff checks.

The H100 workflow now collects explicit schedule comparisons for small,
medium, large and narrow domains in fresh processes with reversed order across
two rounds. H100 remains outside the automatic model until actual measurements
support a rule. The modified CI workflow and CUDA 12.4 build have not been run
locally. No H100 performance result is claimed.

The next useful performance work is to measure more run lengths, source and
monitor densities, CPML thicknesses and GPU models. Those results can extend the
selector without creating another backend. Any OpenCL implementation should
share the schedule contract and correctness cases; it remains a separate
portability proposal rather than part of this local CUDA change.
