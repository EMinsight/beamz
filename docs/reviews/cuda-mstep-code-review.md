# Review: CUDA whole-column multistep versus streamed execution

This is the review before unification. The [follow-up](cuda-schedule-unification.md)
records the shared workspace/IO changes and conservative selector implemented
in response. The recommendations below describe the earlier snapshot.

Reviewed locally on 2026-10-09, after cleanup commit `1e3ebe79`. The selected
kernel implementation is `b1560508`; cleanup changed comments, benchmark tooling
and evidence organization. The [benchmark report](cuda-mstep-rtx3090.md) retains
the algorithm credit and historical comparison with PR #308.

Keep both explicit schedules for now. They already use one CUDA engine, and the
multistep implementation has a useful measured advantage. Unifying the public
backend names should follow shared memory planning and schedule selection.
No numerical defect was reproduced in this review. The findings below describe
concrete performance/contract gaps and maintenance risks, not failed parity.

## Findings, in priority order

1. **P2: fixed multistep workspace has no capacity check.**
   [`cuda_capacity_schedule`](../../beamz/simulation/memory.py#L60) returns false
   for every backend except `cuda_streamed`. Consequently a
   `BEAMZ_CUDA_MEMORY_POLICY=capacity` request does not constrain `cuda_mstep`,
   and the existing estimator omits its compact scratch and plan arrays.
   This follows the current explicit-schedule policy, but leaves no workspace
   check for the larger allocation footprint.
   [`run_mstep`](../../beamz/simulation/cuda/mstep.py#L179) always allocates the
   alternate six-field bank plus six compact intermediate arrays.
   On the recorded large +x-CPML case, peak JAX live allocation increases from
   8,187 to 10,254 MiB; on all-wall CPML it increases from 10,444 to 10,912 MiB.
   This can exhaust memory for a problem that fits the streamed schedule.
   Add a multistep workspace estimate and headroom check. An explicitly forced
   schedule should report insufficient headroom clearly; a future automatic
   selector can choose streamed execution. No out-of-memory run was induced.

2. **P2: single-step and edge-only requests pay for unused temporal machinery.**
   The unconditional allocation at
   [`mstep.py:181`](../../beamz/simulation/cuda/mstep.py#L181) also occurs for
   `nsteps=1`, although
   [`program.cu:942`](../../cuda/src/program.cu#L942) executes only ordinary
   in-place phases in that case. A zero-core plan still builds two full cell
   queues, compact slot metadata and a padded scratch bank, then executes the
   pair edge schedule without any temporal core savings. For physical shape
   `(3,4,5)`, the host planner requests 15,360 scratch bytes against 2,052 bytes
   for all six canonical fields, before the alternate bank and plan metadata.
   Reuse the ordinary shared CUDA program for single-step and zero-core cases,
   and report the effective schedule. This preserves the requested API while
   avoiding work that cannot benefit from tiling.

3. **P2 before automatic promotion: the performance gate does not measure mstep.**
   The [H100 job](../../.github/workflows/cuda.yml#L91) includes mstep parity,
   but runs only when explicitly enabled and outside pull-request events.
   Its [benchmark loop](../../.github/workflows/cuda.yml#L125) does not include
   `cuda_mstep`. The current evidence is RTX3090/CUDA 13.3; the CUDA 12.4 build
   and other GPU architectures have not been exercised locally. Add named
   mstep warm/first-use/memory comparisons to the supported-device gate before
   changing `auto`. Correctness on an architecture does not establish a
   favorable schedule crossover there.

4. **P3: scheduler orchestration and temporal stages remain duplicated.**
   [`run_mstep`](../../beamz/simulation/cuda/mstep.py#L158) repeats monitor
   placeholders, phase buffers, flags and output reconstruction from
   [`runtime.py`](../../beamz/simulation/cuda/runtime.py).
   [`BeamzLaunchMstep`](../../cuda/src/program.cu#L794) repeats source/observation
   ordering and graph-buffer key serialization. The general and packed mstep
   kernels also repeat stage logic found in
   [`temporal_pair.cuh`](../../cuda/src/temporal_pair.cuh#L23).
   Their numerical primitives are shared, but changes to clocks, observation
   features or workspace contracts still require several edits. Extract common
   IO lowering, buffer-key serialization and source/observation helpers first.
   Preserve specialized kernel staging until a refactor passes parity and
   throughput checks; merging all fast kernels into one general kernel is not
   required to share their contract.

## What is shared and what differs

| Concern | Established `cuda_streamed` | `cuda_mstep` |
|---|---|---|
| Compiler and arithmetic | Shared materials, Yee/FMA rules, CPML, PEC, sources, vector DFT, ABI validation and graph cache | Uses the same operations and component |
| Scheduling | Native multi-step graphs, ordinary/combined spatial updates, existing optional temporal kernels and capacity-aware choices | Fixed two-step whole-column core plus compact first/second edge queues |
| Temporal geometry | Existing bounded 3D tiles and coupling/publication paths where eligible | 16×32 transverse tiles, 8×16 eligibility quadrants, full z traversal, H1/E1/H2 rings |
| Workspace | Schedule-dependent in-place or alternate banks | Alternate canonical bank, compact first-step fields and explicit cell queues |
| Source handling | Existing fused or separate source operations, depending on schedule | Source columns and a two-cell halo are excluded from the core; shared injection visits both substeps |
| Observation | Native DFT for supported plans; other observations retain the shared execution path | Native vector DFT at both substeps; additional first-step samples are recomputed into scratch |
| Feature envelope | Broader grid, observation, storage and explicit sharding support | Unsharded isotropic uniform float32 3D, diagonal materials, packed slab sources and vector DFT |

The distinction is between scheduling and storage strategies. Streamed execution
already supports multi-step graphs and optional temporal tiling. Whole-column
tiling saves repeated field traffic but gives each eligible transverse tile a
long-running block. The packed fast core has 720 threads, 61,392 shared bytes
and 56 registers per thread on SM86. A narrow transverse plane offers fewer
independent columns, so large total cell count alone cannot ensure utilization.

The shared numeric helpers fix derivative rounding, field FMA order, material
lookup, CPML recurrence and PEC constraints. Full banks stay canonical; only
intermediate edge fields use compact addressing. Core eligibility proves that
owned results avoid CPML, sources and boundary dependencies. The spare compact
quadrant is safe only under that dependency proof; preserve it in planner tests.

## Measured implications

The final ten-case matrix shows 19–74% higher warm throughput on the tested
16-million-cell and larger workloads. On the narrow `(512,256,64)` all-wall
case, multistep reaches 3.895 GCUPS against streamed's 6.435. The roughly
four-million-cell case is tied, and the small cases favor streamed execution.
The historical temporal reference is beaten in nine cases and matched within
0.3% in the largest one-wall case; that last difference is within observed
between-round variation.

For the large all-wall case, fresh-process setup + compilation + first execution
is 15.66 seconds for mstep versus 12.07 for streamed despite the warm-throughput
win. The selector must account for invocation length and executable reuse.
These measurements cover binary dielectric material, ModeSource and two DFT
planes on one GPU, rather than every supported material/source/monitor family.

## Recommended next work

1. Share workspace estimates, capability checks and IO lowering. Keep numerical
   parity tests common across every CUDA schedule. Handle single-step/zero-core
   requests with the established program before allocating compact storage.
2. Extend plan diagnostics with active tile count, owned quadrants, z depth,
   edge/intermediate work, scratch and total workspace. Measure those inputs
   against warm time and first-use cost across GPU models, dense/lossy material,
   source coverage and monitor density. Avoid a universal cell-count threshold.
3. Publish observed first-step fields from the column rings where profitable.
   The existing temporal-pair publication mechanism offers a precedent. Today
   mstep recomputes those samples with edge kernels; dense monitoring can remove
   much of its traffic advantage. Preserve interpolation dependencies and both
   substep observations when adapting publication.
4. Reduce large queue metadata using structured face/column ranges plus sparse
   source/monitor exceptions. The large all-wall case has 663 MiB of compact
   scratch and 123 MiB of plan buffers. Source exclusions currently discard an
   entire column even for a small z-local patch; bounded column segments may
   restore parallelism, with extra halo/launch work to measure.
5. Expose one CUDA engine with `schedule="auto"|"streamed"|"mstep"` when the
   selector is validated. Retain current names as compatibility overrides.
   Filter candidates by feature support and memory before comparing costs.
   Keep the [OpenCL proposal](opencl-backend-proposal.md) as a local portability
   draft until shared planning and observation contracts are stable.

The first two steps should precede further kernel tuning. Improvements to queue
traffic, publication and shared planning can benefit the existing temporal
paths as well as whole-column stepping. Avoid cross-block H/E fusion without a
valid global synchronization or frozen-input dependency scheme.

## Review verification

Cleanup preserves all 60 raw warm-timing records and all ten aggregated cases
unchanged. The simplified CUDA-only driver completes four fresh-process cases
(two backend orders) on `(32,48,64)`, three steps and three samples. These smoke
runs verify orchestration and are not added to the performance matrix.

The 42 selected planner/ABI/runtime host tests, ABI generation check, Ruff and
whitespace checks pass. Earlier evidence retains 36 mstep hardware cases,
20 existing CUDA regressions, and zero-error memcheck/racecheck gates. This
review additionally compares complete fields, CPML and DFT state against both
streamed CUDA and JAX with unequal nonzero CPML thicknesses on all six faces
at 2, 5 and 17 steps. These three cases are retained in the hardware suite:

```console
python -m pytest tests/hardware/test_cuda_mstep.py -k unequal_six_face -q
```

CUDA implementation behavior is unchanged by cleanup. The identified workspace
and orchestration improvements are recommendations; this review does not
install a selector or change scheduler behavior.
