# H100 regression checks — 2026-10-10

## Eight-GPU capacity follow-up (additional $20 budget)

The follow-up reached near-full physical GPU memory with **streamed CUDA**.
JAX completed large 3D and flat workloads, but its larger 3D trial failed in
XLA compilation on **both the branch and origin/main**. `cuda_mstep` rejects
multi-device sharding, so an eight-GPU distributed mstep capacity result is
not available. These results do not establish that all three backends can
saturate eight GPUs, or an unconditional no-regression sign-off.

All rows below use eight H100 SXM GPUs. Shapes are physical **(z, y, x)** cells.
Memory is the range of per-GPU **peak NVML usage**, including allocator pools
and native allocations; the last column is maximum live memory tracked by JAX.
These are different measures, not quantities to add together. No unrelated
allocations were used to fill memory.

| Backend / revision | Shape | Billion cells | Warm GCUPS | Peak GiB/GPU (min–max) | Live JAX peak GiB/GPU |
|---|---|---:|---:|---:|---:|
| cuda_streamed / branch | 6912×1024×1024 | 7.248 | 160.486 | 66.60–66.98 | 52.86 |
| cuda_streamed / main | 6912×1024×1024 | 7.248 | 160.785 | 66.60–66.98 | 52.86 |
| jax / branch | 4096×1024×1024 | 4.295 | 69.350 | 66.91–67.29 | 48.03 |
| cuda_streamed / branch | 40×10240×12288 | 5.033 | 88.025 | 66.65–67.03 | 47.27 |
| cuda_streamed / branch | 40×12288×13312 | 6.543 | 88.297 | 73.99–74.36 | 61.40 |
| jax / branch | 40×8192×12288 | 4.027 | 54.962 | 66.66–67.61 | 53.31 |
| cuda_streamed / branch | 13312×40×12288 | 6.543 | 98.278 | 73.99–74.36 | 61.40 |

The matched 7.25-billion-cell streamed comparison measured **160.486 GCUPS on
this branch and 160.785 on main: −0.19%**. Peak memory matched to the displayed
precision, and all 30 output leaves matched exactly at the inspected samples.
This is one fresh process per revision with three timed executions, not a
confidence interval or proof about every workload. Other capacity rows have
no matched main run at their exact shape and size.

The 6.54-billion-cell flat-Z streamed case reached **73.99–74.36 GiB on every
GPU**, up to **93.37%** of the 79.65 GiB reported per device. The flat-Y
case reached the same peak. One-second telemetry independently shows all
eight GPUs above **73.53 GiB simultaneously** in the final flat-Y run. Its live JAX peak
was 61.40 GiB/GPU; the remaining NVML use includes pool capacity and native
allocations. JAX reached 67.29 GiB/GPU in the successful 4.29-billion-cell 3D
case and 67.61 GiB/GPU in the 4.03-billion-cell flat case. Those are substantial
loads, but they are **not near-full physical memory results**. These trials
are observed successful sizes, not a binary search for the largest fitting grid.

### Validation and limitations

- The eight-GPU public continuation gates passed for streamed CUDA with Z
  sharding and JAX with X sharding: **2 tests passed in 63.88 seconds**. They
  compare 17+31 donated steps with an unsharded 48-step JAX reference, including
  seeded fields, CPML and monitors. These are small full-state correctness gates.
- A separate eight-GPU small case compared ordinary serial placement with the
  optional parallel host placement used here: all **30 state leaves were
  exactly equal**, and peak NVML usage matched on every GPU.
- Large capacity trials inspect uniform samples from every addressable shard,
  additional field samples around the mode source, and the complete small
  monitor arrays. They check sampled finiteness and retain nonzero DFT signals.
  Full-volume finiteness/equality was not checked. A self-reference row is not
  an independent numerical validation; CSV labels distinguish it from paired
  comparisons.
- At **5120×1024×1024**, both branch and main JAX fail during
  `scan.lower(...).compile()` with
  `IndexError: absl::container_internal::raw_hash_map<>::at`, following an XLA
  rematerialization memory warning. Neither reaches execution. This is an
  observed compiler failure on both revisions, not a demonstrated GPU OOM or
  a diagnosed root cause. Reducing to 4096×1024×1024 completed on the branch.
- A live `cuda_mstep` request with eight devices fails with
  `CUDA multi-device sharding requires backend='cuda_streamed'`. Independent
  single-GPU replicas were not substituted for a distributed mstep simulation.
- Flat-X capacity and JAX flat-Y coverage were not completed within this budget.
  The queued JAX flat-Y process was stopped before producing a result after the final
  streamed result was retrieved; it is not recorded as an OOM or compiler failure.
  No claim is made that this follow-up closes every shape or backend combination.

### Method and reproducibility

The runtime snapshots, native builds, Python and GPU dependencies are the same
as in the initial study below. This node used driver 570.211.01 and all-to-all
NV18 links, with 700 W default power limits. There were eight visible GPUs and
actual nonreplicated field sharding, asserted by the worker. GPU preallocation
was disabled and the JAX allocator fraction was 0.90.

Every capacity workload retains all-face **12-cell float32 CPML**, one TE mode
source and one mode monitor with three frequencies eight cells downstream,
80 nm cells, binary waveguide material and 32 timesteps. Warm execution has
one warmup, at least 0.5 seconds of additional warmup, and three samples.
Inputs are borrowed and prior outputs are freed between samples. This does
not prove the maximum capacity of every public API output/cropping path.

The distinct `--transverse-anchor low` fixture moves the waveguide, source and
monitor together near the low transverse corner, keeping the mode apertures
in the non-PML interior. This lets coefficient preparation find the nonuniform material
without scanning half of a multi-billion-cell domain. Do not compare its
fields or DFTs with the earlier centered fixture. Thin-Y apertures are reduced
where needed to stay in the non-PML interior. Local full-state JAX/streamed/mstep checks
passed for both the low-anchor fixture and the thin-Y aperture.

The optional `--host-placement-workers 8` benchmark path evaluates the existing
bounded region readers concurrently across devices. Solver kernels and default
public API behavior are unchanged. Both matched revisions use this option.
Setup timings therefore describe this benchmark preparation path. The large
streamed branch case took 151.13 seconds to prepare, 2.31 seconds to compile,
and about 1.45 seconds per 32-step warm execution. The near-full flat-Z case
took 196.50 seconds to prepare and 2.39 seconds to compile. CPU preparation,
not CUDA compilation, dominated time to first result in these checks.

An initial serial-placement 7.25-billion-cell attempt was stopped after about
247 seconds while still preparing coefficients; it is retained as a cancelled
setup attempt, not a timing or memory-capacity result. Worker outputs now use
atomic replacement, and an independent collector saves immutable snapshots
locally throughout the run. This avoids the lost-final-export problem of the
initial study.

Example capacity worker invocation (with eight GPUs visible):

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/benchmark_cuda_revision.py --root /checkouts/branch \
  --backend cuda_streamed --shape 40 12288 13312 --steps 32 \
  --material binary --monitors 1 --frequencies 3 --monitor-offset-cells 8 \
  --transverse-anchor low --devices 8 --shard-axis auto --host-setup \
  --host-placement-workers 8 --samples 3 --warmups 1 --warmup-seconds .5 \
  --validation sampled --reference /tmp/capacity-ref --save-reference \
  --output /tmp/capacity.json
```

The pod was deleted. Rental-time cost including quoted storage is **$18.911**, within the additional $20 budget. The account-balance change observed after teardown was **$18.908**; billing can lag. An account-wide pod listing after teardown was empty.

The existing [CSV](cuda-h100-regression.csv) includes the follow-up rows under
`capacity_followup_*`. The existing [evidence packet](cuda-h100-regression.json.gz)
retains the initial results and adds a `capacity_followup` object with raw
worker outputs, failures, telemetry, source, dependency and controller records.

## Initial $30 study (retained historical results)

This study does **not yet support an unconditional no-regression sign-off**.
Larger completed branch/main comparisons are within 1% in warm throughput,
and all 62 retained branch trials match their main references exactly within
the validation scope below. Two small-case slowdowns need follow-up. At the end of this initial study, eight-GPU
near-full-memory coverage and its continuation gate remained incomplete;
the follow-up above updates that status.

The observed account charge after teardown is **$29.334 of the $30 budget**.
Elapsed rental time times the quoted rates gives $29.274, including setup and
idle time. All three pods were terminated and an account-wide pod listing was
empty afterward. Billing history can lag; these observations are not a settled
invoice.

The [summary CSV](cuda-h100-regression.csv) includes main, branch and release,
individual-round throughput, compilation, first result, memory and validation.
The [compressed evidence](cuda-h100-regression.json.gz) contains 139 retained
worker records, logs, hardware/dependency details, worker sources and exclusions.
Two additional public-API records are separate from that count.

## Workload and controls

The solver snapshots are:

- Branch: `27a3700354c0afefa5a213dfc233697a342fd23c`.
- Fetched `origin/main`: `3847a523f5a908d3fa88e77350b22d885735415f`.
- Latest release at study start, `v0.5.3`: `736b23778544a8b5effaccb2003fb2dbcfad664e`.
- Benchmark harness: `8372dd27`; optional transverse-offset fixture: `25451a17`.

Every timed case uses a binary dielectric waveguide, isotropic 80 nm cells,
float32 fields **and CPML**, twelve CPML cells on every face, one TE ModeSource,
and one ModeMonitor with three frequencies eight cells downstream. Conductivity
is zero. Shapes are physical **(z, y, x)** cells. These are short, complete solver
workloads; they do not establish convergence of a long optical experiment.

The shape matrix uses 128 timesteps, two warmups, at least another 0.5 seconds
of warm execution, and seven timed executions. The one-GPU matrix has two fresh
processes per revision/configuration, reversing their order in round two.
Two GPUs completed two rounds for small/cube and one matched round for flat-z;
an additional release flat-z repetition is retained separately. Eight GPUs
completed one round for small/cube/flat-z. There are no confidence intervals
based on treating repetitions within a process as independent machines.

Capacity trials use 32 timesteps and three timed executions in one fresh process
per completed configuration. GCUPS means physical cells × full timesteps /
median execution seconds / 1e9. Execution includes boundaries, source injection,
DFTs and internal storage conversion. Inputs are borrowed, previous outputs are
released before the next execution, and GPU completion is synchronized.

Setup builds the common workload under a CPU device context and places solver
state on the requested GPU mesh. The worker asserts the device count and that
multi-GPU fields are actually partitioned. The largest-axis sharding policy is
unchanged. Setup, lowering/compilation, first execution and warm execution are
separate. First-result time excludes imports, initial driver setup and validation.
Driver/JIT caches were not erased between processes.

All pods use H100 SXM 80 GB, the same image, CPython 3.11.13, JAX/JAXlib 0.9.0,
and the frozen dependencies in the packet. Distinct native sources were built
with CUDA 12.8.93, SM90, Release and fast math disabled. Main/release CUDA sources
are identical; branch/main Rust sources are identical. Matching native builds
were reused across snapshots and pods. Every record contains extension and
harness hashes. GPU memory preallocation is disabled, the JAX allocation limit
is 90%, and inherited BEAMZ_CUDA overrides are cleared.

The pods are different hosts: drivers are 580.126.09 on one/two GPUs and
570.211.01 on eight. Multi-GPU topology reports NV18 links. Power limits are
700 W; clock/power settings were not changed. Revision comparisons use the same
pod. Cross-pod scaling also includes host, driver and topology differences.

## Existing backend performance

Numbers below are warm GCUPS. Main and branch have matching round counts in
this table. More GPUs help some larger cases and hurt small cases; two GPUs also
hurt the flat-z streamed case.

| Shape | GPUs | JAX main | JAX branch | Streamed main | Streamed branch |
|---|---:|---:|---:|---:|---:|
| 64×64×96 | 1 | 3.147 | 3.111 | 8.184 | 8.042 |
| 64×64×96 | 2 | 1.684 | 1.642 | 2.697 | 2.744 |
| 64×64×96 | 8 | 0.992 | 0.990 | 2.757 | 2.746 |
| 256×256×256 | 1 | 7.711 | 7.688 | 20.361 | 20.202 |
| 256×256×256 | 2 | 16.495 | 16.523 | 23.721 | 23.693 |
| 256×256×256 | 8 | 38.626 | 38.660 | 48.576 | 48.710 |
| 40×512×1024 | 1 | 11.555 | 11.560 | 19.488 | 19.387 |
| 40×512×1024 | 2 | 12.153 | 12.121 | 15.060 | 15.049 |
| 40×512×1024 | 8 | 27.219 | 27.308 | 45.992 | 45.848 |

The smallest one-GPU streamed case is **1.73% slower** in the pooled samples;
the two round medians are 1.95% and 1.57% slower. This is a regression candidate,
not a result to dismiss as noise. Small two-GPU JAX is 2.48% slower pooled, with
0.50–3.54% slowdown by round. Its larger between-round variation also needs a
controlled repeat. Small one-GPU JAX is 1.14% slower pooled. Larger retained
branch/main pairs range from −0.94% to +0.33%. These observations do not prove
that every untested workload is free of regressions.

Release throughput is broadly similar to main in the measured shape matrix;
all release rows remain in the CSV. Release numerical comparisons fail because
source normalization changed upstream between v0.5.3 and tested main. Commit
`d2e86ed6` removed the cell-center N/(N−1) aperture-area correction in
`beamz/lattice.py`. Main and branch agree on inputs, coefficients, sources and
outputs. Release source fingerprints differ even where its initial-state and
material-coefficient fingerprints agree. Release numeric mismatches are retained
and are not counted as passes or attributed to this PR. Release capacity limits
were not measured.

## Streamed versus multistep on one H100

The multistep column shows the fastest of the two tested requests: default
16×32 whole columns, or 8×16 tiles with 32-plane segments. This is a comparison
of those configurations, not an exhaustive H100 tuning search.

| Shape | Streamed | Best requested mstep | Request |
|---|---:|---:|---|
| 64×64×96 | 8.042 | 4.311 | 8×16 / segment 32 |
| 256×256×256 | 20.202 | 18.517 | 16×32 / whole column |
| 40×512×1024 | 19.387 | 13.467 | 16×32 / whole column |
| 512×40×1024 | 18.995 | 12.597 | 8×16 / segment 32 |
| 512×512×40 | 11.159 | 9.608 | streamed capacity fallback* |
| 512×768×768 | 25.524 | 27.923 | 16×32 / whole column |

\* For flat-x, the request has no eligible columns and selects streamed capacity
execution. Its lower memory and slower timing are not evidence of temporal
stepping. The actual selection is recorded in the JSON/CSV.

On the 302-million-cell case, default mstep is **9.4% faster than branch streamed**
(8.7% faster than main streamed). This gain comes from one fresh process and three
warm samples. Small and flat cases still favor streamed. Mstep remains single-GPU;
there is no two/eight-GPU mstep result or silent sharding fallback in this study.

Cold results matter: the 302-million-cell, 32-step branch case takes 39.97 s with
JAX, 50.08 s with streamed and 52.37 s with mstep to produce its first result.
Compilation alone is 2.98 / 1.31 / 2.06 s respectively. Preparation dominates.
Those single-process first-result observations do not justify a universal
amortization threshold. Keep H100 automatic selection conservative.

## Memory and capacity

NVML is sampled every 50 ms from before imports through timing, before result
validation. It includes native allocations and reserved JAX pool memory. JAX
peak live allocation is recorded separately. Sampling can miss brief peaks;
neither statistic is an exact maximum-fit search or just the size of field arrays.

| GPUs | Global shape | Backend | Main / branch GCUPS | Maximum NVML GiB/GPU | Branch peak live JAX GiB/GPU |
|---:|---|---|---:|---:|---:|
| 1 | 512×768×768 | streamed | 25.684 / 25.524 | 33.02 | 22.77 |
| 1 | 1024×1024×1024 | streamed | 21.651 / 21.583 | 65.02 | 53.34 |
| 2 | 1536×1024×1024 | streamed | 39.826 / 39.770 | 66.36 | 47.09 |

Main and branch reserve the same GPU memory in these matched large cases. The
largest completed one/two-GPU cases use about 82–83% of reported VRAM. Main's
larger two-GPU 2048×1024×1024 streamed attempt runs out of memory during execution,
requesting another 4.01 GiB. Main JAX also runs out of memory at 1536×1024×1024 on
two GPUs, requesting 18.26 GiB. Branch versions of these failed baselines were
not run; they are unverified, not passes.

At 512×768×768, mstep raises peak live JAX allocation from **22.77 to 24.12 GiB**
(about 6%), while both reserve **33.02 GiB**. Geometry changes that tradeoff:
flat-x's capacity fallback uses less memory. Mstep's largest fit was not measured.
Borrowed-state measurements do not establish the maximum capacity of a donated
`Simulation.run` workflow. Large trials measure the prepared solver scan;
public result construction and cropping were checked only in the small
`advance` comparison, not at capacity.

One small branch streamed trial peaks at 1.53 GiB in NVML, versus 1.08 GiB in its
repeat and both main trials; live JAX peaks agree. This isolated transient is
retained and unresolved, rather than hidden by reporting only the best repeat.

## Validation, exclusions and unfinished work

The 127 retained shape trials compare every final state leaf. The 12 retained
capacity trials compare deterministic samples of up to 2,048 elements per
addressable shard and the complete small leaves. Finiteness has the same scope;
large field samples can miss the localized pulse. A nonzero DFT monitor confirms
injection/observation, but sampled equality is not whole-volume validation.
The existing tolerance is unchanged: absolute error <=
`max(3e-6, 1e-6 * reference_leaf_peak) + 3e-5 * abs(reference)`; integers are exact.
All 62 retained branch records are bit-exact against the corresponding main
reference within that scope. Mstep uses the main streamed reference. Matched
branch/main state, coefficient and source fingerprints also agree.

Both two-GPU streamed continuation tests (x/z sharding, auto memory policy,
seeded fields, 17+31 steps versus unsharded JAX) pass. The eight-GPU z-sharded
continuation test started, but its outcome was not retained before budget shutdown;
it is not counted as passed. A separate small public `advance` comparison on one
H100 produces identical full-state fingerprints and the same 1.087 GiB peak memory.
Its three call timings are still declining, so they are not used as a stable
public-API speed claim.

Eight-GPU small, cube and flat-z branch/main comparisons are complete. Main also
completed the centered 1024³ large case at 139.876 GCUPS and 18.97 GiB/GPU.
The branch's terminal output showed 139.722 GCUPS (−0.11%), but live-log export
failed when telemetry changed during tar creation. Budget shutdown then removed
the pod before its final raw JSON could be recovered. This avoidable collection
failure is recorded explicitly: that branch timing is terminal-only evidence,
excluded from the retained/validated trial count, and its final memory/numeric
record is unavailable.

The centered 6144×1024×1024 eight-GPU case hit the ten-minute limit during
preparation, before timed GPU execution. A diagnostic stack on the smaller case
found full-volume material symmetry checks for the centered mode source. A
second realistic fixture shifts the waveguide/source/monitor together by one
cell. It avoids that symmetry check, but a second full-volume bottleneck remains
in uniform material-coefficient detection (`_elide_uniform_grid`). That attempted
capacity run was stopped during preparation to reserve budget for the matched
billion-cell timing. No eight-GPU near-full-memory claim is supported.

The optional offset fixture was subsequently checked locally on RTX3090 with
JAX, streamed and mstep; full-state comparisons pass the existing tolerance.
These local checks are stored separately and are not H100 performance data.

Before a no-regression sign-off: repeat the small-case slowdowns under controlled
CPU placement, resolve the transient memory observation, finish eight-GPU
capacity/continuation checks, and test JAX and release capacity on all counts.
The source/material preparation scans are a concrete time-to-result opportunity;
any optimization should preserve the existing symmetry/uniformity criteria.
Retain per-trial artifacts atomically and synchronize completed results before
resource teardown; a live telemetry file must not invalidate a results export.

No solver, default backend policy, precision or public API was changed during
this study. Only benchmark options and evidence were added. Nothing was pushed.

## Reproduce one matched pair

Use the recorded dependency lock, native build flags and snapshot hashes. Run
the versioned common worker in fresh processes, changing only `--root` and
`--output` between main and branch. GPU visibility must match `--devices`.
For example, with the SM90 extensions built into both checkouts:

```bash
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/benchmark_cuda_revision.py --root /checkouts/main \
  --shape 256 256 256 --steps 128 --backend cuda_streamed --material binary \
  --monitors 1 --monitor-offset-cells 8 --host-setup --devices 1 \
  --samples 7 --warmups 2 --warmup-seconds .5 --validation full \
  --reference /tmp/beamz-h100-reference --save-reference --output /tmp/main.json
# Repeat with --root /checkouts/branch and --output /tmp/branch.json,
# using the same reference and omitting --save-reference.
```

The newer `--transverse-offset-cells 1` option creates the distinct offset fixture;
never compare it against a centered reference. The default remains zero.
