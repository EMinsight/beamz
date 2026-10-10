# CUDA tile, layout and startup study

This study optimizes time to the result: preparation, first execution and all
requested timesteps. Warm throughput alone is insufficient. The CUDA engine
retains streamed and column multistep schedules, with shared numerical operations,
source injection, boundaries and observation code.

The saved study contains 2,234 recorded trials across 26 included workloads,
214 fresh-process summaries and five owned-state checks. Files:

- [Measured 256-step choices](cuda-schedule-tables.md).
- [First-result plot](cuda-schedule-time-to-result.png).
- [Fresh-process timing CSV](cuda-schedule-space.csv).
- [Complete evidence, fingerprints and validation](cuda-schedule-space.json.gz).

Recorded trials include explicitly excluded historical pilots and failed choices;
they are preserved for auditing and are not treated as performance wins. A separate
compatibility section in the same packet retains 40 current-main comparison
workers and 14 diagnostic workers from before the streamed address fix.

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
- Rotated monitor arenas move six component blocks per monitor instead of
  embedding a permutation index for every accumulated value. Ragged arenas,
  untouched gaps, missing components and caller ownership are preserved.
- Mstep and storage-order selection now both compare execution savings against
  preparation plus initial capture. PEC/one-wall mstep uses 8x16 / segment 32;
  ordinary CPML retains the original tile when the alternatives are close.

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
Material/source/monitor/state setup and Python import are reported separately.
The driver first prepares a baseline program. A subsequent baseline request can
hit that program cache, while alternatives repeat the same source/monitor
preparation. Ranking excludes that repeated `program_setup_s` for every choice;
otherwise the baseline receives an artificial advantage. The raw pipeline
timings remain in the evidence packet. Fresh owned-state/public API checks also
measure the ordinary execution path without constructing a comparison baseline.
Common setup contributes to complete first-use time but does not favor a kernel.
CUDA C++
compilation builds the optional component ahead of use; per-program XLA compilation
is a different stage.

This is a finite measured search on one RTX3090, not proof of an optimum on every
shape, source footprint or GPU. Shape, source coverage, monitored points, coefficient
representation, boundary thickness, ownership and allocator headroom all matter.
Unsupported mstep features retain the established streamed/JAX capability paths.

The historical permeability-only pilot named `magnetic` did not exercise dense H
coefficient banks and is excluded from recommendations. Current main separates
physical electric conductivity from magnetic sponge damping. The current
`dense_conductive` case therefore combines a spatial electric-conductivity
gradient with sponge boundaries and verifies dense H source coefficients before
timing. Earlier dense-conductive profiles use the old loss semantics; they remain
in the raw packet but are excluded from current recommendations.

## Startup improvement

Fresh-process medium-case medians at 256 steps, in seconds:

| Configuration | Build scan, including plan | Plan construction | Lower | XLA compile | Schedule preparation, excluding common program setup | First execution | Warm execution |
|---|---:|---:|---:|---:|---:|---:|---:|
| Streamed | 0.3935 | 0 | 0.0170 | 0.1341 | 0.5446 | 0.6043 | 0.5649 |
| Mstep, original tile / whole | 0.5267 | 0.1284 | 0.0168 | 0.1239 | 0.6674 | 0.5180 | 0.4755 |
| Mstep, 8x16 / segment 32 | 0.5237 | 0.1277 | 0.0175 | 0.1249 | 0.6662 | 0.5114 | 0.4713 |

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

The committed packet is replayable without a GPU:

```console
python -m scripts.export_cuda_schedule_study --archive docs/reviews/cuda-schedule-space.json.gz --compress-archive --output-dir .cache/replay
python scripts/analyze_cuda_schedule_space.py docs/reviews/cuda-schedule-space.json.gz --fresh-only --cases medium --steps 1024 --output .cache/ranking.json
```

Each ranked configuration includes its backend and environment settings. The
analysis supports `--cached` for an existing executable and `--expected-runs N`
for repeated executions; `N` describes reuse, not an instruction to execute N
simulations. Ranking a different horizon is labeled as an extrapolation. Validate
close choices at the actual requested horizon. Keep the complete workload,
physical GPU, driver, power limit, implementation and numerical precision matched.
The analyzer and exporter share one comparison identity: complete workload,
hardware UUID/driver/power limit, JAX, native binary, planner, implementation,
workload code, benchmark method and monitor layout. Workers are separated before
calculating medians. A staged study can therefore produce several rankings and
table rows for the same case. Every CSV row has a profile ID that resolves to
inspectable metadata in the packet's `profiles` mapping. Table comparisons stay
within that profile; a row with only one candidate is explicitly labeled and
cannot establish a win over a candidate from another profile. Plot lines connect
horizons only within the same profile. Missing fingerprints are not treated as
matching known fingerprints. Raw records retain each worker's identity.

Do not run the full search on every short simulation: the search itself can cost
more than the useful work. Use measured profiles and cheap capability/geometry
checks first, then calibrate a small candidate set when sufficient future work can
repay it. Refresh profiles after implementation or hardware changes.

## Findings and automatic policy

The original tile and preparation comparisons below precede the merge of current
main. The selector now uses fresh `main-anchors` profiles from the same workload
family, following the streamed speed fix described in the compatibility check.

The same tile does not win on every shape. Small and thin CPML domains often
favor streamed. Rotating streamed is particularly useful for narrow-x and thin-x
domains. Smaller mstep tiles recover much of the original whole-column loss on
those shapes, but the best streamed layout can still be faster. Mstep remains a
schedule of the same engine, rather than a replacement required to win every case.

The 8x16 / segment-32 tile improves PEC and one-wall CPML execution by about
9–12% over the original mstep tile at both tested sizes. Matched-profile medium
PEC finalists reach about 13.5 versus 8.3 GCUPS, and medium one-wall CPML about
12.4 versus 7.3. Later rotated streamed-only profiles reach approximately 9.1
and 7.6 respectively; those are separate measurements, not paired comparisons.
These boundary cases also repay startup at 256 steps.
Ordinary medium CPML reaches about 9.1 versus 7.6 for canonical streamed; its
first 256-step result is close, so a conservative policy keeps streamed. Longer
runs or known reuse favor mstep. Tile differences below 3% are treated as ties.

Compact monitor permutations reduce narrow-x XLA compilation from approximately
1.66 to 1.47 seconds and lowering from 0.12 to 0.065 seconds. Warm speed is
unchanged within noise. Rotation still has an extra startup cost of about 1.15
seconds at 512x256x64 and 1.88 seconds at 1024x256x64. It loses on first-result
time at 256 steps and wins at 4096. The 1024-step checks also retain canonical
streamed for the smaller narrow-x case.

The new selector, explicitly enabled with `schedule="auto"`, runs no candidate trials. Existing calls without that option retain main's streamed schedule and geometry-based layout selection. It uses the two measured sizes
per boundary family, geometry/core coverage, coefficient representation, monitor
load, GPU model, precision, graph caching and actual memory headroom. It admits
mstep only when predicted savings exceed the extra startup by 25%. Calls shorter
than 16 steps, or known total work below 256 steps, stay streamed without querying
the GPU or constructing a dependency plan. Progress chunks contribute their known
reuse count; unknown automatic-termination duration is not treated as known work.

The separate layout candidate generator retains the existing 25% queue-work
reduction requirement. Its new cost gate currently covers the measured narrow-x
family (z=512–1024, y=256, x=63–65), packed lossless materials, CPML12, modest slab
sources and at most two three-frequency DFT monitors. Unknown cost families retain
canonical storage. Both policies leave explicit schedules and layout overrides
available. The earlier geometry-only selection notes are historical evidence.

## Actual run lengths

These historical profiles precede the current-main merge and streamed speed fix.
They measure fresh processes at the stated run length, in seconds after
common setup. They include schedule preparation and first execution; they are
not projections from the 64-step screening. The mstep comparison uses 8x16 /
segment 32 for medium, PEC and one-wall cases, and 16x32 / segment 64 for large.
The narrow comparison rotates streamed to storage order 120.

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

† These historical timings come from different implementation profiles, or have
missing method/layout fingerprints. They remain useful individual observations,
but cannot establish a matched comparison or a precise crossover. All other
rows compare candidates with the same complete profile and requested horizon.
The derived table and plot now enforce that distinction automatically.

The two algorithms should remain available inside the shared engine. Short runs
need the cheap startup of canonical streamed. Large ordinary domains and the
tested boundary families repay mstep preparation; narrow domains can instead
repay a streamed rotation. An explicit override remains useful for a prepared
executable or a workload outside the measured automatic-policy families.

These are conservative policies fitted to this machine, not universal optimum
proofs. CPU speed, allocator behavior, driver/JAX changes, source placement and
buffer reuse can shift the crossover. Another GPU needs its own profile. A broad
search should happen once during profiling; ordinary short simulations should
not pay for dozens of trial compilations.


## Historical ownership and public API checks

Separate fresh-process checks advance the pulse for 32 steps before timing 256
steps. Each direct trial owns a copied state and donates it to the executable;
copying is outside execution timing. Three warm samples and one measured public
API call follow warmup. These checks confirm the ranking without relying on the
initial nearly empty fields or borrowed-state behavior.

| Case and schedule | Owned warm GCUPS | Public API seconds | Peak device allocations, GiB |
|---|---:|---:|---:|
| Medium streamed | 7.58 | 0.624 | 3.38 |
| Medium mstep 8x16 / 32 | 9.10 | 0.525 | 3.70 |
| Large streamed | 7.90 | 2.390 | 11.95 |
| Large mstep 16x32 / 64 | 10.88 | 1.801 | 12.86 |
| Narrow rotated streamed 120 | 6.42 | 0.393 | 2.04 |

The public API includes work outside the direct executable, so it is slower.
One measured public sample is a functional/performance cross-check, not a precise
latency distribution. The allocator peaks include setup and state conditioning;
they are not the kernel's scratch requirement. Mstep uses more memory, so the
existing workspace/headroom check remains necessary.

The historical replacement dense-conductive case reaches about 5.32 GCUPS with 16x16 /
segment 32 versus 5.14 with ordinary streamed. This modest gain fails to repay
startup at 256 steps: 1.438 versus 1.275 seconds after common setup. At 1024
steps, capacity streamed and mstep are within 1% of each other. Dense coefficients
therefore remain outside the automatic mstep cost model rather than inheriting
the much larger binary-material gain.
These timings precede the merge of current main's corrected magnetic-loss
semantics. They do not describe the current dense-H workload; its correctness
checks now use sponge damping to produce magnetic loss independently of physical
electric conductivity.

## Compatibility with current main

The branch includes main at `3847a523`. Existing calls keep streamed execution
and main's geometry-based layout selection. The new cost selector requires an
explicit `schedule="auto"`; existing parameter positions and defaults remain
available. Dispersion, Bloch fields, plane-wave sources and complex continuation
from main are retained.

A separate Release build of main uses the same CUDA 13.3 toolkit and SM86
architecture as this branch. Two fresh processes per checkout, in reversed
order, compare seven binary-material workloads with mode monitors. Initial
fields, coefficients and sources match exactly. Every returned state array from
the established default path matches main bit for bit.

| Default case | Steps | Main warm GCUPS | Branch warm GCUPS | Main first result, s | Branch first result, s |
|---|---:|---:|---:|---:|---:|
| Tiny 32x48x64 | 64 | 1.628 | 1.577 | 7.738 | 7.619 |
| Small 64x96x128 | 256 | 5.272 | 5.199 | 7.875 | 7.876 |
| Medium 128x256x512 | 256 | 8.727 | 8.661 | 8.880 | 8.895 |
| Large 256x512x512 | 256 | 9.383 | 9.318 | 11.011 | 11.031 |
| Narrow 512x256x64 | 256 | 8.275 | 8.236 | 8.738 | 8.738 |
| Medium, longer run | 1024 | 8.744 | 8.669 | 10.359 | 10.358 |
| Medium, all PEC | 256 | 8.829 | 8.827 | 5.380 | 5.331 |

First-result time includes material/source setup, preparation and the first
execution; module imports and result validation are outside timing. The usual
check uses two warmups and five samples per process. Tiny executions last only
a few milliseconds: a repeat with four warmups and fifteen samples puts the
branch 3.3% ahead, while 128 warmups and 31 samples put it 2.2% behind. These
short-run measurements do not support a precise zero-overhead claim. All samples
and fingerprints remain in the packet; other default warm differences are below
1.5%, and first-result differences are below 2%.

Opt-in automatic mstep passes the same complete-state tolerance checks. It
reaches 9.545 GCUPS and a 10.218-second first result for the medium 1024-step
case, versus main's 8.744 GCUPS and 10.359 seconds. For the medium PEC case it
reaches 14.931 GCUPS and 5.189 seconds, versus 8.829 GCUPS and 5.380 seconds.
These cases use mode monitors and are separate from the field-monitor profiles
used to fit the automatic cost models.

The comparison first exposed a roughly 10% streamed slowdown caused by compact
address checks entering ordinary field updates. A compile-time addressing choice
now removes those checks from streamed updates and existing temporal kernels;
only mstep edge updates opt into compact field addressing. The numerical
operations remain shared. Fresh field-monitor profiles refit the selector after
recovering streamed speed. The refreshed all-face, one-wall and PEC models each
use two domain sizes; the layout model uses both narrow-x sizes. Each candidate
and baseline has two fresh processes in reversed order, two warmups and three
samples. The current sponge dense-H case favors streamed (6.28 versus 5.34 GCUPS)
and stays outside automatic mstep promotion. Raw measurements, derived model
inputs and the older profiles are preserved separately in the evidence packet.

## Validation

After merging current main, the CPU regression suite passes 716 cases with two
skips. It covers existing compiled execution, dispersion, Bloch fields/monitors,
plane-wave sources and termination alongside the CUDA host policies. The final
policy/runtime checks pass 210 cases, and the refreshed model/analysis checks
pass 105. These scopes overlap and are not added into a unique-case total.

After the native addressing fix, all 192 selected GPU cases pass: 129 mstep,
14 capacity and 49 storage cases. They cover complete fields, CPML state, DFT
accumulation and clocks, continuation, odd/even lengths, tile/segment variants,
cyclic layouts, boundaries, dense coefficients, automatic dispatch and donation.

The baseline backend comparisons still have three numerical failures:

- `test_streamed_cuda_matches_jax_for_extended_feature_envelope[conductive]`
- `test_streamed_cuda_realistic_sources_and_cpml_match_jax[0-smooth]`
- `test_streamed_cuda_realistic_sources_and_cpml_match_jax[1-smooth]`

Each also fails on the isolated main checkout. Tolerances are unchanged; these
failures prevent claiming the entire existing CUDA suite is green. Two obsolete
lossy-material eligibility assertions were corrected for main's separation of
magnetic damping from physical conductivity, and those cases now pass.

The native Release component was built with CUDA 13.3 for SM86. Compute Sanitizer
memcheck reports zero errors and racecheck reports zero hazards/errors/warnings
for the selected segmented cyclic case after the addressing fix. Matching native
kernel filters restrict coverage to that case, rather than all 72 configurations.
Host plan comparisons cover all five dependency arrays. Lint, formatting,
generated ABI and whitespace checks pass.

Exporting the committed packet again produces byte-identical compressed JSON,
CSV, tables and PNG. Validation logs, per-worker hashes and complete raw samples
are retained in the packet. Historical validation keeps its original revision;
current-main checks are recorded separately. No other GPU or architecture was
measured.

## Review fixes

Automatic mstep selection now rejects unknown headroom or an impossible minimum
workspace before constructing geometry or dependency arrays. The complete plan
estimate is still checked against a fresh memory reading before selection. Tests
forbid geometry construction in those early rejection cases and exercise a
budget that becomes inadequate between the two readings. The estimate remains
a guard rather than a memory reservation.

Progress execution validates the caller's reuse hint before multiplying by the
known chunk count. Invalid values, including booleans, now fail consistently with
ordinary execution instead of becoming valid integers through multiplication.

The exporter uses the analyzer's comparison identity before computing medians,
tables or plot lines. Synthetic mixed-worker tests cover changed native binaries,
dispatch code, benchmark method and hardware. They also verify exclusion of
shared-process workers, separate table rows and disconnected plot series. The
recorded timings were not rerun or reassigned to the fix revision. These review
fixes affect host selection, validation and analysis. The later compile-time
addressing fix is documented separately above.
