# Recovering the CUDA multistep advantage

Futhark's temporal stepping advantage is real. The initial CUDA port's slower
results identified a porting gap, rather than a limitation of the algorithm.
The revised `cuda_mstep` retains the shared CUDA compiler, field-update
primitives, CPML, source and monitor operations. It remains an explicit
schedule; `auto` and established CUDA keep their existing policy.

The original PR benchmark reached 9.394 GCUPS for tiled Futhark versus
8.581 for CUDA at 256×512×512 cells. Rerunning the Futhark case gives
**9.420 GCUPS**. On a matched ModeSource workload, the first CUDA multistep
port reached 6.524 versus 7.918 for established CUDA. Its earlier Gaussian,
128-step sweep is preserved in the [initial report](cuda-mstep-rtx3090-2026-10-09.md)
and must not be treated as a measurement of the revised core.

## Profile and changes

CUPTI concurrent-kernel activity exposes the main gap. The initial CUDA core
takes 13.108 ms per two-step pair, while Futhark's core takes 5.056 ms.
The CUDA edge phases are already faster: their combined mean cost is
6.012 ms per pair, versus 8.027 ms for Futhark's four edge kernels.

The CUDA port missed three relevant parts of Futhark's winning core:

- One thread per input-plane lane: 720 threads for the 20×36 window.
- Staging E0 once into shared memory, reused by the H1 and E1 stages.
- Hoisting scalar H decay/source coefficients outside the column loop.

The first two changes reduce CUDA core time to 9.728 ms. Scalar H
specialization and direct addressing of validated canonical full field banks
reduce it further to **7.916 ms**. Dense H coefficients retain the general
path. Compact addressing remains in edge kernels. Scalar/packed/dense/lossy
material updates continue to call the same CUDA arithmetic primitives.

| Core revision | Threads/block | Shared bytes | Registers/thread | Mean ms/pair |
|---|---:|---:|---:|---:|
| Initial CUDA port | 256 | 44,112 | 48 | 13.108 |
| E0 window and plane mapping | 720 | 52,752 | 44 | 9.728 |
| Scalar H and canonical full-bank addressing | 720 | 52,752 | 44 | 7.916 |
| Futhark temporal reference | 720 | 52,752 | 76 | 5.056 |

These are separate instrumented runs of the same 67-million-cell case, each
with 640 core invocations: 128 pairs × (first execution + one warmup + three
timed samples). They are kernel durations, not whole-program throughput or
hardware-counter measurements. The CUDA core still has room to improve.
Futhark uses tiled, interleaved global field storage; measuring the benefit
and conversion costs of such storage is a remaining experiment.

## Matched workload and selection

All follow-up cases use float32 fields/CPML, an isotropic 80-nm grid, a binary
dielectric waveguide, ModeSource, two plane field DFT monitors at three
frequencies, 12-cell CPML and 256 full timesteps. Shapes are physical `(z,y,x)`.
The GPU is the same RTX3090 24 GB, driver 610.43.03, CUDA 13.3.73, SM86,
JAX 0.9.0 and existing 370-W power limit. Clocks/power settings are unchanged.

Every CUDA schedule runs in a fresh process, with first-use costs excluded
from throughput but recorded separately. Two rounds reverse backend order,
each with three warmups and five samples. Sources, boundaries and DFTs are
inside executable timing. Input state is not donated. GPU workloads and
sanitizer runs are serialized. Per-process GPU telemetry and source/binary
fingerprints are recorded in [the raw evidence](cuda-mstep-futhark-followup.json).

The table uses the median of ten warm timings per schedule/case. Round ranges
are the two per-process median throughputs, not confidence intervals.

| CPML | Shape | CUDA GCUPS | CUDA multistep GCUPS | Multistep / CUDA |
|---|---|---:|---:|---:|
| all | 64×96×128 | 4.740 | 2.972 | 0.63× |
| all | 128×256×512 | 7.587 | 7.780 | 1.03× |
| all | 256×512×512 | 7.894 | 8.982 | **1.14×** |
| all | 512×256×64 | 6.431 | 3.205 | 0.50× |
| right | 256×512×512 | 6.811 | 9.567 | **1.40×** |

The large all-wall result repeats in both rounds: CUDA 7.889–7.897, multistep
8.978–8.989 GCUPS. The revised CUDA multistep path is **13.8% faster** than
established CUDA and reaches 95.4% of the rerun Futhark throughput. It is
37.7% faster than the first matched CUDA multistep port. The 2.6% medium-case
advantage is too small and narrowly sampled to motivate automatic selection.
Small and narrow cases retain substantial established-CUDA advantages.

For the large all-wall case, median setup is 9.05 / 9.05 seconds,
compilation 0.79 / 4.99 seconds and first execution 2.22 / 1.97 seconds for
CUDA / multistep. A fresh 256-step run therefore takes about 12.1 / 16.0
seconds including setup and compilation. Longer runs and repeated executions
can amortize the extra compilation; the warm-throughput crossover alone is
insufficient for a selector.

Peak JAX live allocations are 10,444 / 10,912 MiB and executable temporaries
1,734 / 2,205 MiB. Multistep compact scratch is 663 MiB and plan buffers another
123 MiB, with 77.8% core coverage. Compact scratch does not imply lower total
memory use. Allocation counters describe live buffers separately from pools
and `nvidia-smi` memory. The new core optimizations leave the storage plan intact.

The one-wall win also repeats: CUDA 6.807–6.821, multistep 9.566–9.568
GCUPS. Established CUDA uses its single-bank schedule here, while multistep
retains its alternate bank and compact scratch. Peak live allocations are
8,187 / 10,254 MiB; the throughput gain has a larger memory cost on this case.

The matched Futhark one-wall rerun reaches **11.915 GCUPS**, above the revised
CUDA multistep's 9.567. CUDA multistep recovers an advantage over established
CUDA on both large boundary cases, but it has not surpassed the Futhark
reference. Its remaining gap is about 4.6% for all-wall CPML and 19.7% for
one-wall CPML. These measurements justify further core/layout and asymmetric
edge profiling, rather than assuming the two implementations are equally fast.

A large-domain win establishes that this scheduler can be useful. Choosing it
automatically needs a broader envelope: eligible core fraction, number of
column tiles, first/second edge work, source and monitor coverage, material
representation, run length, memory headroom and GPU architecture. Cell count
alone misses the narrow-domain case. Preparation, compilation and graph
creation also matter for short runs.

Keep `cuda_mstep` as an explicit override while collecting that envelope.
Then unify the public names under a CUDA schedule option with a measured
selection policy. Preserve the shared compiler and numerical operations as
the maintenance boundary. The [OpenCL proposal](opencl-backend-proposal.md)
remains a local issue draft; nothing was published or pushed.

## Validation and reproduction

After both core changes, all 29 multistep hardware cases pass against
established CUDA and JAX, including complete field/CPML/DFT state, clocks,
continuation, odd tails, scalar/packed/dense/lossy E, dense H, BF16 recurrence,
source phases, mode sources, interval/windowed monitors, donation and native
chunk boundaries. The selected planner/ABI/runtime host suite passes 42 tests.
The earlier 93-host/20-existing-GPU regression checks remain recorded in the
initial report. ABI generation and Ruff checks pass.

Memcheck reports zero errors for symmetric two-step and asymmetric odd-step
CPML. Racecheck reports zero hazards/errors/warnings for the short-column
core with shared-ring wraparound. Both were rerun after the final optimization.
The commands and the reason for suppressing JAX's recoverable API probes are
in the initial report. CUDA 12.4 compilation and H100 remain untested locally.

```console
python scripts/benchmark_cuda_mstep.py \
  --shapes 64x96x128 128x256x512 256x512x512 512x256x64 \
  --steps 256 --samples 5 --rounds 2 --source mode --boundary all \
  --output-dir .cache/mstep-recovered-all
python scripts/benchmark_cuda_mstep.py \
  --shapes 256x512x512 --steps 256 --samples 5 --rounds 2 \
  --source mode --boundary right --output-dir .cache/mstep-recovered-right
```

The Futhark reference is the previously built PR checkout `55287d41`; its
temporal kernel matches reviewed head `17ff1fa3` except for the compiler's
`intrablock_result` annotation spelling. Source/generated-code/library hashes
and raw timing samples are included in the evidence. It uses its existing
Futhark benchmark driver and FFI binding, so this is a matched simulation
workload rather than an identical executable or memory representation.

```console
BEAMZ_FUTHARK_TEMPORAL=2 python scripts/benchmark_futhark_jax.py \
  --case futhark:256x512x512 --steps 256 --samples 5 --warmups 2 --pml-edges all
BEAMZ_FUTHARK_TEMPORAL=2 python scripts/benchmark_futhark_jax.py \
  --case futhark:256x512x512 --steps 256 --samples 5 --warmups 2 --pml-edges right
```

For kernel attribution, build the optional activity-injection helper with the
CUDA 13.3 SDK. This is independent of the CUDA component build. The raw CSV
records are embedded in the evidence JSON. Geometry/register metadata describes
the last launch of each kernel name; the compared core launches share geometry.

```console
g++ -std=c++17 -shared -fPIC -O2 -I"$CUDA_HOME/include" \
  scripts/profile_cuda_kernels.cc -L"$CUDA_HOME/lib64" \
  -Wl,-rpath,"$CUDA_HOME/lib64" -lcupti -o /tmp/beamz-kernel-profile.so
CUDA_INJECTION64_PATH=/tmp/beamz-kernel-profile.so \
BEAMZ_KERNEL_PROFILE_OUT=/tmp/beamz-mstep-kernels.csv \
  python scripts/benchmark_cuda_realistic.py --backend cuda_mstep \
  --shape 256 512 512 --steps 256 --samples 3 --warmups 1 \
  --source mode --boundary all --output /tmp/beamz-mstep-profile.json
```

All implementation and evidence changes are committed locally on
`feat/cuda-mstep`. No push, PR modification or issue publication was performed.
