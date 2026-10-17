# Uniform CPML constants: RTX 3090 regression check

Measured 2026-10-15 on a local RTX 3090 (24 GiB, driver 610.43.03), comparing
`origin/main` **80ef0182** with PR #341 **ce53138a**. Python 3.11.15,
JAX/jaxlib/CUDA plugin 0.9.0; float32 fields/materials, x64 disabled.
Both checkouts used the same matching raster binary and release SM86 CUDA
component (0.21.0 / ABI 21); their native sources are identical.

Across **18 workload/backend pairs in 36 fresh processes**, warm runtime changed
by **−0.57% to +0.60%**. Every final field, CPML value and monitor accumulator
was finite and **bit-identical**. The nine scalar-compaction tests also passed
on GPU. No material speed, preparation-time or host-memory regression was observed.

| Workload | Backend | Main → PR, seconds | Peak live GPU GiB, main → PR |
|---|---|---:|---:|
| 224³ sphere, diagonal | JAX | 0.7583 → 0.7586 | 2.000 → 1.521 |
| 224³ sphere, full tensor | JAX | 1.5502 → 1.5510 | 2.376 → 2.000 |
| 224³ sphere, diagonal | CUDA | 0.3516 → 0.3521 | 2.001 → 1.624 |
| 30 nm U-bend, diagonal | JAX | 7.4918 → 7.5366 | 0.852 → 0.653 |
| 30 nm U-bend, full tensor | JAX | 10.9698 → 10.9532 | 1.066 → 0.868 |
| 30 nm U-bend, diagonal | CUDA | 2.9077 → 2.9081 | 1.012 → 0.832 |

The [18-row CSV](curved-interface-smoothing/rtx3090.csv) includes volume smoothing,
small grids, conductive and sponge controls, cold setup/JIT times, host RSS,
and sampled process VRAM. `n` means cube side in cells, or spacing in nm for the
U-bend; runtime is per scan at the listed step count. Memory units are GiB.
Ordinary CPML cases used 16–32% less peak live GPU memory; conductive controls
saved 6–8%, while sponge controls retained their memory footprint.

## Method and scope

- Sphere: 96³ and 224³ cells, 100 nm spacing, dielectric sphere in oxide,
  Gaussian source, two-component DFT plane, 10-cell CPML, 256 steps.
  All three smoothing methods use JAX; volume/diagonal also use native CUDA.
- Controls: 128³, diagonal smoothing, sphere conductivity 50 S/m or sponge
  boundaries, both backends, 256 steps.
- U-bend: the tutorial's 30 nm builder (4.8 M cells), mode source and two mode
  monitors, 4096 steps. All methods use JAX; diagonal also uses CUDA. This is
  shorter than the tutorial's converged transmission run.
- One GPU worker at a time, main then PR. Three warmups, seven synchronized
  timings (five for U-bends), median reported. Each scan starts from the same
  state with donation disabled. Setup/JIT and detached modal postprocessing
  are excluded from warm timing. Persistent compilation caching is disabled.
- GPU preallocation is disabled. JAX `peak_bytes_in_use` measures peak live
  allocations across preparation/execution; NVML samples process VRAM every
  50 ms, including retained pools/context. Host RSS sampling stops before
  numerical validation. Cached pools can keep process VRAM unchanged despite
  fewer live allocations; brief process peaks can be missed.
- GPU clocks were unlocked. Repetitions share a process/executable; sub-percent
  differences are not distinguished from between-process/clock variation.
  Cold setup/JIT have one sample per process. This does not establish maximum
  capacity, multi-GPU behavior or independent smoothing accuracy. Ordinary
  single-GPU native full-tensor execution is unsupported and was excluded.

## Reproduction

Use a CUDA-enabled environment with `psutil` and `nvidia-ml-py`, build each
checkout's matching native extensions, and run from the candidate checkout:

```bash
python scripts/benchmark_cpml_constants.py \
  --baseline-root /path/to/main-at-80ef0182 \
  --output benchmarks/results/cpml-constants
```

The runner writes per-process samples, memory statistics, source revisions,
CUDA binary hashes and per-leaf parity hashes. Raw results and the original
measurement harness/environment are archived locally under ignored
`benchmarks/results/pr341-rtx3090-20261015/`; they are not included in a fresh
checkout. The committed script preserves the measured workloads and protocol.
