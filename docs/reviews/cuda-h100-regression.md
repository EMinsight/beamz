# H100 performance and capacity evidence — 2026-10-10

Streamed CUDA reached 6.54 billion cells and 73.99–74.36 GiB per GPU on eight
H100s. Mstep improved one large single-GPU case but remains single-GPU only.
Small-case slowdowns and incomplete coverage prevent an unconditional
no-regression sign-off.

All raw records, source/dependency fingerprints, logs, failures and telemetry
are preserved in [cuda-h100-regression.json.gz](cuda-h100-regression.json.gz).
The initial study retained 139 worker records (127 shape, 12 capacity) plus two
public-API records; `capacity_followup` adds nine completed records (two small,
seven large). Original commit identifiers describe pre-consolidation snapshots:

- Branch solver: `27a3700354c0afefa5a213dfc233697a342fd23c`.
- Main: `3847a523f5a908d3fa88e77350b22d885735415f`.
- Release v0.5.3: `736b23778544a8b5effaccb2003fb2dbcfad664e`.
- Initial harness: `8372dd27`; follow-up worker sources/hashes are in the packet.

These measurements predate the PR's later main merge and are not measurements
of the rewritten branch tip.

## Workloads and controls

Every case uses a binary dielectric waveguide, isotropic 80-nm cells, float32
fields and all-face 12-cell CPML, one TE mode source and one mode monitor with
three frequencies eight cells downstream. Shapes are physical (z, y, x).
GCUPS counts full timesteps including sources, boundaries, DFTs and storage
conversion. GPU completion is synchronized. Inputs are borrowed and prior
outputs freed; maximum capacity of public result construction/cropping was not
tested. These short runs do not establish optical convergence.

H100 SXM 80 GB, CPython 3.11.13, JAX/JAXlib 0.9.0, CUDA 12.8.93 Release SM90,
fast math off. GPU preallocation is disabled, allocator fraction 0.90.
Drivers: 580.126.09 on one/two GPUs, 570.211.01 on eight; NV18 links and 700-W
limits. Matched revisions share the same pod; cross-pod scaling includes host
and driver differences. Device counts and actual field partitioning are asserted.
Setup, XLA compilation, first result and warm execution are measured separately.
First result excludes imports, initial driver setup and validation; driver/JIT
caches were not erased.

Shape trials use 128 steps, two warmups plus at least 0.5 s, and seven samples.
One GPU has two fresh processes per configuration with reversed order;
two GPUs have two rounds for small/cube and one matched flat-z round;
eight GPUs have one round. Capacity trials use 32 steps and three samples.
Samples within one process do not constitute independent-machine confidence
intervals.

## Existing backend performance

Warm GCUPS, matching round counts:

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

Small streamed on one GPU is 1.73% slower pooled (1.95%/1.57% by round);
small JAX on two GPUs is 2.48% slower pooled (0.50–3.54% by round).
Small one-GPU JAX is 1.14% slower. These are unresolved regression candidates.
Larger matched pairs range from −0.94% to +0.33%.

Release throughput was broadly similar, but numerical comparisons failed:
upstream `d2e86ed6` changed mode-source aperture normalization after v0.5.3.
Main/branch input, coefficient, source and output fingerprints agree; release
source fingerprints differ. Release mismatches remain failures, not PR passes.

## Streamed versus mstep on one GPU

Best of two mstep requests (16×32/whole or 8×16/32); this is not exhaustive tuning:

| Shape | Streamed | Best requested mstep | Request |
|---|---:|---:|---|
| 64×64×96 | 8.042 | 4.311 | 8×16 / segment 32 |
| 256×256×256 | 20.202 | 18.517 | 16×32 / whole column |
| 40×512×1024 | 19.387 | 13.467 | 16×32 / whole column |
| 512×40×1024 | 18.995 | 12.597 | 8×16 / segment 32 |
| 512×512×40 | 11.159 | 9.608 | streamed capacity fallback* |
| 512×768×768 | 25.524 | 27.923 | 16×32 / whole column |

* Flat-x has no eligible columns and actually runs streamed capacity execution.
The 302M-cell mstep gain is 9.4% over branch streamed, from one fresh process
and three samples. Small/flat cases favor streamed. H100 automatic selection
therefore remains conservative; distributed mstep is explicitly rejected.

At 512×768×768 with 32 steps, first result took 39.97/50.08/52.37 s for
JAX/streamed/mstep; XLA compilation alone was 2.98/1.31/2.06 s. Preparation
dominated. Mstep live JAX peak rose from 22.77 to 24.12 GiB (~6%), while both
reserved 33.02 GiB in NVML. Mstep's largest fitting domain was not measured.

## Capacity and memory

NVML peaks are sampled every 50 ms before validation and include reserved pools
and native allocations; JAX live peaks are separate, not additive. Sampling can
miss short peaks. These are successful tested sizes, not maximum-fit searches.

| GPUs | Global shape | Backend | Main / branch GCUPS | Maximum NVML GiB/GPU | Branch peak live JAX GiB/GPU |
|---:|---|---|---:|---:|---:|
| 1 | 512×768×768 | streamed | 25.684 / 25.524 | 33.02 | 22.77 |
| 1 | 1024×1024×1024 | streamed | 21.651 / 21.583 | 65.02 | 53.34 |
| 2 | 1536×1024×1024 | streamed | 39.826 / 39.770 | 66.36 | 47.09 |

Main/branch memory peaks match in these trials. Larger two-GPU main attempts
failed: streamed at 2048×1024×1024 during execution (another 4.01 GiB requested),
JAX at 1536×1024×1024 (18.26 GiB requested). Branch counterparts were not run.
One small branch streamed trial had a 1.53-GiB NVML peak versus 1.08 GiB in its
repeat and main; live JAX peaks agreed. That transient remains unresolved.

Eight-GPU follow-up:

| Backend / revision | Shape | Billion cells | Warm GCUPS | Peak GiB/GPU (min–max) | Live JAX peak GiB/GPU |
|---|---|---:|---:|---:|---:|
| cuda_streamed / branch | 6912×1024×1024 | 7.248 | 160.486 | 66.60–66.98 | 52.86 |
| cuda_streamed / main | 6912×1024×1024 | 7.248 | 160.785 | 66.60–66.98 | 52.86 |
| jax / branch | 4096×1024×1024 | 4.295 | 69.350 | 66.91–67.29 | 48.03 |
| cuda_streamed / branch | 40×10240×12288 | 5.033 | 88.025 | 66.65–67.03 | 47.27 |
| cuda_streamed / branch | 40×12288×13312 | 6.543 | 88.297 | 73.99–74.36 | 61.40 |
| jax / branch | 40×8192×12288 | 4.027 | 54.962 | 66.66–67.61 | 53.31 |
| cuda_streamed / branch | 13312×40×12288 | 6.543 | 98.278 | 73.99–74.36 | 61.40 |

The paired 7.25B-cell streamed result is −0.19%, with equal displayed memory
peaks and exact equality at the inspected samples of all 30 state leaves.
Other follow-up capacity rows lack a matched main run at the exact size.
The flat-Z/flat-Y streamed peaks reach 93.37% of reported 79.65 GiB capacity;
independent one-second telemetry shows all eight GPUs above 73.53 GiB
simultaneously in flat-Y. JAX's successful sizes are substantial loads but do
not saturate physical memory.

Both revisions' JAX 5120×1024×1024 attempts failed during XLA compilation with
`IndexError: absl::container_internal::raw_hash_map<>::at` after a
rematerialization warning. Neither reached execution. The root cause is unknown;
this is not demonstrated GPU OOM. Flat-X capacity and JAX flat-Y remain untested;
the queued flat-Y trial was stopped before a result, not counted as a failure.

## Validation and exclusions

All 62 initial branch trials matched their main references exactly within their
validation scope. Shape trials compare complete final state. Capacity trials
sample up to 2,048 points per shard plus complete small leaves; follow-up trials
also sample the source neighborhood and check nonzero monitor signals. Large
full-volume equality/finiteness was not checked. Self-reference rows are not
independent parity checks. Floating tolerance remains
`max(3e-6, 1e-6 * reference_leaf_peak) + 3e-5 * abs(reference)`; integers are exact.

Two-GPU streamed x/z continuation passed. The follow-up completed eight-GPU
streamed-z and JAX-x continuation: 17+31 donated steps against 48-step unsharded
JAX, with complete seeded field/CPML/monitor comparisons. Small eight-GPU
parallel versus serial host preparation matched all 30 state leaves exactly
and had identical memory peaks. A small public one-GPU `advance` comparison
also matched full state and 1.087-GiB peak memory; its unstable timings are not
used as a speed claim.

The first study lost the final raw output for branch's centered 1024³ eight-GPU
run during collection. Its terminal-only 139.722 GCUPS is excluded from the
validated count. Centered/offset larger attempts timed out in CPU preparation,
not GPU execution. The follow-up uses `--transverse-anchor low` and, for thin Y,
smaller apertures to stay inside the non-PML interior. This is a distinct fixture;
never compare its fields with centered references. Local full-state parity for
the changed fixtures passed. The optional parallel host reader affects benchmark
setup only and is used for both matched revisions. The 7.25B branch case required
151.13 s preparation versus 2.31 s compilation. Atomic result writes now prevent
the earlier collection failure.

Observed rental charges were $29.334 of the initial $30 and $18.908 of the
additional $20 (rental-time estimates $29.274/$18.911). All pods were deleted;
post-teardown listings were empty. Billing observations are not settled invoices.

## Reproduce

Use the recorded dependencies, native builds and worker hashes. For a matched
pair, save main's reference, then run the same worker with branch `--root`, a
new output path and no `--save-reference`. GPU visibility must match `--devices`.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/benchmark_cuda_revision.py --root /checkouts/branch \
  --backend cuda_streamed --shape 40 12288 13312 --steps 32 \
  --material binary --monitors 1 --frequencies 3 --monitor-offset-cells 8 \
  --transverse-anchor low --devices 8 --shard-axis auto --host-setup \
  --host-placement-workers 8 --samples 3 --warmups 1 --warmup-seconds .5 \
  --validation sampled --reference /tmp/capacity-ref --save-reference \
  --output /tmp/capacity.json
```

For the original one-GPU shape matrix use one visible device, centered geometry,
128 steps, two warmups, seven samples, serial host placement and full validation.
Complete coverage still needs controlled small-case repeats, transient-memory
investigation, missing capacity shapes, and JAX/release capacity comparisons.
