# Performance evidence

Performance tests record rasterization, tracing, compilation, warm execution,
memory, result extraction, and monitor overhead separately. Hard regression
gates belong on controlled hardware; shared runners should publish measurements
without making noisy pass/fail claims.

`benchmark_schema.py` defines the portable record and comparison policy. Every
record includes the BeamZ/JAX/Python versions, backend, processor or accelerator,
device count, precision, grid, timestep count, boundaries, sources, monitors,
compilation time, multiple warm samples, and peak memory. A comparison deliberately
returns an ungated result unless the caller declares that it ran on controlled
hardware. Both the compiled executable and the full public execution path are timed;
the record derives GCUPS from each median using
``prod(grid_dimensions) * timesteps`` cell updates.

`h100_workloads.py` owns the two canonical 3D profiles. `bare_3d` isolates the
field stencil. `realistic_3d` includes heterogeneous material, six-face CPML,
source injection, and a plane DFT monitor. Run either from the repository root:

```console
python scripts/benchmark_h100.py --workload realistic_3d \
  --backend jax \
  --output benchmarks/realistic-h100.json
```

Pass `--devices 4 --shard-axis auto` for a four-device JAX record. CUDA backends
currently require `--devices 1`. The runner derives device identity, count, and peak
memory only from devices that own the placed field state, so unrelated visible GPUs
do not change the record.

Use `compare_benchmarks()` for same-backend regression gates and
`compare_backend_speedup()` to compare JAX, streamed CUDA, and Hopper records for the
same physical workload and hardware.

`RTX3090.md` documents the controlled PR-versus-`origin/main` harness. Unlike the
H100 schema records, it deliberately checks out `origin/main` and emits a table,
raw JSON statistics, and a graph for the custom CUDA-kernel decision on one RTX
3090.

The default `(nz, ny, nx) = (128, 256, 384)` grid and 500 timesteps are part of
the workload identity. Regression records with different features, backend,
hardware, device count, shape, or timestep count are deliberately not comparable.
See `H100.md` for measurement protocol and the pre-harness observations that
motivate the CUDA work.
