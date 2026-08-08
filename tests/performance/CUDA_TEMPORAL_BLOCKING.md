# CUDA temporal-blocking experiment log

This log records parity-correct experiments against the retained
`32 × 16 × 2` fused full-step kernel on an RTX 3090. Experimental kernels only
become defaults when repeated same-binary measurements beat that kernel.

## Dependency model

One BeamZ leapfrog step computes `H(t+1/2)` from forward differences of `E(t)`,
then `E(t+1)` from backward differences of the new H field. Therefore an exact
second step crosses every low and high tile face. A correct multi-step schedule
must do one of the following:

1. recompute an overlapping space-time halo;
2. synchronize resident neighboring blocks; or
3. preserve first-step tile fronts and schedule the dependent boundary work as
   a later wave.

The direct T=2 volumetric, asymmetric, and N.5D overlap designs were already
parity-correct but 21–45% slower than full-step fusion. A cooperative persistent
grid was also slower because each H/E phase required a device-wide barrier.
The remaining untested category is a swept schedule that writes compact fronts
and advances independent tile interiors before resolving their boundaries.

For a rectangular T=2 tile, a one-cell halo is not sufficient: the composed
H→E→H→E dependence reaches both sides of each spatial dimension. Increasing T
widens the live front again, so T=3/T=4 are only sensible after a T=2 schedule
amortizes its front traffic and synchronization.

## Locality and tile sweep

All timings below use a `64 × 96 × 128` uniform full-PEC grid, 160 timesteps,
five warmups, and 21 or more already-compiled samples in isolated processes.
The retained kernel measured 10.53–10.60 ms during this sweep.

| Experiment | Median | Outcome |
| --- | ---: | --- |
| Register core + warp x-shuffles + y/z shared faces | 11.18 ms | 6.2% slower |
| Stage the input E dependency box once in shared memory | 12.51 ms | 18.0% slower |
| `32 × 16 × 1` tile | 11.11 ms | 5% slower |
| `32 × 16 × 3` tile | 11.96 ms | 13% slower |
| `64 × 8 × 2` tile | 10.53 ms | statistically neutral |
| `64 × 4 × 4` tile | 12.06 ms | 14% slower |
| `16 × 16 × 4` tile | 12.97 ms | 22% slower |
| `32 × 8 × 4` tile | 12.00 ms | 13% slower |

The face-only kernel reduces shared storage but not occupancy: the retained
512-thread block already reaches three resident blocks per SM. Its additional
halo evaluation and shuffle instructions therefore lose. Staging E removes
redundant global loads, but its second barrier and roughly 50 KiB allocation
reduce residency to two blocks per SM. The tile sweep confirms that two z cells
balance halo duplication and exposed parallelism on GA102; doubling x is at
best neutral.

These results narrow the next experiment to swept front communication. Merely
changing the block geometry, adding another barrier, or increasing the serial z
depth does not create useful temporal reuse.
