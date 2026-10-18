# Full-tensor workflows: validation and performance

This implements [issue #361](https://github.com/beamzorg/beamz/issues/361) for
lossless guided 3D dielectric modes on uniform and rectilinear Yee grids.
See the [workflow guide](../../../docs/tensor-modal-workflows.md) for supported
combinations, material compression, phasor conventions and limitations.

## Operator consistency

For a longitudinally invariant guide, the discrete temporal and normal derivative
symbols are `Omega = 2 sin(omega dt/2)/dt` and
`q = 2 sin(beta dx/2)/dx`. Cross-component constitutive terms also include the
normal averaging factor `cos(beta dx/2)`. With component-to-node interpolation
`A_j` and node-to-component interpolation `T_i`, the inverse constitutive operator
has diagonal blocks `B_ii = diag(epsilon_at_Ei^-1)_ii` and cross blocks
`B_ij = T_i diag(epsilon_at_nodes^-1)_ij A_j`.

The mode solver refines beta in `B C_H C_E e = (Omega/c)^2 e`, reconstructs H,
and checks both first-order discrete Maxwell equations. It uses the same
inverse tensors and physical grid metrics as propagation and TFSF injection.
Failed solves and unsupported material/support combinations raise errors.

## Validation

Independent checks cover dense tensor inversion, anisotropic plane waves,
Fourier reduction versus the production update, first-order Maxwell residuals,
rectilinear adjoint identities, both directions on all axes, CPML, reciprocal
complex transmission, locally diagonal ports in tensor domains, zero-cross-term
recovery, and complete result archives. Exact material compression is checked
against the original coefficient bit patterns, including signed zero and
one-ULP differences; CPU/GPU tests also exercise dynamic coefficients and
finite-difference gradients.

The independent [Meep/MPB reference generator](meep_reference.py) uses a 0.6 by
0.6 um core with epsilon `[[6,.2,.1],[.2,5,.3],[.1,.3,4]]` in air at 1.55 um.
The two retained MPB fixtures are test inputs, not benchmark logs. At 80 samples
per um, the fundamental index is 1.7430003 in a 4 um transverse supercell and
1.7430624 in a 2 um cell. Uniform and rectilinear BeamZ refinements approach that
reference monotonically; the finest tested relative error is below 0.2%.

### Source flux and modal power

Source calibration retains the production source-plane Poynting quadrature.
For each directional mode `u`, let `m = G u` be its measured field after normal
Yee interpolation. Normalize the complete E/H basis as
`b = m / sqrt(abs(<m,m>))`. A measured field `c m` then has coefficient
`a = c sqrt(abs(<m,m>))`, so `abs(a)^2` equals that mode's discrete flux.
The real scalar preserves the directional interpolation phases needed to reject
false backward waves. Rectilinear projection shares the flux monitor's clipped
aperture area weights. Coherent cross terms remain in `modal_flux`.

On October 20, 2026, 274 targeted tests passed: 194 CPU launch/projection/API
checks, 75 GPU tensor/rectilinear/CUDA checks, and five differentiable projection
and optimization parity checks. These include 81 native-wave projection cases,
36 independent launch-flux cases, and end-to-end assertions of **both** flux and
modal power for 1 W and 2.5 W sources. Partial-cell apertures and off-grid monitor
positions are covered. Off-grid modal watts must track flux at the same plane;
arbitrary subcell offsets need not have identical interpolated flux.

The full modal-sources and cosine-crossing notebooks ran on RTX 3090 JAX and
`cuda_streamed` at their original grids and durations. The full-tensor companion
also executed all 12 code cells on CPU JAX, including absolute-power and archive
checks. The versioned notebooks retain runnable source; executed copies remain
in the local benchmark results archive.

| Broadband straight guide at 1.55 um | JAX | CUDA |
| --- | ---: | ---: |
| Flux (W) | 1.000078118 | 1.000078125 |
| Fundamental modal power (W) | 1.000038146 | 1.000038153 |
| Unresolved flux after reconstruction (W) | 0.000040413 | 0.000040413 |

Within ±15% of the center frequency, broadband flux spans 1.000016–1.000177 W
and fundamental modal power spans 0.999974–1.000146 W. No response curve is clipped
or scaled to force unity. For each backend, all nine raw modal-notebook flux/field
exports and all ten crossing exports exactly match the retained pre-regression
baseline. Cross-backend differences remain in tiny backward powers (at most
1.03e-9 W) and crossing Ez (relative L2 3.21e-5); the latter is unchanged from
the baseline. These checks establish the exercised discrete contract, not exact
continuum power for every mesh, aperture, bandwidth or device.

## Performance

Measurements used RTX 3090 24 GiB, JAX 0.12.0/CUDA 13, one OpenBLAS/OMP thread,
CPU setup, disabled persistent JIT caching and GPU pool preallocation. Main was
`ba85c851bbff0a959a97843f9ad08a293b6bbe62`; each checkout used its matching native
extension (ABI 28 versus 30). The realistic cases have silicon/silica/air,
oblique sidewalls, all-face CPML, seven source profiles, three-mode monitors at
17 frequencies, flux/field monitors, and 250 fs propagation.

| Diagonal-smoothing case | Backend | Main ms/step | Updated ms/step | Speedup |
| --- | --- | ---: | ---: | ---: |
| Uniform straight guide, 200×100×75 | JAX | 0.8512 | 0.8317 | 1.023× |
| Uniform straight guide | CUDA | 0.7542 | 0.7476 | 1.009× |
| Graded junction, 180×88×56 | JAX | 0.7061 | 0.6869 | 1.028× |
| Graded junction | CUDA | 0.6595 | 0.6549 | 1.007× |

Three fresh processes per case gave median JIT times of 0.919–1.141 s, versus
0.923–1.152 s on main, without a consistent regression. Setup changed by at most
0.17 s across the four cases (about 9–11 s total). JAX active GPU peaks decreased
from 266.2 to 222.9 MB for the straight guide and 140.2 to 137.9 MB for the
junction. CUDA whole-workflow peaks were unchanged. Host setup/JIT memory and
NVML footprints did not show consistent savings; compression still needs dense
materials during setup. These are modest end-to-end gains, not a general claim
that coefficient compression saves the same fraction of total runtime/memory.

Separate full-tensor cases used one confined mixed-polarization mode. CUDA
propagation was 2.04× faster than JAX in the straight guide (0.4672 versus
0.9538 ms/step), and 3.08× in the junction (0.2582 versus 0.7950 ms/step).
These are one fresh process per case with five warmed repetitions. They are
backend comparisons of the new capability, not same-physics comparisons with
main, which lacks matching full-tensor modal/native-CUDA support.

Timings predate the final modal power normalization; subsequent full notebook
runs verify the final fields/flux. Phase timings exclude imports, device
initialization and extension builds. Sampled process-memory peaks can miss short
spikes. The benchmark runners retain the complete protocol for reproduction.

## Reproduction

Run from the repository root with the project environments installed:

```bash
# Analytical, contract, source, mode, archive and MPB-reference checks.
JAX_PLATFORMS=cpu bash benchmarks/reports/issue361/validate_suite.sh

# Native backend/material compression checks (requires built CUDA extension).
OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cuda,cpu .venv-gpu/bin/python -m pytest \
  tests/performance/test_cuda_tensor.py tests/performance/test_cuda_material_blocks.py \
  tests/performance/test_jax_material_blocks.py -q

# Representative full-tensor port workflow and independent reference.
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
  benchmarks/reports/issue361/validate_ports.py --rectilinear \
  --output /tmp/tensor-port.json
python benchmarks/reports/issue361/meep_reference.py \
  --resolution 80 --width 4 --output /tmp/mpb-reference.json

# Full notebook: repeat with cuda_streamed and cosine_waveguide_crossing.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cuda,cpu \
  XLA_PYTHON_CLIENT_PREALLOCATE=false .venv-gpu/bin/python \
  scripts/validate_modal_notebook.py --root "$PWD" --backend jax \
  --notebook modal_sources_monitors --expected-device 'RTX 3090' \
  --output benchmarks/results/modal-jax
```

`benchmarks/modal_backend_comparison.py` measures realistic cold setup/JIT,
warmed propagation, postprocessing and memory. `benchmarks/tensor_jax.py`,
`tensor_vs_diagonal.py`, `jax_material_blocks.py` and `cuda_material_blocks.py`
provide focused operator/compression benchmarks. Use their `--help` for cases
and output paths; generated evidence belongs under `benchmarks/results/`.
