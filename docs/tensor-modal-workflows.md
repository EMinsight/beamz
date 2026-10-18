# Full-tensor dielectric ports

The [full-tensor mode sources and monitors notebook](https://github.com/beamzorg/beamz/blob/main/examples/notebooks/modal_sources_monitors_full_tensor.ipynb)
demonstrates mixed eigenmodes, single-profile and broadband excitation,
independent flux checks, a width-step junction, and result save/load through
the public `results.mode()` interface.

A full-tensor port uses the same inverse-permittivity samples and interpolation
as propagation: diagonal inverse coefficients at electric Yee supports and
inverse cross coefficients at shared nodes. It requires the propagation
timestep. The standalone `solve_grid` tensor solver supplies a candidate; the
port then solves the discrete Maxwell problem and checks both first-order
Maxwell residuals. An unsuccessful coupled solve raises an error instead of
returning a diagonal approximation or an unrefined candidate.

## Bulk anisotropy and interface smoothing

Using diagonal coefficients in the bulk and tensor cross terms only near
interfaces is correct for isotropic bulk materials. It is not a general rule
for anisotropic materials: intrinsic cross coupling must also be retained
throughout their interiors. BeamZ selects the coupled update from the actual
inverse-permittivity coefficients, not an interface-versus-bulk classification.

Diagonal does not necessarily mean isotropic:

| Material or location | Required constitutive treatment |
| --- | --- |
| Homogeneous isotropic bulk | Scalar permittivity; no cross coupling |
| Anisotropic bulk with principal axes aligned to the grid | Three potentially different diagonal entries |
| Anisotropic bulk whose tensor has cross terms in the grid coordinates | Full tensor, including bulk cross coupling |
| Tilted interface between isotropic materials | Effective tensor from subpixel smoothing, generally including cross terms |

For a locally planar interface between isotropic, lossless dielectrics, the
effective inverse permittivity is

```text
epsilon_eff^-1 = I / <epsilon> + (<epsilon^-1> - 1 / <epsilon>) n n^T
```

Here `n` is the interface unit normal, `I` is the identity matrix, and
angle brackets denote a volume average over the smoothing neighborhood.
Tangential fields use the arithmetic mean of permittivity, while normal
fields use its harmonic mean. In homogeneous isotropic bulk the
parenthesized term vanishes. At a tilted interface, the products `n_i n_j`
generally generate off-diagonal entries. For an interface normal aligned
with a grid axis, the effective tensor is diagonal but can still be
anisotropic. Intrinsically anisotropic constituent materials require a more
general averaging formula. See
[Meep's explanation of the effective tensor](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/#smoothed-permittivity-tensor-via-perturbation-theory).

Consequently, `farjadpour_diagonal` and `farjadpour_full` can represent
different interface constitutive operators even when the underlying materials
are isotropic. Omitting nonzero interface cross terms is a change to the
discretization. Skipping work where those coefficients are already exactly
zero preserves the full-tensor operator.

### Scalar coefficients in homogeneous interiors

A scalar material such as `epsilon_r=12` uses the same permittivity for all
three electric components inside its homogeneous interior. The components
still have different curls and must each be updated. Diagonal propagation
already uses scalar multiply-adds, so the remaining opportunity is to reduce
coefficient storage and reads.

The native streamed backend identifies constant blocks during compilation.
Each block contains 32 consecutive samples on a component's staggered Yee
support. If their float32 bit patterns are identical, the block references one
constant in a pool shared by the three components. Otherwise it retains every
coefficient in the block. This is exact coefficient compression, not a
geometric assumption that the material is isotropic: anisotropic bulk and
mixed interface coefficients are preserved too. Globally uniform coefficients
use scalars directly.

This applies to ordinary diagonal updates and to the diagonal part of
full-tensor updates. Full-tensor cross coupling is unchanged. No additional
kernel launch or matrix solve is needed. The compiler retains the previous
scalar, byte-ID or dense representation when block storage would not save at
least 25%; fewer stored material bytes do not imply the same percentage
reduction in total runtime, since field traffic and curl calculations remain.
See the [measurements and correctness checks](https://github.com/beamzorg/beamz/blob/main/benchmarks/reports/issue361/README.md#performance).

Single-device 3D **JAX** also compresses electric permittivities and tensor
inverse diagonals. Uniform coefficients become scalars. Arrays with at most
256 distinct values use byte indices into a small table; other compressible
arrays use 32-sample constant/literal blocks. Incompressible arrays stay dense.
Equality is checked by bit pattern, so neither interface values nor small
anisotropic terms are approximated. Each component retains its own Yee
support and coefficients. The JAX byte-table preference gives XLA a simpler
gather than the native CUDA block encoding; it is not a change in physics.

The lookup feeds the existing update, allowing XLA to fuse it without keeping
a dense coefficient volume. Uniform-CPML source scales are computed on the
value pool when conductivity is scalar. Spatially varying conductivity retains
the ordinary dense scale calculation. Dispersion, sharding and 2D keep their
previous representations.

This helps larger simulations fit in GPU memory, but field arrays, CPML state,
monitors and tensor workspaces still consume memory. Use
`Simulation(..., setup_device="cpu")` when capacity matters: otherwise original
material grids retained by the simulation may also occupy device memory.
Keep the CPU backend enabled (for example `JAX_PLATFORMS=cuda,cpu`), and
construct any externally supplied JAX material arrays on the CPU as well;
the setting does not migrate arrays already held by the caller.
JAX's reserved allocator memory is different from bytes actively used by a
simulation. The [JAX compression report](https://github.com/beamzorg/beamz/blob/main/benchmarks/reports/issue361/README.md#performance)
records compiled-buffer estimates, independent allocator peaks and timing.

Value pools remain differentiable, with gradients summed across samples that
share a pool entry. The integer map is fixed. To change formerly equal cells
independently, supply dense material arrays or recompile; the public
differentiable material binding supplies dense arrays and retains per-cell
gradients. Source, mode and monitor material snapshots are unchanged.

For realistic silicon/silica/air guides with CPML, seven broadband source
profiles and 17-frequency mode monitors, the measured gains over main are
smaller: 2.3-2.8% for JAX and less than 1% for CUDA on the tested diagonal
workflows. Cold setup and JIT remain close to main; mode solving and modal
postprocessing dominate time to the first analyzed result. Host setup still
needs dense material grids. See the
[modal workflow comparison against main](https://github.com/beamzorg/beamz/blob/main/benchmarks/reports/issue361/README.md#performance)
for separate preparation, compilation, propagation and memory measurements.

## Supported scope

The implemented workflow supports confined 3D guided modes on uniform and rectilinear
Cartesian grids, with real, symmetric, positive-definite electric tensors,
`mu=1`, zero conductivity, and no dispersion. Both intrinsic anisotropy and
cross terms from `farjadpour_full` smoothing are supported. Ports must be
longitudinally invariant over their constitutive neighborhood, with uniform
normal-axis cell spacing there. Transverse cells may vary, and the device
interior may be graded along any axis. All three axes,
both directions, and CPML boundaries have end-to-end CPU and RTX 3090 JAX validation.
Execution supports single-device JAX and native streamed CUDA; sharded tensor
modal workflows are rejected. GPU correctness and performance measurements
are recorded in the [JAX performance report](https://github.com/beamzorg/beamz/blob/main/benchmarks/reports/issue361/README.md#performance)
and [streamed CUDA report](https://github.com/beamzorg/beamz/blob/main/benchmarks/reports/issue361/README.md#performance).

Rectilinear full-tensor propagation rejects periodic/Bloch boundaries and
sharding because their constitutive seam/halo averaging is not implemented.

Full-tensor 2D modal workflows, magnetic coupling, conductive
or dispersive media, homogeneous/radiation port bases, and forced mirror
symmetry are outside this implementation. Full-tensor propagation itself has
separate capabilities. A mode solve does not extend those capabilities.

Use an explicit mixed-polarization selection:

```python
mode = bz.ModeSpec(num_modes=2, polarization=None, target_neff=2.0)
port = bz.Port(
    center=(2.0e-6, 1.8e-6, 1.8e-6),
    size=(0.0, 2.2e-6, 2.2e-6),
    direction="+", name="input", mode_spec=mode,
)
source = port.to_source(frequency, bandwidth)
monitor = port.to_monitor([frequency])
```

An omitted `Port.mode_spec` retains the existing TE default. Passing an explicit
`ModeSpec(polarization=None)` now preserves that choice instead of converting
it to TE. TE/TM ordering is rejected for coupled tensor ports because the modes
can mix both polarizations.

Candidates are ordered by the existing effective-index ordering near the
requested target. Discrete refinement tracks the selected candidate by electric
field overlap and excludes the static curl-curl nullspace. Mode indices are
local to each frequency/geometry solve: automatic branch tracking across a
frequency sweep or through degeneracies is not implemented. For such sweeps,
compare field overlaps and treat a degenerate modal subspace together.

## Support, timing, and power

Eligibility is local. Mode assembly checks its transverse aperture plus the
existing outer padding, the two normal Yee planes, two additional neighboring
cells on each normal side, and the upper nodes required by each staggered
support. Monitor snapshots retain a sufficient normal slab and all shared-node
tensors. Missing support is an error; cropping cannot wrap negative indices.
A tensor in the device interior therefore does not automatically reject a
locally diagonal 3D source or port. The older global restriction remains for
2D full-tensor modal workflows.

The source phasor convention is `exp(+i omega t - i beta x)`, with E at integer
steps and H at half steps. The monitor DFT integrates against `exp(+i omega t)`,
so projection conjugates these launch phasors. For real reciprocal tensors the
opposite-traveling mode is `(conj(E), -conj(H))`; simply changing the H sign is
insufficient for complex mixed-polarization fields. Axis transformations rotate
the tensor and treat H as an axial vector.

The modal basis is normalized using the same discrete Poynting quadrature as
the flux monitor, **after** applying normal Yee interpolation separately to each
propagation direction. Consequently `ModeMonitorData.power` (`abs(amps)**2`)
reports each mode's discrete flux contribution. Normalization scales the whole
E/H basis together and preserves the phases needed to reject false reflection.
The source retains its measured-flux calibration, including normal interpolation.
This applies to JAX and CUDA, diagonal and full-tensor materials, and uniform
and rectilinear grids.

Use `modes.modal_flux` and `modes.unresolved_flux` for complete accounting,
including cross terms for nonorthogonal modes. For matched-reference transmission,
use `output.normalized_to(incident, mode_index=0, direction="+").power`.
Different subcell plane positions can have different interpolation responses;
modal watts track the measured flux at the same plane. Equivalent sampling
positions or matched references avoid mixing these finite-grid responses.

See the [monitor API documentation](https://github.com/beamzorg/beamz/blob/main/beamz/devices/monitors/README.md#3d-modal-sampling-and-power)
and [joint flux/modal validation](https://github.com/beamzorg/beamz/blob/main/benchmarks/reports/issue361/README.md#validation).
No curve is clipped or forced to stay below the requested source power.

`ModeSource.solve_modes()` on a full-tensor material grid now previews the
same discrete modes, collocated for `ModeData` display. Its per-frequency
`diagnostics[...]["modes"]` includes the coupled residuals. Diagonal-only grids
retain the previous preview path. The returned tensor-mode `neff` is
`beta*c/omega`, including spatial and temporal discretization.

## Rectilinear constitutive and curl metrics

At a node between cells of widths `h_left` and `h_right`, the constitutive
cell-to-node average is `(h_left*u_left + h_right*u_right)/(h_left+h_right)`.
The reverse node-to-cell average is one half on each adjacent node. This pair
is adjoint in the corresponding Yee control-volume inner products; it is not
pointwise linear interpolation. Symmetric shared-node cross coefficients then
produce a reciprocal constitutive operator. Equal widths recover the uniform
operator exactly. Source commutators and the energy diagnostic use this same
weighted constitutive averaging.

The mode solver takes primal curl distances from actual cell edges and dual
distances from neighboring cell widths, including neighbors outside a cropped
aperture. Along the uniform port normal it retains the discrete sine derivative
and cosine averaging symbols. Axis rotations transform grid metrics together
with fields and tensors. Monitor sampling uses physical coordinates and area
weights; detached archives retain these metrics for subsequent mode solves.

See the [rectilinear validation report](https://github.com/beamzorg/beamz/blob/main/benchmarks/reports/issue361/README.md#validation)
for the adjoint identity, production-update checks, source/monitor experiments,
grid-transition reflection, convergence, and independent MPB comparison.

## JAX performance

Single-device JAX automatically compacts localized inverse-tensor cross
coefficients into a fixed bounding box when its interpolation halo occupies
at most half the node grid. Coupled corrections run on that region; diagonal
updates still cover the full grid. Intrinsic anisotropy and interface smoothing
use the same criterion, and the compactor retains every nonzero coefficient.
Cross coupling whose bounding box is too large, including widespread bulk
anisotropy and widely separated regions, retains the original dense update.
Here, **dense** describes full-grid coefficient storage; it does not mean
diagonal, isotropic, or physically dense material. The compact path uses one
bounding box, so it still processes zero coefficients inside that box rather
than operating on an arbitrary sparse list of interface cells.

This is exact support compression, with an immutable zero halo preserving
neighboring Yee contributions. Within the compact region, cross contributions
are summed at shared nodes before interpolating back to each electric
component. Linearity makes this algebraically equivalent to interpolating each
contribution separately, with possible floating-point summation differences.
Dense-reference and gradient tests check that equivalence. These checks
establish equivalence to BeamZ's original full-tensor update; they do not by
themselves establish identical discretization or convergence to Meep.

Packed coefficient values remain differentiable.
Changing their spatial support requires recompilation or supplying dense
inverse-tensor coefficients through `run_coefficients`. Measured speedups,
memory costs, hardware, reproduction commands, and a separate comparison
against `farjadpour_diagonal` are in the
[JAX performance report](https://github.com/beamzorg/beamz/blob/main/benchmarks/reports/issue361/README.md#performance).

## How Meep handles tensors

Meep preserves intrinsic bulk anisotropy and computes effective interface
tensors during initialization. Its additional tensor storage extends across
the enclosing rectangular processor chunk, even when cross coupling occurs
only near interfaces. See
[Meep's smoothing and runtime documentation](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/#impact-on-runtime-performance).

For linear, nondispersive electric materials, Meep applies the precomputed
inverse permittivity to the displacement field to obtain the electric field.
Its kernels average cross-component contributions on the staggered grid and
select diagonal or coupled loops according to which coefficient arrays are
present. This is not a matrix inversion at every cell on every timestep.
See Meep's
[constitutive field update](https://github.com/NanoComp/meep/blob/master/src/update_eh.cpp)
and [diagonal and off-diagonal kernels](https://github.com/NanoComp/meep/blob/master/src/step_generic.cpp).

BeamZ likewise precomputes inverse coefficients before stepping. Its compact
JAX update reduces the region of cross-coupling work while retaining bulk and
interface terms wherever they are nonzero. This follows the same physical
distinction as Meep; the implementations and Yee interpolation stencils need
not be identical.

## Native streamed CUDA

The same full-tensor material grid, mode sources and monitors can run through
the native streamed backend:

```python
results = simulation.run(backend="cuda_streamed")
```

This requires CUDA component **0.30.0 / ABI 30** built against compatible JAX
headers. Explicit `backend="cuda"` with the streamed schedule also works;
`backend="auto"` continues to select JAX for full-tensor materials.

The electric phase reuses the native Yee curl and CPML recurrences, advances
the inverse-diagonal contribution everywhere, and stores corrected curls only
over the tensor bounding box plus its interpolation halo. A second native
kernel adds the full cross-component correction using the same shared-node
averaging and rectilinear weights as JAX. Widespread anisotropy uses full-grid
curl scratch. The existing in-place CUDA graph schedules both kernels before
electric-source injection and DFT observation, preserving leapfrog timing.
The ordinary diagonal kernels remain unchanged.

Both phase-by-phase execution and complete graph execution are supported.
Source profiles, mode solving, monitor projection and detached result loading
use the shared tensor workflow; the eigensolver itself has not been rewritten
in CUDA. Coefficients remain dynamic inputs to the native graph.

This implementation supports float32 fields on unpadded, canonical 3D storage,
with uniform, axis-uniform or rectilinear grids and PEC/CPML boundaries.
Full-tensor CUDA autodiff, the `mstep` schedule, rotated/padded storage, and
rectilinear tensor sharding or periodic/Bloch seams are outside this path.
Use JAX for full-tensor gradients. Changing a compact tensor's spatial support
requires recompilation or dense coefficient inputs, as in the JAX path.

See the [streamed CUDA validation and benchmark report](https://github.com/beamzorg/beamz/blob/main/benchmarks/reports/issue361/README.md#performance)
for dense/localized tensor checks, all modal axes and directions, graph
continuation, numerical tolerances, memory costs, and reproduction commands.

## Detached results

```python
results.save("tensor-guide.npz")
restored = bz.SimulationResults.load("tensor-guide.npz")
```

The versioned NPZ archive retains complex DFT arrays, raw and normalized
acquisitions, geometry, grid metrics, timestep, source specifications, and
complete cell/electric-support/shared-node material context. Loading does not
use pickle or a live simulation. Standard frequency field, flux, and mode
monitors and registered source specifications are supported. Unsupported
objects, including recorder types outside that registry, raise an error
instead of being omitted. Archives are version-checked; compatibility with
unimplemented future schema versions is not promised.

The mathematical reduction, independent MPB comparison, measured errors,
limitations, and reproducible commands are in the
[issue 361 validation report](https://github.com/beamzorg/beamz/blob/main/benchmarks/reports/issue361/README.md).
