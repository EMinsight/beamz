# Curved interfaces: when full-tensor smoothing pays off, and its memory

Measured on a local RTX 5070 Laptop GPU (8 GB; 6.87 GB usable at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`).
**SiN bend:** accuracy and GPU time on BeamZ v0.5.2 pinned at `8ab968ac` (2026-10-12/13;
driver 580.173.02, Python 3.11.15, JAX 0.9.0, x64 enabled, raster/material arrays float32); GPU
memory measured on `main` at `80ef0182`, with and without the CPML fix below (times re-confirmed
there). **Si structures** (U-bend, S-bend coupler, U-bend
pair): released BeamZ 0.5.3, JAX 0.10.2 (cuda13). Upstream `main` is now at `0e3838d8`.
pp = percentage points.

## Finding

`smoothing="farjadpour_diagonal"` (the default) drops only the off-diagonal term ε_xy of the
exact cut-cell tensor. That term is zero for grid-aligned walls and largest for walls at 45°.
Full-tensor smoothing therefore helps only when the observable depends on fields at oblique
walls. On four structures:

| Structure | Observable | Diagonal vs full | Verdict |
|---|---|---|---|
| Single-mode Si U-bend | T, phase | ≤ 0.13 pp, ≤ 166 mrad; grid moves phase more | diagonal adequate |
| S-bend directional coupler | cross power κ² | 2.4–3.7 % relative; grid effect larger | diagonal adequate |
| Back-to-back Si U-bend pair | cross power κ² | diagonal 4–10 % high | full 1.6–4× cheaper at equal accuracy |
| Multimode SiN bend | mode-0 → second-mode split | 3.3 % vs 4.1 % converted at 62.5 nm | full 13.2× faster, 5.9× less memory |

**Practical rule:** run diagonal and full at the working grid on the observable that matters, and
switch to full if they differ by more than the tolerance. Expect a difference for curved
coupling gaps (ring and racetrack couplers) and for multimode or mode-conversion devices; not
for single-mode bends or S-bend couplers. Diagonal remains the right default: it is exact at
grid-aligned walls (straight-guide controls agree with full tensor to roundoff) and the
cheapest per cell.

## Why: what each method assigns to a cut cell

At a flat wall the field component along the wall sees the arithmetic mean of ε, and the
component across it sees the harmonic mean. The exact cell tensor is
`ε = ε_harm·n nᵀ + ε_arith·(I − n nᵀ)`, aligned with the wall normal **n**
(Farjadpour et al., Opt. Lett. 31, 2972 (2006); Kottke et al., PRE 77, 036611 (2008)).
Its off-diagonal term is ε_xy = (ε_harm − ε_arith)·n_x·n_y ∝ sin 2θ.

![Cut-cell permittivity ellipses for volume, diagonal and full smoothing, and |ε_xy| against wall angle](curved-interface-smoothing/methods.svg)

Volume is isotropic and diagonal locks the ellipse to the grid axes; only full tensor matches the
exact wall-aligned tensor. The ε_xy that diagonal drops vanishes at 0° and 90° and peaks at 45°.

| | `volume` | `farjadpour_diagonal` (default) | `farjadpour_full` |
|---|---|---|---|
| Cut-cell value | arithmetic mean, all directions | diagonal of the exact tensor | full exact tensor |
| Exact for a flat wall at | no orientation | grid-aligned walls only | any orientation |
| Ex update reads | Dx at the same point | Dx at the same point | Dx plus neighbouring Dy, Dz |
| Measured cost per cell | 1× | 1× | ~1.45× time, ~1.5× memory (SiN bend) |

## When it matters: where ε_xy sits relative to the coupling

![ε_xy maps for the single U-bend, S-bend coupler and U-bend pair, and wall obliquity along the coupling gap](curved-interface-smoothing/where_exy_lives.svg)

Every bend carries ε_xy on its walls, but in the S-bend coupler the guides couple where the walls
are nearly grid-aligned, while the U-bend pair's gap-facing walls are more oblique (model
coupling-weighted mean 0.26 vs 0.17) and its diagonal-vs-full gap is larger; two devices, so
consistency, not proof.

## Results per structure

### Single-mode Si U-bend (500 × 220 nm, R = 2 µm)

![Phase of S21 and same-grid |ΔT| against full tensor for the single U-bend](curved-interface-smoothing/ubend_same_grid.svg)

Diagonal stays within 0.13 pp of full on every grid. From 60 to 30 nm the grid moves the phase
of diagonal and full by 390–470 mrad, 2.4–5× their same-grid difference, so the grid, not the
smoothing method, limits accuracy. Volume's phase barely moves: its wall error happens to offset
the grid error here, which is not a sign of accuracy.

| Grid | 60 nm | 50 nm | 40 nm | 30 nm | 20 nm* |
|---|---:|---:|---:|---:|---:|
| \|ΔT\| diagonal vs full | 0.133 pp | 0.101 pp | 0.066 pp | 0.058 pp | 0.045 pp |
| \|Δφ\| diagonal vs full | 166 mrad | 114 mrad | 104 mrad | 92 mrad | 65 mrad |

\*20 nm from a separate run. Full tensor's phase moves 466 mrad from 60 to 30 nm. With no second
guided mode to convert into, and full costing 1.5–1.9× per grid, diagonal is the adequate choice.

### Multimode SiN bend (0.5 × 2 µm core, R = 20 µm)

![Total guided power, power remaining in mode 0, and power converted to the second mode for the SiN bend](curved-interface-smoothing/sin_mode_conversion.svg)

The methods agree on total guided power but differ in how much the bend converts mode 0 into
the second guided mode; all three converge toward about 4 %, with full tensor already there at
125 nm.

Total guided power agrees to ~0.01 pp at 62.5 nm (99.39 / 99.38 / 99.39 % for volume /
diagonal / full). The difference is conversion into the second guided mode (`o2_m7`), whose
n_eff is close to mode 0 (1.674 vs 1.694 at 125 nm, 1.666 vs 1.686 at 62.5 nm). With
Δn_eff ≈ 0.02 the beat length is ≈ 78 µm, longer than the ~31 µm bend path, so the coupling
builds up coherently. Converted fraction:

| Grid | volume | diagonal | full |
|---|---:|---:|---:|
| 125 nm | 0.2 % | 2.3 % | 3.8 % |
| 100 nm | 0.6 % | 2.8 % | 4.0 % |
| 62.5 nm | 1.7 % | 3.3 % | 4.1 % |

That all three converge toward ~4 % is evidence beyond the same-solver full-tensor reference.
The 13.2× / 5.9× gain below applies to this mode-0 split.

### Si S-bend directional coupler and back-to-back U-bend pair (gap 200 nm)

![Cross power κ² against grid for the U-bend pair and the S-bend coupler, and the relative same-grid gap](curved-interface-smoothing/coupler_convergence.svg)

Diagonal over-predicts κ² for the U-bend pair by 4–10 % on the same grid, against 2.4–3.7 % for
the S-bend coupler, where the grid effect is larger than the method effect.

**S-bend coupler** (R = 5 µm arcs, 1.5 µm offset). Diagonal ≈ full: the same-grid relative κ²
gap is ~2.4–3.7 %, and coupling happens where walls are nearly grid-aligned. A variant with a
12 µm straight section (≈ 65/33 split) shows the same: diagonal − full cross power differs by
0.10–0.17 pp, against ~2.5 pp grid variation. That variation is non-monotonic (32.4 / 34.8 /
32.7 % at 60 / 50 / 40 nm); at 50 nm the 200 nm gap falls on exactly 4 cells.

**U-bend pair** (⊃⊂, R = 2 µm). Diagonal over-predicts cross power. Reference: full at 20 nm,
0.644 %.

| Grid | Cross, diag / full | GPU time, diag / full | Peak GPU, diag / full |
|---|---:|---:|---:|
| 60 nm | 0.814 / 0.740 % | 3 / 5 s | 0.14 / 0.17 GB |
| 50 nm | 0.815 / 0.761 % | 7 / 12 s | 0.25 / 0.29 GB |
| 40 nm | 0.782 / 0.738 % | 20 / 31 s | 0.45 / 0.58 GB |
| 30 nm | 0.697 / 0.658 % | 71 / 98 s | 1.12 / 1.36 GB |
| 25 nm | 0.703 / 0.675 % | 158 / 231 s | 1.93 / 2.49 GB |
| 20 nm | 0.671 / 0.644 % | 386 / 564 s | 3.66 / 4.64 GB |

- Full at 30 nm (+2 % vs reference) beats diagonal at 20 nm (+4 %): 3.9× faster and 2.7× less
  memory. The reference is full tensor and favours it; the fair pair against that bias is
  diagonal at 25 nm vs full at 30 nm, 1.6× faster. The gain is therefore 1.6–4×.
- Above ~30 nm the grid error dominates: both methods are 15–21 % high at 40 nm.
- Grid-alignment noise is about ±1 mrad of coupling angle (both rise at 25 nm). The incident
  amplitude a_in is ≈ 1.010 at 60 nm and 1.0025 at 30 nm, against 1.000 elsewhere.
- The same-grid check (4–10 % in κ²) flags this case without a fine reference.

## SiN measurements

- **Structure:** a 0.5 × 2 µm SiN core (n = 2.0) in SiO2 (n = 1.44), with an R = 20 µm, 90° bend at λ = 1.55 µm.
- **Domain:** 31 × 34 × 10 µm with CPML, run for 800 fs.
- **Source and observable:** a Gaussian mode-0 source. The observable is mode-0 power 6 µm after the bend, normalised by the measured incident modal power and jointly projected onto six bound modes.
- **Grids and timing:** uniform 125, 100 and 62.5 nm, with straight-guide controls per method and grid; synchronized GPU time excluding setup and JIT (rows with four timed runs: median of three warm repeats, spread ≤ 0.15 s).

The **reference** is the 62.5 nm full-tensor run. Its screening margin is **0.74 pp**: twice the
refinement change, plus incident-mixture and straight-control indicators. It favours full
tensor. Volume (99.3 → 97.7 %) and diagonal (96.5 → 96.1 %) both move monotonically toward it.

Error and time are from v0.5.2. Guided powers on `main` differ from v0.5.2 by ~4e-6 (a CPML
change on `main`), which is negligible. Peak GPU memory is JAX `peak_bytes_in_use` on `main`
(80ef0182); "—" = not re-measured on `main`.

| Smoothing | Grid | Cells | Error vs reference | GPU time | Peak GPU, main (80ef0182) | Peak GPU, main + fix |
|---|---|---:|---:|---:|---:|---:|
| volume | 125 nm | 5.4 M | 4.00 pp | 9.1 s | — | — |
| volume | 100 nm | 10.5 M | 3.54 pp | 27.8 s | — | — |
| volume | 62.5 nm | 43.2 M | 2.37 pp | 174.3 s | — | — |
| diagonal (default) | 125 nm | 5.4 M | 1.15 pp | 9.0 s | 0.78 GB | 0.54 GB |
| diagonal (default) | 100 nm | 10.5 M | 1.12 pp | 27.9 s | — | — |
| diagonal (default) | 62.5 nm | 43.2 M | 0.76 pp | 174.2 s | 5.62 GB | 4.29 GB |
| full tensor | 125 nm | 5.4 M | 0.32 pp | 13.2 s | 1.07 GB | 0.73 GB |
| full tensor | 100 nm | 10.5 M | 0.09 pp | 39.6 s | — | — |
| full tensor | 62.5 nm | 43.2 M | reference | 251.1 s | **out of memory** | 6.43 GB |

| Need | main (80ef0182) | main + fix | Gain |
|---|---|---|---|
| ~1.2 pp, defaults | diagonal 125 nm: 9.0 s, 0.78 GB | same run: 9.0 s, 0.54 GB | 1.45× less memory |
| ≤ 0.8 pp, default smoothing | diagonal 62.5 nm: 174 s, 5.62 GB | full 125 nm: 13.2 s, 0.73 GB | 13.2× faster, 7.7× less memory |
| ≤ 0.8 pp, already using full | full 125 nm: 13.2 s, 1.07 GB | full 125 nm: 13.2 s, 0.73 GB | 1.46× less memory |
| Fine full-tensor run, 8 GB GPU | out of memory | 251 s, 6.43 GB | now possible |

Like-for-like, with the fix applied to both methods, full at 125 nm against diagonal at 62.5 nm
is 13.2× faster with 5.9× less memory (0.73 vs 4.29 GB). The time gain comes from choosing full
tensor, which `main` already offers; the fix contributes the memory reduction (diagonal at
62.5 nm: 1.31× less).

## The memory fix

**Uniform CPML constants** (`beamz/devices/_boundary_compile.py`). For 3D all-CPML grids,
spatially uniform conductivity/permeability, and the matching `sig_*`/`mu_h*` Yee arrays, are
stored as exact 0-d scalars. The change is visible: compiled `conductivity`, `permeability`,
`sig_*` and `mu_h*` are now 0-d when uniform (broadcast them to recover arrays). At a coarse
500 nm check grid, CPML arrays, profiles and masks are bit-identical after broadcast, and stored
boundary material bytes fall from 12.6 to 4.2 MB. Tests: `tests/kernels/test_compact_cpml_constants.py`
(8 tests).

The fix changes setup only, not physics: guided powers and flux are bit-identical with and without
it for diagonal@125, full@125 and diagonal@62.5, and run times are unchanged. Full@62.5 has no
unpatched baseline: on `main` it runs out of memory on a 1.46 GiB allocation at first execution.
With the fix it peaks at 6.43 of 6.87 GB and agrees with the earlier v0.5.2 run to ~5e-8. A 50 nm
reference would need about 12.5 GB. An inverse-tensor upload change was also tried on v0.5.2; it
is not needed on `main` and is not part of this fix.

## Limitations

- **Four structures, chosen observables.** One SiN bend (mode-0 split) and three Si devices (T, phase, κ²). Radiation accounting and other geometries are not measured.
- **Same-solver references.** Both references are full-tensor runs and favour full tensor. For SiN, within the 0.74 pp margin, full@125 (95.00 %) and diagonal@62.5 (96.08 %) are not separable; "at least as accurate" is the defensible claim. The SiN ~4 % convergence of all three methods and the U-bend pair's 1.6× lower bound are what survive that bias.
- **Orientation not tested.** A 45° variant was run, but it is a different device: the extra arcs change the mode at the gap and add output bend loss, and mode ports must be axis-aligned. No orientation result is claimed.
- **Mechanism evidence is correlational.** The ε_xy-at-the-gap ordering matches two coupler devices; it is not a controlled test.
- **Laptop GPU, two revisions.** SiN on pinned v0.5.2, Si on released 0.5.3. Memory is the JAX peak in use, measured on `main` (80ef0182).
- **Not a new sampling scheme.** The 2D published eight-triplet update (Werner et al., arXiv:1212.4857) showed a 7.57× equal-accuracy speedup on a ring cavity, but it has no open-boundary 3D implementation, so it is not part of this comparison.

## Reproduction

**Si structures.** The single U-bend and U-bend pair runs are in the tutorial notebook
[`examples/notebooks/interface_smoothing_choice.ipynb`](../../examples/notebooks/interface_smoothing_choice.ipynb).
The S-bend coupler and the 20 / 25 nm reference runs were one-off scripts and are not in the
repository.

**SiN bend.** The benchmark scripts, raw per-case JSON and notebook are **not part of this
repository**. They are on the `cdaunt/beamz` fork, branch `feat/fit-interface-prototype`, as
working files, and will move to an examples repository. The notebook is
`docs/explicit_anisotropic_experiment.ipynb`; the runner is `scripts/run_sin_gpu_study.py`. They
need a pinned upstream checkout with its native extension built (this is the v0.5.2 accuracy/time
run; `--compact-inverse-coefficients` is the inverse-upload change, not needed on `main`):

```bash
.venv/bin/python scripts/run_sin_gpu_study.py --beamz-root <pinned-beamz-checkout> \
  --output-dir <results-dir> --report-name <name> \
  --compact-inverse-coefficients --compact-boundary-constants --compact-boundary-scope all \
  --extra-repeats bend:farjadpour_full:125 bend:farjadpour_diagonal:62.5 --fresh
```
