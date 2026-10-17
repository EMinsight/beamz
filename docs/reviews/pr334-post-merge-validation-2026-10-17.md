# PR #334: post-merge RTX 3090 validation, 2026-10-17

The merged solver matches current `main` exactly on both full tutorials and all four benchmark grids. No new CUDA numerical regression was observed. Streamed timing remains close to both parents; mstep retains shape-dependent tradeoffs. The merge also brings an upstream modal-projection change: compared with the original PR, the modal tutorial has different projection coefficients and slower CPU analysis. This is not a blanket claim that every output or runtime is unchanged.

## Revisions and method

- Main: `5a39a926b4fbfddc33350dac0625728aaf22430f` (CUDA ABI 27).
- Original PR: `742d72052d2de24d4cdb25c7333f150a0084f55d` (CUDA ABI 23).
- Merge: `d9daa93263d839c4ad369210fbceb41a20f15777` (CUDA ABI 28).

One RTX 3090 24 GB, driver 610.43.03, 370 W power limit, Python 3.11.15, JAX/jaxlib 0.9.0. Each revision has its own native CUDA build, compiled with CUDA 13.3.73 for SM86 with fast math disabled. Tracked source files were checked against their Git blobs. All jobs ran serially; JAX preallocation was disabled. No solver changes or tolerance changes were made for this validation.

Full tutorial settings were used with `BEAMZ_DOCS_TEST` disabled. Original cells ran top to bottom, followed by raw numerical exports and three repeated simulations. Each repeat checks its fields against its first run with exact equality. The saved notebooks pass format/execution checks; the field and spectrum plots were visually inspected. All 10 modal and 5 cosine images match `main` byte for byte on the merged streamed backend.

## Full notebooks

Times below are medians of three warm solver runtimes, excluding plotting and CPU modal decomposition.

| Simulation | Main streamed (s) | Original PR streamed (s) | Merged streamed (s) | Merged vs main |
|---|---:|---:|---:|---:|
| Modal single profile | 2.452 | 2.470 | 2.476 | +0.98% |
| Modal broadband | 2.469 | 2.482 | 2.490 | +0.84% |
| Modal junction | 2.838 | 2.853 | 2.850 | +0.44% |
| Cosine crossing | 19.593 | 19.661 | 19.648 | +0.28% |

- Modal: nonuniform 180×77×56 `(x, y, z)` grid for the straight waveguide, 12,605 steps, 17 monitor frequencies, three modes and seven broadband source profiles. The junction has its own discretization. All **37 exported arrays are bitwise identical to main**, including unnormalized fields, flux, complex mode amplitudes, effective indices and plotted power accounting. The nonuniform grid uses streamed CUDA; mstep supports uniform grids.
- Cosine crossing: uniform 384×384×56 `(x, y, z)` grid, 14,695 steps, 101 frequencies. All **10 exported arrays are bitwise identical across main, original PR and merged streamed/mstep**, including raw electric fields, both raw fluxes, launched power and normalized spectra.
- Cosine mstep: original PR **22.328 s**, merge **22.376 s** (+0.22%). Merged mstep is **13.9% slower** than streamed for this geometry. Keep streamed for this notebook.

The exported spectra are not the only accuracy check: raw unnormalized arrays are compared as well, so normalization cannot hide a propagation error. Warm end-to-end `simulation.run` times, first-run solver times and individual samples are retained in the evidence archive. The first repeated call can rebuild a cache because progress is disabled; all three samples are retained.

### Upstream modal-projection difference

Notebook comparisons use `abs(error) <= 1e-6 * reference_array_peak + 3e-5 * abs(reference)` with no absolute unit floor. Against the original PR, 21 of the 37 modal exports are exact, three differ only at floating-point roundoff, and 13 exceed the fixed comparison tolerance. Raw fields, raw fluxes, source effective indices and frequency grids remain exact. The changes are in the projected complex amplitudes and derived power accounting.

Upstream commit `38479de47d240b1b36018e724e2ace14ad521b56` expands the candidate pool when TE/TM polarization is requested. The mode-monitor adapter defaults 3D projection to TE, so this tutorial now searches 11 candidates instead of three. Its source solves remain unpolarized and unchanged. A controlled diagnostic uses the same detached single-source FDTD results and changes only this candidate-count block in memory: the current implementation reproduces the main amplitudes exactly, and the legacy candidate count reproduces the original PR amplitudes exactly. The function is restored afterward; no tracked source is modified.

The original modal notebook cells completed in about **78.2 s on the original PR, 91.9 s on main and 91.0 s on the merge** in these fresh processes. This is roughly a 16% increase versus the original PR, concentrated in CPU mode analysis; it is already present on main. These are single notebook executions, not repeated cold-start statistics. Additional export cells repeat projections and are excluded from those figures.

The maximum change in plotted forward mode power is about 0.000543 for the single profile and 0.000188 for broadband (plots scaled near one). At the central single-profile frequency, forward mode-0 power changes from 0.999665014 to 0.999657500. Relative errors in near-zero residual channels can look large. The diagnostic identifies a basis-selection change; it does not establish a new independent physical-accuracy ground truth. Preserve the upstream selection fix and assess any desired projection-speed optimization separately.

## Matched-grid benchmarks

Each case uses float32 fields, a mode source, two mode-monitor DFT planes with three frequencies, 12-cell all-wall CPML and 256 steps. Shapes below are `(z, y, x)`. Binary material uses the compact representation; smooth material exercises dense coefficients. The common workload builder is identical across revisions, and all initial-state, coefficient and source fingerprints match.

Two fresh processes per variant use four warmups plus at least two additional warmup seconds, nine synchronized execution samples and three public `advance` samples. The initial sequence is main → original streamed → merged streamed → merged mstep → merged mstep → merged streamed → original streamed → main. Original mstep was measured afterward in two fresh processes per case. These short runs characterize this GPU and workload; small percentage differences are not statistical proof of universal equivalence.

| Shape / material | Main streamed GCUPS | Original streamed | Merged streamed | Original mstep | Merged mstep |
|---|---:|---:|---:|---:|---:|
| 48×96×128 / binary | 5.535 | 5.484 | 5.540 | 3.649 | 3.663 |
| 96×192×256 / smooth | 7.455 | 7.438 | 7.436 | 6.678 | 6.696 |
| 160×320×384 / binary | 8.691 | 8.639 | 8.641 | 9.912 | 9.921 |
| 512×256×96 / smooth | 7.403 | 7.406 | 7.403 | 4.998 | 4.997 |

GCUPS is the median of the two process medians, including sources, CPML, DFTs and storage conversion but excluding setup/compilation. Higher is better. Public-call timings below include the warmed Python API overhead; lower is better.

| Shape | Main public (ms) | Merged public (ms) | Streamed execution vs main | Mstep execution vs original mstep |
|---|---:|---:|---:|---:|
| 48×96×128 | 35.36 | 35.67 | -0.08% | -0.37% |
| 96×192×256 | 180.41 | 180.87 | +0.25% | -0.27% |
| 160×320×384 | 634.30 | 638.64 | +0.58% | -0.09% |
| 512×256×96 | 474.50 | 473.19 | +0.00% | +0.02% |

All **40 benchmark processes** passed the full-state check. Each has 30 state leaves; every leaf is bitwise identical to the matching main reference, including electric/magnetic fields, CPML state, DFT buffers and monitor counters. Validation reads every element, not a sample. In the large 256-step case, the pulse has not substantially reached the distant monitors (maximum DFT signal about 2.1e-18); its field-state comparison is meaningful, while the full notebooks provide the long-duration spectral checks. The unchanged tolerance fallback was therefore unnecessary. The 40 processes include the four reference-generation runs, so there are 36 independent comparisons against those references.

First-result setup/compilation timings and individual execution/public samples are retained in the archive. Driver/JIT caches were not purged, so these are not clean-machine installation/startup measurements. Two-second `nvidia-smi` telemetry is retained. Fine-grained NVML sampling was unavailable (`pynvml` missing); this report makes no peak-memory comparison.

## Reproduction and evidence

Run each checkout in a fresh process after building its matching native component. For example:

```console
python scripts/validate_modal_notebook.py --root /path/to/checkout --output /path/to/new-modal-output --notebook modal_sources_monitors --backend cuda_streamed --warm-runs 3 --expected-device "RTX 3090"
python scripts/validate_modal_notebook.py --root /path/to/checkout --output /path/to/new-cosine-output --notebook cosine_waveguide_crossing --backend cuda_streamed --warm-runs 3 --expected-device "RTX 3090"
python scripts/benchmark_cuda_revision.py --root /path/to/main --shape 160 320 384 --material binary --steps 256 --warmups 4 --warmup-seconds 2 --samples 9 --public-samples 3 --output /path/to/main.json --reference /path/to/new-reference --save-reference
python scripts/benchmark_cuda_revision.py --root /path/to/merge --shape 160 320 384 --material binary --steps 256 --warmups 4 --warmup-seconds 2 --samples 9 --public-samples 3 --output /path/to/merged.json --reference /path/to/new-reference
```

Use `--backend cuda_mstep` for the uniform-grid mstep comparisons. The notebook runner now supports that backend and `--warm-runs`; it retains executed outputs and provenance.

[Compressed evidence](pr334-post-merge-validation-2026-10-17.json.gz) includes all samples, array comparison metrics/hashes, input fingerprints, revision/native build identities, commands, logs, telemetry, notebook execution/image checks and the controlled diagnostic. Executed notebooks, raw NPZ exports and full-state references remain locally under `.cache/pr334-regression-20261017/`; they are not shipped in Git.

This is RTX 3090 forward-simulation regression evidence, not an H100/multi-GPU result or a fresh adjoint benchmark. The previously reported three streamed hardware-test failures still reproduce on both parents; this work neither changes their assertions nor claims a fully green hardware suite.
