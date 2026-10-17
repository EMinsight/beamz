# PR #334: CUDA validation summary

The integration keeps streamed CUDA as the default. On the RTX 3090, merged
streamed results match main bitwise in both full tutorials and four benchmark
grids. Multistep performance depends on shape; it is not a universal speedup.

## Post-merge validation — 2026-10-17

Compared main `5a39a926`, original PR `742d7205`, and integration `d9daa932`.
Each had its own native build: RTX 3090 24 GB, Python 3.11.15, JAX 0.9.0,
CUDA 13.3.73, SM86, fast math disabled. Runs were serial, with preallocation off.
Subsequent documentation/tooling cleanup does not change the measured solver.

Full notebooks ran without reduced documentation settings. Warm solver times
are medians of three repeats; they exclude plotting and CPU mode projection.

| Workload | Main streamed (s) | Merged streamed (s) | Merged mstep (s) |
|---|---:|---:|---:|
| Modal single profile | 2.452 | 2.476 | Unsupported nonuniform grid |
| Modal broadband | 2.469 | 2.490 | Unsupported nonuniform grid |
| Modal junction | 2.838 | 2.850 | Unsupported nonuniform grid |
| Cosine crossing | 19.593 | 19.648 | 22.376 |

All 37 modal exports match main bitwise. All 10 cosine exports match across
main, original PR and merged streamed/mstep, including raw fields and fluxes.
Cosine mstep is 13.9% slower than streamed, and 0.22% slower than original mstep.

Matched-grid trials used float32, a mode source, two three-frequency mode
monitors, CPML12 and 256 steps. Each variant ran in two fresh processes with
four warmups plus two warmup seconds, nine synchronized execution samples and
three public API samples. Shapes are `(z, y, x)`; GCUPS excludes preparation.

| Shape / material | Main streamed GCUPS | Merged streamed | Original mstep | Merged mstep |
|---|---:|---:|---:|---:|
| 48×96×128 / binary | 5.535 | 5.540 | 3.649 | 3.663 |
| 96×192×256 / smooth | 7.455 | 7.436 | 6.678 | 6.696 |
| 160×320×384 / binary | 8.691 | 8.641 | 9.912 | 9.921 |
| 512×256×96 / smooth | 7.403 | 7.403 | 4.998 | 4.997 |

All 40 processes passed: every element of all 30 state leaves matches its main
reference bitwise, with identical input/source/coefficient fingerprints. Four
runs generated references; 36 compared against them. The largest short-run DFT
signal is near zero, so the full notebooks supply the long-duration spectral
check. Small timing differences are observations, not statistical guarantees.
Driver/JIT caches were not cleared; peak memory was not compared.

CPU integration checks passed 321 tests, multistep GPU checks 129, and autodiff
GPU checks 40. Streamed hardware checks had 331 passes, 24 multi-GPU skips and
three failures reproduced on both parents: two smooth-material near-zero
mismatches and one conductive-case nonfinite mismatch. Tolerances are unchanged.
This is not a fully green hardware-suite claim or a fresh H100/adjoint benchmark.

## Modal-projection follow-up

Main commit `38479de4` expands the polarized candidate search from three to 11
modes. Compared with the original PR, 13 of 37 modal exports exceed the fixed
comparison tolerance, while raw fields, fluxes and source indices stay exact.
Changing only the candidate count reproduces each revision's amplitudes exactly.
This identifies basis selection as the cause; it is not independent accuracy
validation. Single full-notebook executions took 78.2 s on the original PR,
91.9 s on main and 91.0 s on the merge, with the increase in CPU analysis.
Keep the polarization-selection fix; reuse candidate solves as tracked in
[#343](https://github.com/beamzorg/beamz/issues/343).

## Historical studies and raw evidence

Detailed reports, samples, hashes, logs and the retired schedule-search tools
remain available at the immutable pre-cleanup commit below. These links preserve
the measured snapshots without carrying their generated data in the branch tip.

| Study | Finding and limits | Archived evidence |
|---|---|---|
| Futhark comparison | Based on Pepijn de Vos's [PR #308](https://github.com/beamzorg/beamz/pull/308): CUDA mstep won nine of ten matched RTX 3090 cases; the remaining case tied within 0.3%. Small/narrow cases still favored streamed. | [Report](https://github.com/beamzorg/beamz/blob/5c389a04b70b315dbd3d385aa09159d182810b05/docs/reviews/cuda-mstep-rtx3090.md), [data](https://github.com/beamzorg/beamz/blob/5c389a04b70b315dbd3d385aa09159d182810b05/docs/reviews/cuda-mstep-rtx3090.json.gz) |
| Schedule/startup search | 2,234 trials across 26 included workloads informed conservative selection. Startup can erase warm gains; the `main-anchors` records underpin the schedule/layout cost tables. This is not a universal optimum. | [Report and reproduction commands](https://github.com/beamzorg/beamz/blob/5c389a04b70b315dbd3d385aa09159d182810b05/docs/reviews/cuda-schedule-space.md), [data](https://github.com/beamzorg/beamz/blob/5c389a04b70b315dbd3d385aa09159d182810b05/docs/reviews/cuda-schedule-space.json.gz), [runner](https://github.com/beamzorg/beamz/blob/5c389a04b70b315dbd3d385aa09159d182810b05/scripts/benchmark_cuda_schedule_space.py), [analyzer](https://github.com/beamzorg/beamz/blob/5c389a04b70b315dbd3d385aa09159d182810b05/scripts/analyze_cuda_schedule_space.py) |
| H100 | Historical one/two/eight-GPU tests precede integration. Small cases had unresolved 1–2.5% slowdown candidates, a transient memory discrepancy and incomplete capacity coverage. Large capacity checks used sampled validation; release-source differences failed parity. Mstep remains single-GPU. | [Report](https://github.com/beamzorg/beamz/blob/5c389a04b70b315dbd3d385aa09159d182810b05/docs/reviews/cuda-h100-regression.md), [data](https://github.com/beamzorg/beamz/blob/5c389a04b70b315dbd3d385aa09159d182810b05/docs/reviews/cuda-h100-regression.json.gz) |
| Post-merge RTX 3090 | Full notebook and benchmark methods, individual samples, numerical comparisons and the modal diagnostic behind this summary. | [Report](https://github.com/beamzorg/beamz/blob/5c389a04b70b315dbd3d385aa09159d182810b05/docs/reviews/pr334-post-merge-validation-2026-10-17.md), [data](https://github.com/beamzorg/beamz/blob/5c389a04b70b315dbd3d385aa09159d182810b05/docs/reviews/pr334-post-merge-validation-2026-10-17.json.gz) |

## Reproduce

Build each checkout's matching native component and use fresh processes.
The retained runners support both streamed and mstep (uniform grids only).
Save a main reference, then compare the same workload on the PR checkout:

```console
python scripts/benchmark_cuda_revision.py --root /path/to/main --shape 160 320 384 --material binary --steps 256 --warmups 4 --warmup-seconds 2 --samples 9 --public-samples 3 --output /tmp/main.json --reference /tmp/new-reference --save-reference
python scripts/benchmark_cuda_revision.py --root /path/to/pr --shape 160 320 384 --material binary --steps 256 --warmups 4 --warmup-seconds 2 --samples 9 --public-samples 3 --output /tmp/pr.json --reference /tmp/new-reference
python scripts/validate_modal_notebook.py --root /path/to/pr --output /tmp/new-modal --notebook modal_sources_monitors --backend cuda_streamed --warm-runs 3 --expected-device "RTX 3090"
python scripts/validate_modal_notebook.py --root /path/to/pr --output /tmp/new-cosine --notebook cosine_waveguide_crossing --backend cuda_streamed --warm-runs 3 --expected-device "RTX 3090"
```

Use distinct output paths for each run and `--backend cuda_mstep` for explicit
multistep comparisons. Store new results under `.cache/` or as CI artifacts.
