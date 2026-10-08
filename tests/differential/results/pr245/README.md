# PR245 passive-SOI comparison

The MMI, mode converter, and polarization splitter-rotator use the pinned
geometries from [Liu and Poon](https://arxiv.org/abs/2506.16665).
The latest completed runs used a local RTX3090, float32 fields, and
`cuda_streamed`. The table compares target-mode power at exactly 1550 nm with
a 20 nm source bandwidth. Published values come from the case manifests;
some are approximate figure digitizations. Equal PPW does not imply an
identical realized mesh.

| Device / target mode | PPW | Published Lumerical | Published Tidy3D | BeamZ |
|---|---:|---:|---:|---:|
| MMI cross TE0 | 6 | 37.6% | 35.8% | 37.43% |
| MMI cross TE0 | 10 | 45.9% | 46.0% | 45.98% |
| MMI cross TE0 | 15 | 48.3% | 47.9% | 48.03% |
| MMI cross TE0 | 20 | 48.6% | 48.4% | 48.53% |
| Converter TE1 | 6 | 96.7% | 35.7% | 21.15% |
| Converter TE1 | 10 | 51.2% | 26.0% | 28.96% |
| Converter TE1 | 15 | 44.3% | 49.2% | 45.94% |
| PSR TE0 | 6 | 14.7% | 5.1% | 85.24% |
| PSR TE0 | 10 | 0.0% | 62.0% | 89.19% |
| PSR TE0 | 15 | 94.0% | 90.1% | 91.21% |

All ten runs pass the unchanged `1e-5` field-decay gate and `1.02` selected-output
bound at their six retained frequencies. These checks cover selected modes,
not a complete guided/radiated energy balance.

- MMI at 20 PPW agrees with the published 48.4–48.6% range. The change from
  15 PPW is 0.502 percentage points.
- Converter at 15 PPW lies inside the published 44.3–49.2% range, but its
  10→15 PPW change is 16.979 percentage points. Further refinement is needed.
- PSR at 15 PPW lies inside the published 90.1–94.0% range. It has not reached
  the 20-PPW resolution required for its converged-reference comparison.

These observations do not demonstrate complete paper reproduction or mesh
convergence. BeamZ uses fixed material indices evaluated at 1550 nm, while the
commercial references use fitted dispersive models.

## Evidence

[results.json](results.json) retains the ten configurations, executing commits,
raw-field and source-record hashes, decay/power checks, reference comparisons,
runtime and memory measurements. Each record links one compact NPZ in
[spectra/](spectra/) containing complex S parameters, incident power, modal
and flux diagnostics, and realized grid edges. The nonempty PSR source patch
is retained in [source-patches/](source-patches/). Per-run executing revisions
and patches describe the observations; cleanup does not constitute a fresh
FDTD run. Larger raw fields remain at the local paths recorded in the JSON.
[environment.json](environment.json) records packages, native binary hashes,
and hardware.

The matched 15-PPW MMI material-preparation control retains an identical
complete raw-monitor hash before and after reuse. Peak JAX allocation decreased
from 6.32 to 4.79 GiB; host peak RSS did not improve. These are descriptive
single samples, and JAX statistics omit native CUDA allocations. The paired
record is included in `results.json`.

Cleanup validation: **62 focused CPU checks** and **7 RTX3090 JAX/CUDA parity
checks** pass. All ten retained spectra have verified hashes, finite arrays,
valid incident signals, and center powers matching their complex S parameters.
Source-patch hashes, report links, Ruff, and whitespace checks also pass.
These are focused checks, not a new full-suite or benchmark campaign.

## Reproduction

Install the test and GDS extras and build the optional CUDA component as
explained in [cuda/README.md](../../../../cuda/README.md). Use a fresh output
path for each run:

```sh
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  -m scripts.investigate_passive_soi mmi2x2 --ppw 20 \
  --backend cuda_streamed --exact-center \
  --output validation-artifacts/pr245-repeat/mmi-20
```

Substitute `mode_converter --ppw 15` or
`polarization_splitter_rotator --ppw 15` for the other refined cases.
Omit `--exact-center` when reproducing the default hardware test's original
five-frequency sampling. Use `--backend jax` for the JAX execution path.
No geometry variant or acceptance threshold is changed by these commands.

AI assistance: OpenAI Codex implemented the benchmark follow-up, ran the
recorded experiments, and consolidated this report.
