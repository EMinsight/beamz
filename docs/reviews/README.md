# CUDA optimization evidence

- [CUDA component](https://github.com/beamzorg/beamz/blob/feat/cuda-mstep/cuda/README.md): API, shared architecture and selection limits.
- [PR #334 post-merge validation](pr334-post-merge-validation-2026-10-17.md): full modal/cosine notebooks, matched RTX3090 benchmarks and the upstream modal-projection change.
- [Futhark baseline](cuda-mstep-rtx3090.md): Pepijn de Vos's algorithm ideas and the ten matched RTX3090 cases.
- [Tile/layout/startup study](cuda-schedule-space.md): measured configurations, time-to-result ranking and main compatibility.
- [H100 regression study](cuda-h100-regression.md): one/two/eight-GPU performance, memory, validation and remaining gaps.

Each study keeps its raw measurements, identities and exclusions in a compressed
JSON archive. The schedule analyzer reads its archive directly; generated CSV,
plot and table copies are omitted. Commit identifiers in the three historical
studies refer to the original measured snapshots before history consolidation,
not the current tip. Their three archives are unchanged by that consolidation.

Earlier studies remain historical: [main-merge validation](cuda-merge-validation-2026-09-24.md),
[random domains](cuda-random-final-comparison-2026-09-18.md), and
[layout prediction](cuda-layout-prediction-2026-09-18.md). Their unsuccessful
experiments and settings should not be treated as current defaults.
Binary profiler captures/source bundles remain local and ignored; references
to them in older reports do not imply they are shipped in the repository.
