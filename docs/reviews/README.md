# CUDA optimization evidence

- [CUDA component](../../cuda/README.md): API, shared architecture and selection limits.
- [Futhark baseline](cuda-mstep-rtx3090.md): Pepijn de Vos's algorithm ideas and the ten matched RTX3090 cases.
- [Tile/layout/startup study](cuda-schedule-space.md): measured configurations, time-to-result ranking and main compatibility.
- [H100 regression study](cuda-h100-regression.md): one/two/eight-GPU performance, memory, validation and remaining gaps.

Each study keeps its raw measurements, identities and exclusions in a compressed
JSON archive. The schedule analyzer reads its archive directly; generated CSV,
plot and table copies are omitted. Benchmark commit identifiers refer to the
original measured snapshots before history consolidation, not the current tip.
The three evidence archives are unchanged by that consolidation.

Earlier studies remain historical: [main-merge validation](cuda-merge-validation-2026-09-24.md),
[random domains](cuda-random-final-comparison-2026-09-18.md), and
[layout prediction](cuda-layout-prediction-2026-09-18.md). Their unsuccessful
experiments and settings should not be treated as current defaults.
Binary profiler captures/source bundles remain local and ignored; references
to them in older reports do not imply they are shipped in the repository.
