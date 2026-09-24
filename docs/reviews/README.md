# CUDA optimization evidence

Start with the [final random-domain comparison](cuda-random-final-comparison-2026-09-18.md)
and [automatic layout prediction](cuda-layout-prediction-2026-09-18.md).
The dated studies retain the earlier experiments, including unsuccessful ones;
their settings and measurements should not be confused with the final defaults.

Raw timing/validation JSON, CSV summaries, logs and experimental source patches
are included alongside the reports. Binary profiler captures and compressed
source snapshots remain local and are excluded from Git. References to captured
source archives in historical reports describe those local artifacts. Current
solver and benchmark sources are versioned in the repository.

FP32 remains the default. Consistent 9 GCUPS and complete CPML numerical parity
with origin/main have not been established. H100 numbers discussed so far are
estimates; the [RunPod setup](../../docker/runpod/H100.md) has not yet been tested
end to end on H100.
