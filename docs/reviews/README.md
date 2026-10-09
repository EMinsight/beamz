# CUDA optimization evidence

The current explicit column schedule is documented in the
[CUDA multistep benchmark report](cuda-mstep-rtx3090.md) and
[comparison with streamed CUDA](cuda-mstep-code-review.md). The report retains
the algorithm ideas and historical reference timings from PR #308.

Start with the [main-merge validation and notebook comparison](cuda-merge-validation-2026-09-24.md),
the [original random-domain comparison](cuda-random-final-comparison-2026-09-18.md),
and [automatic layout prediction](cuda-layout-prediction-2026-09-18.md).
The dated studies retain the earlier experiments, including unsuccessful ones;
their settings and measurements should not be confused with the final defaults.

Raw timing/validation JSON, CSV summaries, logs and experimental source patches
are included alongside the reports. Binary profiler captures and compressed
source snapshots remain local and are excluded from Git. References to captured
source archives in historical reports describe those local artifacts. Current
solver and benchmark sources are versioned in the repository.

FP32 remains the default. The September studies did not establish consistent
9 GCUPS and complete CPML numerical parity with origin/main. The October
multistep results above describe a separately validated explicit schedule. H100 numbers discussed so far are
estimates; the [RunPod setup](https://github.com/beamzorg/beamz/blob/fa151d28/docker/runpod/H100.md) has not yet been tested
end to end on H100.
