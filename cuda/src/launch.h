#ifndef BEAMZ_CUDA_LAUNCH_H_
#define BEAMZ_CUDA_LAUNCH_H_

#include <cstdint>

enum BeamzElementType : int32_t {
  kBeamzF32 = 0,
  kBeamzS32 = 1,
  kBeamzBF16 = 2,
};

struct BeamzBuffer {
  void* data;
  int32_t rank;
  int32_t element_type;
  int64_t dims[4];
};

struct BeamzLaunch {
  int32_t abi_version;
  int32_t phase;
  int32_t nterms;
  int32_t metric_kind;
  float dt;
  float resolution;
  float inv_resolution;
  float dt_over_eps;
  float dt_over_mu;
  int32_t metallic_edges;
  BeamzBuffer inputs[37];
  BeamzBuffer metrics[3];
  BeamzBuffer outputs[9];
};

struct BeamzSourceLaunch {
  BeamzBuffer coefficient;
  BeamzBuffer waveform;
  BeamzBuffer current_step;
  int32_t component;
  int32_t starts[3];
};

struct BeamzSourceGroupLaunch {
  BeamzBuffer coefficients;
  BeamzBuffer waveforms;
  BeamzBuffer starts;
  BeamzBuffer current_step;
  int32_t component;
  int32_t timing;
};

struct BeamzDftLaunch {
  BeamzBuffer indices[6];
  BeamzBuffer weights[6];
  BeamzBuffer frequencies;
  BeamzBuffer component_mask;
  BeamzBuffer dft_re;
  BeamzBuffer dft_im;
  BeamzBuffer dft_weight;
  BeamzBuffer time;
  int32_t frequency_count;
  int32_t point_count;
};

struct BeamzDftGroupLaunch {
  BeamzBuffer indices;
  BeamzBuffer weights;
  BeamzBuffer frequencies;
  BeamzBuffer component_masks;
  BeamzBuffer counts;
  BeamzBuffer codes;
  BeamzBuffer windows;
  BeamzBuffer dft_re;
  BeamzBuffer dft_im;
  BeamzBuffer dft_weight;
  BeamzBuffer time;
  BeamzBuffer current_step;
  int32_t monitor_count;
};

// Returns zero after enqueueing all work, otherwise a CUDA runtime error code.
int BeamzLaunchStreamed(void* stream, const BeamzLaunch& launch);
int BeamzLaunchStreamedSteps(void* stream, const BeamzLaunch& h_launch,
                             const BeamzLaunch& e_launch, int32_t nsteps);
int BeamzLaunchTemporalSteps(void* stream, const BeamzLaunch& h_ab,
                             const BeamzLaunch& e_ab,
                             const BeamzLaunch& h_ba,
                             const BeamzLaunch& e_ba, int32_t nsteps);
int BeamzLaunchStreamedSourceSteps(void* stream, const BeamzLaunch& h_launch,
                                   const BeamzLaunch& e_launch,
                                   const BeamzSourceLaunch& source,
                                   int32_t nsteps);
int BeamzLaunchStreamedSourceGroupSteps(
    void* stream, const BeamzLaunch& h_launch, const BeamzLaunch& e_launch,
    const BeamzSourceGroupLaunch* source_groups, int32_t source_group_count,
    int32_t nsteps);
int BeamzLaunchStreamedProgramSteps(
    void* stream, const BeamzLaunch& h_launch, const BeamzLaunch& e_launch,
    const BeamzSourceGroupLaunch* source_groups, int32_t source_group_count,
    const BeamzDftGroupLaunch& monitors, int32_t nsteps);
int BeamzLaunchStreamedSourceMonitorSteps(
    void* stream, const BeamzLaunch& h_launch, const BeamzLaunch& e_launch,
    const BeamzSourceLaunch& source, const BeamzDftLaunch& monitor,
    int32_t nsteps);
int BeamzLaunchHopper(void* stream, const BeamzLaunch& launch);

#endif  // BEAMZ_CUDA_LAUNCH_H_
