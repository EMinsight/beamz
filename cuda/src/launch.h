#ifndef BEAMZ_CUDA_LAUNCH_H_
#define BEAMZ_CUDA_LAUNCH_H_

#include <cstdint>

struct BeamzBuffer {
  void* data;
  int32_t rank;
  int64_t dims[3];
};

struct BeamzLaunch {
  int32_t abi_version;
  int32_t phase;
  int32_t nterms;
  float dt;
  float resolution;
  float inv_resolution;
  float dt_over_eps;
  float dt_over_mu;
  int32_t metallic_edges;
  BeamzBuffer inputs[37];
  BeamzBuffer outputs[9];
};

struct BeamzSourceLaunch {
  BeamzBuffer coefficient;
  BeamzBuffer waveform;
  BeamzBuffer current_step;
  int32_t component;
  int32_t starts[3];
};

// Returns zero after enqueueing all work, otherwise a CUDA runtime error code.
int BeamzLaunchStreamed(void* stream, const BeamzLaunch& launch);
int BeamzLaunchStreamedSteps(void* stream, const BeamzLaunch& h_launch,
                             const BeamzLaunch& e_launch, int32_t nsteps);
int BeamzLaunchStreamedSourceSteps(void* stream, const BeamzLaunch& h_launch,
                                   const BeamzLaunch& e_launch,
                                   const BeamzSourceLaunch& source,
                                   int32_t nsteps);
int BeamzLaunchHopper(void* stream, const BeamzLaunch& launch);

#endif  // BEAMZ_CUDA_LAUNCH_H_
