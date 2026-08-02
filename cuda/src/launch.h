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
  int32_t metallic_edges;
  BeamzBuffer inputs[37];
  BeamzBuffer outputs[9];
};

// Returns zero after enqueueing all work, otherwise a CUDA runtime error code.
int BeamzLaunchStreamed(void* stream, const BeamzLaunch& launch);

#endif  // BEAMZ_CUDA_LAUNCH_H_
