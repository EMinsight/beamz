#ifndef BEAMZ_CUDA_GRAPH_H_
#define BEAMZ_CUDA_GRAPH_H_

#include <cuda_runtime_api.h>

#include <functional>
#include <string>

#include "launch.h"

// Build a stable cache identity from the semantic fields of a complete native
// program. C++ padding and inactive pointer values never define graph identity.
std::string BeamzGraphKey(const char* schedule, void* stream,
                          const BeamzProgramLaunch& program);

// Replay a cached executable or capture, instantiate, cache, and launch the
// supplied enqueue sequence. A nested-capture failure falls back to directly
// enqueueing the sequence into the caller's graph.
cudaError_t BeamzLaunchGraph(cudaStream_t stream, const std::string& key,
                             bool cache_enabled,
                             const std::function<cudaError_t()>& enqueue);

#endif  // BEAMZ_CUDA_GRAPH_H_
