#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>

#include "launch.h"

namespace {

constexpr int kTileX = 32;
constexpr int kTileY = 4;
constexpr int kTileZ = 2;
constexpr size_t kMaxCachedGraphs = 32;

struct GraphCache {
  std::mutex mutex;
  std::unordered_map<std::string, cudaGraphExec_t> entries;
};

GraphCache& CachedGraphs() {
  // Deliberately retain executable handles until process teardown. CUDA contexts may
  // already be gone when static destructors run, and the bounded cache is tiny.
  static auto* cache = new GraphCache();
  return *cache;
}

bool GraphCacheEnabled() {
  const char* value = std::getenv("BEAMZ_CUDA_DISABLE_GRAPH_CACHE");
  return value == nullptr || value[0] == '\0' || value[0] == '0';
}

std::string GraphKey(void* stream, const BeamzLaunch& h_launch,
                     const BeamzLaunch& e_launch, int32_t nsteps) {
  std::string key;
  key.reserve(sizeof(stream) + sizeof(nsteps) + 2 * sizeof(BeamzLaunch));
  key.append(reinterpret_cast<const char*>(&stream), sizeof(stream));
  key.append(reinterpret_cast<const char*>(&nsteps), sizeof(nsteps));
  key.append(reinterpret_cast<const char*>(&h_launch), sizeof(h_launch));
  key.append(reinterpret_cast<const char*>(&e_launch), sizeof(e_launch));
  return key;
}

bool FitsIntOffsets(const BeamzBuffer& value) {
  int64_t elements = 1;
  for (int axis = 0; axis < value.rank; ++axis) {
    if (value.dims[axis] > std::numeric_limits<int>::max()) return false;
    elements *= value.dims[axis];
    if (elements > std::numeric_limits<int>::max()) return false;
  }
  return true;
}

__device__ __forceinline__ int Offset(const BeamzBuffer& value, int z, int y,
                                      int x) {
  if (value.rank == 0) return 0;
  const int iz = value.dims[0] == 1 ? 0 : z;
  const int iy = value.dims[1] == 1 ? 0 : y;
  const int ix = value.dims[2] == 1 ? 0 : x;
  return (iz * static_cast<int>(value.dims[1]) + iy) *
             static_cast<int>(value.dims[2]) +
         ix;
}

__device__ __forceinline__ float Read(const BeamzBuffer& value, int z, int y,
                                      int x) {
  return static_cast<const float*>(value.data)[Offset(value, z, y, x)];
}

__device__ __forceinline__ float Read3D(const BeamzBuffer& value, int z, int y,
                                        int x) {
  const int offset = (z * static_cast<int>(value.dims[1]) + y) *
                         static_cast<int>(value.dims[2]) +
                     x;
  return static_cast<const float*>(value.data)[offset];
}

__device__ __forceinline__ float ForwardDifference(const BeamzBuffer& value,
                                                   int axis, int z, int y,
                                                   int x, float inv_dx) {
  int next_z = z, next_y = y, next_x = x;
  if (axis == 0) {
    if (z + 1 >= value.dims[0]) return 0.0f;
    ++next_z;
  } else if (axis == 1) {
    if (y + 1 >= value.dims[1]) return 0.0f;
    ++next_y;
  } else {
    if (x + 1 >= value.dims[2]) return 0.0f;
    ++next_x;
  }
  return (Read3D(value, next_z, next_y, next_x) - Read3D(value, z, y, x)) *
         inv_dx;
}

__device__ __forceinline__ float BoundaryDifference(const BeamzBuffer& value,
                                                    int axis, int z, int y,
                                                    int x, int edge_mask,
                                                    float inv_dx) {
  const int coordinate = axis == 0 ? z : (axis == 1 ? y : x);
  const int size = static_cast<int>(value.dims[axis]);
  if (coordinate == 0) {
    const bool metallic = edge_mask & (1 << (2 * axis));
    return metallic ? Read3D(value, z, y, x) * inv_dx : 0.0f;
  }
  if (coordinate == size) {
    int last_z = z, last_y = y, last_x = x;
    if (axis == 0) last_z = size - 1;
    if (axis == 1) last_y = size - 1;
    if (axis == 2) last_x = size - 1;
    const bool metallic = edge_mask & (1 << (2 * axis + 1));
    return metallic ? -Read3D(value, last_z, last_y, last_x) * inv_dx : 0.0f;
  }
  int low_z = z, low_y = y, low_x = x;
  if (axis == 0) --low_z;
  if (axis == 1) --low_y;
  if (axis == 2) --low_x;
  return (Read3D(value, z, y, x) - Read3D(value, low_z, low_y, low_x)) * inv_dx;
}

__device__ __forceinline__ float CorrectCpml(float derivative, int term, int z,
                                             int y, int x,
                                             const BeamzLaunch& launch) {
  const auto* descriptor = static_cast<const int32_t*>(launch.inputs[12].data);
  const int axis = descriptor[term * 5 + 1];
  const int low = descriptor[term * 5 + 2];
  const int high = descriptor[term * 5 + 3];
  const float sign = static_cast<float>(descriptor[term * 5 + 4]);
  const BeamzBuffer& target = launch.outputs[term / 2];
  const int coordinate = axis == 0 ? z : (axis == 1 ? y : x);
  const int axis_size = static_cast<int>(target.dims[axis]);
  int packed = -1;
  if (coordinate < low) {
    packed = coordinate;
  } else if (coordinate >= axis_size - high) {
    packed = low + coordinate - (axis_size - high);
  }
  if (packed < 0) return sign * derivative;

  int pz = z, py = y, px = x;
  if (axis == 0) pz = packed;
  if (axis == 1) py = packed;
  if (axis == 2) px = packed;
  const int coefficient_base = 13 + 3 * term;
  const int psi_base = 13 + 3 * launch.nterms;
  const BeamzBuffer& psi_input = launch.inputs[psi_base + term];
  const BeamzBuffer& psi_output = launch.outputs[3 + term];
  const int psi_offset = Offset(psi_output, pz, py, px);
  const float old_psi = static_cast<const float*>(psi_input.data)[psi_offset];
  const float next_psi =
      Read(launch.inputs[coefficient_base + 1], pz, py, px) * old_psi +
      Read(launch.inputs[coefficient_base], pz, py, px) * derivative;
  static_cast<float*>(psi_output.data)[psi_offset] = next_psi;
  return sign *
         (derivative * Read(launch.inputs[coefficient_base + 2], pz, py, px) +
          next_psi);
}

template <int Phase, int Component, bool Cpml>
__device__ __forceinline__ void UpdateComponent(const BeamzLaunch& launch,
                                                int z, int y, int x) {
  const BeamzBuffer& input = launch.inputs[Component];
  const BeamzBuffer& output = launch.outputs[Component];
  if (z >= output.dims[0] || y >= output.dims[1] || x >= output.dims[2]) return;
  const int linear = (z * static_cast<int>(output.dims[1]) + y) *
                         static_cast<int>(output.dims[2]) +
                     x;
  constexpr int normal_axis = 2 - Component;
  constexpr bool constrained = Phase == 0;
  const int coordinate = normal_axis == 0 ? z : (normal_axis == 1 ? y : x);
  const int axis_size = static_cast<int>(output.dims[normal_axis]);
  const bool on_low_wall =
      coordinate == 0 && (launch.metallic_edges & (1 << (2 * normal_axis)));
  const bool on_high_wall =
      coordinate == axis_size - 1 &&
      (launch.metallic_edges & (1 << (2 * normal_axis + 1)));
  if constexpr (constrained) {
    if (on_low_wall || on_high_wall) {
      static_cast<float*>(output.data)[linear] = 0.0f;
      return;
    }
  } else {
    for (int axis = 0; axis < 3; ++axis) {
      if (axis == normal_axis) continue;
      const int axis_coordinate = axis == 0 ? z : (axis == 1 ? y : x);
      const int size = static_cast<int>(output.dims[axis]);
      if ((axis_coordinate == 0 &&
           (launch.metallic_edges & (1 << (2 * axis)))) ||
          (axis_coordinate == size - 1 &&
           (launch.metallic_edges & (1 << (2 * axis + 1))))) {
        static_cast<float*>(output.data)[linear] = 0.0f;
        return;
      }
    }
  }
  const float inv_dx = launch.inv_resolution;

  constexpr int first_source[3] = {2, 0, 1};
  constexpr int second_source[3] = {1, 2, 0};
  constexpr int first_axis[3] = {1, 0, 2};
  constexpr int second_axis[3] = {0, 2, 1};
  const BeamzBuffer& first = launch.inputs[3 + first_source[Component]];
  const BeamzBuffer& second = launch.inputs[3 + second_source[Component]];
  float derivative0;
  float derivative1;
  if constexpr (Phase == 0) {
    derivative0 =
        ForwardDifference(first, first_axis[Component], z, y, x, inv_dx);
    derivative1 =
        ForwardDifference(second, second_axis[Component], z, y, x, inv_dx);
  } else {
    derivative0 = BoundaryDifference(first, first_axis[Component], z, y, x,
                                     launch.metallic_edges, inv_dx);
    derivative1 = BoundaryDifference(second, second_axis[Component], z, y, x,
                                     launch.metallic_edges, inv_dx);
  }
  float curl;
  if constexpr (Cpml) {
    curl = CorrectCpml(derivative0, 2 * Component, z, y, x, launch) +
           CorrectCpml(derivative1, 2 * Component + 1, z, y, x, launch);
  } else {
    curl = derivative0 - derivative1;
  }

  const float old_field = static_cast<const float*>(input.data)[linear];
  const float decay = Read(launch.inputs[6 + Component], z, y, x);
  const float source = Read(launch.inputs[9 + Component], z, y, x);
  if constexpr (Phase == 0) {
    static_cast<float*>(output.data)[linear] =
        decay * old_field - source * curl;
  } else {
    static_cast<float*>(output.data)[linear] =
        decay * old_field + source * curl;
  }
}

template <int Phase, bool Cpml>
__global__ void UpdateAllComponents(BeamzLaunch launch, int y_blocks) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (blockIdx.y < y_blocks) {
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    UpdateComponent<Phase, 0, Cpml>(launch, z, y, x);
  } else if (blockIdx.y < 2 * y_blocks) {
    const int y = (blockIdx.y - y_blocks) * blockDim.y + threadIdx.y;
    UpdateComponent<Phase, 1, Cpml>(launch, z, y, x);
  } else {
    const int y = (blockIdx.y - 2 * y_blocks) * blockDim.y + threadIdx.y;
    UpdateComponent<Phase, 2, Cpml>(launch, z, y, x);
  }
}

template <int Phase>
__device__ __forceinline__ float UncheckedDifference(
    const BeamzBuffer& value, int axis, int z, int y, int x, float inv_dx) {
  int neighbor_z = z, neighbor_y = y, neighbor_x = x;
  if (axis == 0) neighbor_z += Phase == 0 ? 1 : -1;
  if (axis == 1) neighbor_y += Phase == 0 ? 1 : -1;
  if (axis == 2) neighbor_x += Phase == 0 ? 1 : -1;
  const float center = Read3D(value, z, y, x);
  const float neighbor = Read3D(value, neighbor_z, neighbor_y, neighbor_x);
  return (Phase == 0 ? neighbor - center : center - neighbor) * inv_dx;
}

template <int Phase, int Component>
__device__ __forceinline__ void UpdateFullPecScalarComponent(
    const BeamzLaunch& launch, int z, int y, int x) {
  const BeamzBuffer& input = launch.inputs[Component];
  const BeamzBuffer& output = launch.outputs[Component];
  if (z >= output.dims[0] || y >= output.dims[1] || x >= output.dims[2]) return;
  const int linear = (z * static_cast<int>(output.dims[1]) + y) *
                         static_cast<int>(output.dims[2]) +
                     x;
  constexpr int normal_axis = 2 - Component;
  if constexpr (Phase == 0) {
    const int coordinate = normal_axis == 0 ? z : (normal_axis == 1 ? y : x);
    if (coordinate == 0 || coordinate == output.dims[normal_axis] - 1) {
      static_cast<float*>(output.data)[linear] = 0.0f;
      return;
    }
  } else {
    for (int axis = 0; axis < 3; ++axis) {
      if (axis == normal_axis) continue;
      const int coordinate = axis == 0 ? z : (axis == 1 ? y : x);
      if (coordinate == 0 || coordinate == output.dims[axis] - 1) {
        static_cast<float*>(output.data)[linear] = 0.0f;
        return;
      }
    }
  }

  constexpr int first_source[3] = {2, 0, 1};
  constexpr int second_source[3] = {1, 2, 0};
  constexpr int first_axis[3] = {1, 0, 2};
  constexpr int second_axis[3] = {0, 2, 1};
  const float derivative0 = UncheckedDifference<Phase>(
      launch.inputs[3 + first_source[Component]], first_axis[Component], z, y,
      x, launch.inv_resolution);
  const float derivative1 = UncheckedDifference<Phase>(
      launch.inputs[3 + second_source[Component]], second_axis[Component], z,
      y, x, launch.inv_resolution);
  const float old_field = static_cast<const float*>(input.data)[linear];
  const float decay =
      static_cast<const float*>(launch.inputs[6 + Component].data)[0];
  const float source =
      static_cast<const float*>(launch.inputs[9 + Component].data)[0];
  const float curl = derivative0 - derivative1;
  static_cast<float*>(output.data)[linear] =
      Phase == 0 ? decay * old_field - source * curl
                 : decay * old_field + source * curl;
}

template <int Phase>
__global__ void UpdateAllFullPecScalarComponents(BeamzLaunch launch,
                                                 int y_blocks) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (blockIdx.y < y_blocks) {
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    UpdateFullPecScalarComponent<Phase, 0>(launch, z, y, x);
  } else if (blockIdx.y < 2 * y_blocks) {
    const int y = (blockIdx.y - y_blocks) * blockDim.y + threadIdx.y;
    UpdateFullPecScalarComponent<Phase, 1>(launch, z, y, x);
  } else {
    const int y = (blockIdx.y - 2 * y_blocks) * blockDim.y + threadIdx.y;
    UpdateFullPecScalarComponent<Phase, 2>(launch, z, y, x);
  }
}

}  // namespace

int BeamzLaunchStreamed(void* raw_stream, const BeamzLaunch& launch) {
  const int input_count = launch.nterms == 0 ? 12 : 13 + 4 * launch.nterms;
  const int output_count = 3 + launch.nterms;
  for (int index = 0; index < input_count; ++index) {
    if (!FitsIntOffsets(launch.inputs[index])) return cudaErrorInvalidValue;
  }
  for (int index = 0; index < output_count; ++index) {
    if (!FitsIntOffsets(launch.outputs[index])) return cudaErrorInvalidValue;
  }
  auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
  int64_t max_x = 0, max_y = 0, max_z = 0;
  for (int component = 0; component < 3; ++component) {
    const BeamzBuffer& output = launch.outputs[component];
    max_x = output.dims[2] > max_x ? output.dims[2] : max_x;
    max_y = output.dims[1] > max_y ? output.dims[1] : max_y;
    max_z = output.dims[0] > max_z ? output.dims[0] : max_z;
  }
  const int y_blocks = static_cast<int>((max_y + kTileY - 1) / kTileY);
  const dim3 threads(kTileX, kTileY, kTileZ);
  const dim3 blocks((max_x + kTileX - 1) / kTileX, 3 * y_blocks,
                    (max_z + kTileZ - 1) / kTileZ);
  const bool scalar_coefficients =
      launch.nterms == 0 && launch.inputs[6].rank == 0 &&
      launch.inputs[7].rank == 0 && launch.inputs[8].rank == 0 &&
      launch.inputs[9].rank == 0 && launch.inputs[10].rank == 0 &&
      launch.inputs[11].rank == 0;
  if (scalar_coefficients && launch.metallic_edges == 63) {
    if (launch.phase == 0) {
      UpdateAllFullPecScalarComponents<0>
          <<<blocks, threads, 0, stream>>>(launch, y_blocks);
    } else {
      UpdateAllFullPecScalarComponents<1>
          <<<blocks, threads, 0, stream>>>(launch, y_blocks);
    }
  } else if (launch.phase == 0 && launch.nterms == 0) {
    UpdateAllComponents<0, false>
        <<<blocks, threads, 0, stream>>>(launch, y_blocks);
  } else if (launch.phase == 0) {
    UpdateAllComponents<0, true>
        <<<blocks, threads, 0, stream>>>(launch, y_blocks);
  } else if (launch.nterms == 0) {
    UpdateAllComponents<1, false>
        <<<blocks, threads, 0, stream>>>(launch, y_blocks);
  } else {
    UpdateAllComponents<1, true>
        <<<blocks, threads, 0, stream>>>(launch, y_blocks);
  }
  return static_cast<int>(cudaPeekAtLastError());
}

int BeamzLaunchStreamedSteps(void* raw_stream, const BeamzLaunch& h_launch,
                             const BeamzLaunch& e_launch, int32_t nsteps) {
  if (nsteps < 1 || h_launch.phase != 0 || e_launch.phase != 1 ||
      h_launch.nterms != e_launch.nterms ||
      (h_launch.nterms != 0 && h_launch.nterms != 6)) {
    return cudaErrorInvalidValue;
  }
  auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
  const std::string graph_key =
      GraphKey(raw_stream, h_launch, e_launch, nsteps);
  GraphCache& cache = CachedGraphs();
  if (GraphCacheEnabled()) {
    std::lock_guard<std::mutex> lock(cache.mutex);
    const auto cached = cache.entries.find(graph_key);
    if (cached != cache.entries.end()) {
      return static_cast<int>(cudaGraphLaunch(cached->second, stream));
    }
  }
  auto launch_steps = [&]() {
    cudaError_t launch_error = cudaSuccess;
    for (int32_t step = 0; step < nsteps; ++step) {
      launch_error =
          static_cast<cudaError_t>(BeamzLaunchStreamed(raw_stream, h_launch));
      if (launch_error != cudaSuccess) break;
      launch_error =
          static_cast<cudaError_t>(BeamzLaunchStreamed(raw_stream, e_launch));
      if (launch_error != cudaSuccess) break;
    }
    return launch_error;
  };
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  cudaError_t error =
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  // Nested capture is not supported. If XLA already owns capture on this stream,
  // enqueue the kernels directly so they become part of the outer graph.
  if (error != cudaSuccess) {
    (void)cudaGetLastError();
    return launch_steps();
  }
  error = launch_steps();
  cudaError_t end_error = cudaStreamEndCapture(stream, &graph);
  if (error == cudaSuccess) error = end_error;
  if (error == cudaSuccess) error = cudaGraphInstantiate(&executable, graph, 0);
  if (error == cudaSuccess && GraphCacheEnabled()) {
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (cache.entries.size() >= kMaxCachedGraphs) {
      for (const auto& [unused_key, cached] : cache.entries) {
        (void)unused_key;
        cudaGraphExecDestroy(cached);
      }
      cache.entries.clear();
    }
    const auto [entry, inserted] = cache.entries.emplace(graph_key, executable);
    if (!inserted) {
      cudaGraphExecDestroy(executable);
      executable = entry->second;
    }
  }
  if (error == cudaSuccess) error = cudaGraphLaunch(executable, stream);
  if (!GraphCacheEnabled() && executable != nullptr) {
    cudaGraphExecDestroy(executable);
  }
  if (graph != nullptr) cudaGraphDestroy(graph);
  return error;
}
