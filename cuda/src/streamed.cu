#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "launch.h"

namespace {

__device__ __forceinline__ int64_t Offset(const BeamzBuffer& value, int z,
                                          int y, int x) {
  if (value.rank == 0) return 0;
  const int iz = value.dims[0] == 1 ? 0 : z;
  const int iy = value.dims[1] == 1 ? 0 : y;
  const int ix = value.dims[2] == 1 ? 0 : x;
  return (static_cast<int64_t>(iz) * value.dims[1] + iy) * value.dims[2] + ix;
}

__device__ __forceinline__ float Read(const BeamzBuffer& value, int z, int y,
                                      int x) {
  return static_cast<const float*>(value.data)[Offset(value, z, y, x)];
}

__device__ __forceinline__ float Read3D(const BeamzBuffer& value, int z, int y,
                                        int x) {
  const int64_t offset =
      (static_cast<int64_t>(z) * value.dims[1] + y) * value.dims[2] + x;
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
  const int64_t psi_offset = Offset(psi_output, pz, py, px);
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
  const int64_t linear =
      (static_cast<int64_t>(z) * output.dims[1] + y) * output.dims[2] + x;
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

}  // namespace

int BeamzLaunchStreamed(void* raw_stream, const BeamzLaunch& launch) {
  auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
  constexpr int tile_x = 32;
  constexpr int tile_y = 4;
  constexpr int tile_z = 2;
  int64_t max_x = 0, max_y = 0, max_z = 0;
  for (int component = 0; component < 3; ++component) {
    const BeamzBuffer& output = launch.outputs[component];
    max_x = output.dims[2] > max_x ? output.dims[2] : max_x;
    max_y = output.dims[1] > max_y ? output.dims[1] : max_y;
    max_z = output.dims[0] > max_z ? output.dims[0] : max_z;
  }
  const int y_blocks = static_cast<int>((max_y + tile_y - 1) / tile_y);
  const dim3 threads(tile_x, tile_y, tile_z);
  const dim3 blocks((max_x + tile_x - 1) / tile_x, 3 * y_blocks,
                    (max_z + tile_z - 1) / tile_z);
  if (launch.phase == 0 && launch.nterms == 0) {
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
