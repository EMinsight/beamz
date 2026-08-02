#include "launch.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace {

constexpr float kEps0 = 8.8541878128e-12f;
constexpr float kMu0 = 1.25663706212e-6f;
constexpr int kTileX = 32;
constexpr int kTileY = 4;
constexpr int kTileZ = 2;
constexpr int kSharedX = kTileX + 2;
constexpr int kSharedY = kTileY + 2;
constexpr int kSharedZ = kTileZ + 2;
constexpr int kSharedElements = kSharedX * kSharedY * kSharedZ;

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

__device__ __forceinline__ float ReadChecked(const BeamzBuffer& value, int z,
                                             int y, int x) {
  if (z < 0 || y < 0 || x < 0 || z >= value.dims[0] || y >= value.dims[1] ||
      x >= value.dims[2]) {
    return 0.0f;
  }
  return Read(value, z, y, x);
}

__device__ __forceinline__ int SharedOffset(int z, int y, int x) {
  return (z * kSharedY + y) * kSharedX + x;
}

__device__ __forceinline__ float BoundaryDifference(
    const BeamzBuffer& value, int axis, int z, int y, int x, int edge_mask,
    float inv_dx) {
  const int coordinate = axis == 0 ? z : (axis == 1 ? y : x);
  const int size = static_cast<int>(value.dims[axis]);
  if (coordinate == 0) {
    return edge_mask & (1 << (2 * axis)) ? Read(value, z, y, x) * inv_dx
                                         : 0.0f;
  }
  if (coordinate == size) {
    int last_z = z, last_y = y, last_x = x;
    if (axis == 0) last_z = size - 1;
    if (axis == 1) last_y = size - 1;
    if (axis == 2) last_x = size - 1;
    return edge_mask & (1 << (2 * axis + 1))
               ? -Read(value, last_z, last_y, last_x) * inv_dx
               : 0.0f;
  }
  int low_z = z, low_y = y, low_x = x;
  if (axis == 0) --low_z;
  if (axis == 1) --low_y;
  if (axis == 2) --low_x;
  return (Read(value, z, y, x) - Read(value, low_z, low_y, low_x)) *
         inv_dx;
}

__device__ __forceinline__ float CorrectCpml(
    float derivative, int term, int z, int y, int x,
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

__device__ __forceinline__ void DerivativePlan(int component, int* first_source,
                                               int* second_source,
                                               int* first_axis,
                                               int* second_axis) {
  if (component == 0) {
    *first_source = 2;
    *second_source = 1;
    *first_axis = 1;
    *second_axis = 0;
  } else if (component == 1) {
    *first_source = 0;
    *second_source = 2;
    *first_axis = 0;
    *second_axis = 2;
  } else {
    *first_source = 1;
    *second_source = 0;
    *first_axis = 2;
    *second_axis = 1;
  }
}

__device__ __forceinline__ float SharedForward(const float* tile, int axis,
                                               int lz, int ly, int lx,
                                               float inv_dx) {
  int nz = lz, ny = ly, nx = lx;
  if (axis == 0) ++nz;
  if (axis == 1) ++ny;
  if (axis == 2) ++nx;
  return (tile[SharedOffset(nz, ny, nx)] -
          tile[SharedOffset(lz, ly, lx)]) *
         inv_dx;
}

__device__ __forceinline__ float SharedBackward(const float* tile, int axis,
                                                int lz, int ly, int lx,
                                                float inv_dx) {
  int pz = lz, py = ly, px = lx;
  if (axis == 0) --pz;
  if (axis == 1) --py;
  if (axis == 2) --px;
  return (tile[SharedOffset(lz, ly, lx)] -
          tile[SharedOffset(pz, py, px)]) *
         inv_dx;
}

__global__ __launch_bounds__(256, 2) void UpdateTiled(BeamzLaunch launch,
                                                       int component) {
  __shared__ float first_tile[kSharedElements];
  __shared__ float second_tile[kSharedElements];
  int first_source, second_source, first_axis, second_axis;
  DerivativePlan(component, &first_source, &second_source, &first_axis,
                 &second_axis);
  const BeamzBuffer& first = launch.inputs[3 + first_source];
  const BeamzBuffer& second = launch.inputs[3 + second_source];
  const int base_x = blockIdx.x * kTileX;
  const int base_y = blockIdx.y * kTileY;
  const int base_z = blockIdx.z * kTileZ;
  const int thread_linear =
      (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  for (int index = thread_linear; index < kSharedElements;
       index += blockDim.x * blockDim.y * blockDim.z) {
    int value = index;
    const int local_x = value % kSharedX;
    value /= kSharedX;
    const int local_y = value % kSharedY;
    const int local_z = value / kSharedY;
    const int global_x = base_x + local_x - 1;
    const int global_y = base_y + local_y - 1;
    const int global_z = base_z + local_z - 1;
    first_tile[index] = ReadChecked(first, global_z, global_y, global_x);
    second_tile[index] = ReadChecked(second, global_z, global_y, global_x);
  }
  __syncthreads();

  const BeamzBuffer& input = launch.inputs[component];
  const BeamzBuffer& output = launch.outputs[component];
  const int x = base_x + threadIdx.x;
  const int y = base_y + threadIdx.y;
  const int z = base_z + threadIdx.z;
  if (x >= output.dims[2] || y >= output.dims[1] || z >= output.dims[0]) return;
  const int lx = threadIdx.x + 1;
  const int ly = threadIdx.y + 1;
  const int lz = threadIdx.z + 1;
  const float inv_dx = 1.0f / launch.resolution;
  float derivative0;
  float derivative1;
  if (launch.phase == 0) {
    const int c0 = first_axis == 0 ? z : (first_axis == 1 ? y : x);
    const int c1 = second_axis == 0 ? z : (second_axis == 1 ? y : x);
    derivative0 = c0 + 1 < first.dims[first_axis]
                      ? SharedForward(first_tile, first_axis, lz, ly, lx, inv_dx)
                      : 0.0f;
    derivative1 = c1 + 1 < second.dims[second_axis]
                      ? SharedForward(second_tile, second_axis, lz, ly, lx, inv_dx)
                      : 0.0f;
  } else {
    const int c0 = first_axis == 0 ? z : (first_axis == 1 ? y : x);
    const int c1 = second_axis == 0 ? z : (second_axis == 1 ? y : x);
    derivative0 = c0 > 0 && c0 < first.dims[first_axis]
                      ? SharedBackward(first_tile, first_axis, lz, ly, lx, inv_dx)
                      : BoundaryDifference(first, first_axis, z, y, x,
                                           launch.metallic_edges, inv_dx);
    derivative1 = c1 > 0 && c1 < second.dims[second_axis]
                      ? SharedBackward(second_tile, second_axis, lz, ly, lx, inv_dx)
                      : BoundaryDifference(second, second_axis, z, y, x,
                                           launch.metallic_edges, inv_dx);
  }
  const float curl =
      launch.nterms
          ? CorrectCpml(derivative0, 2 * component, z, y, x, launch) +
                CorrectCpml(derivative1, 2 * component + 1, z, y, x, launch)
          : derivative0 - derivative1;
  const int64_t linear = Offset(output, z, y, x);
  const float old_field = static_cast<const float*>(input.data)[linear];
  if (launch.phase == 0) {
    const float sigma = Read(launch.inputs[6 + component], z, y, x);
    const float alpha = sigma * (0.5f * launch.dt / kMu0);
    static_cast<float*>(output.data)[linear] =
        ((1.0f - alpha) * old_field - (launch.dt / kMu0) * curl) /
        (1.0f + alpha);
  } else {
    const float conductivity = Read(launch.inputs[6 + component], z, y, x);
    const float inverse_permittivity =
        1.0f / Read(launch.inputs[9 + component], z, y, x);
    const float beta = conductivity * (0.5f * launch.dt / kEps0) *
                       inverse_permittivity;
    static_cast<float*>(output.data)[linear] =
        ((1.0f - beta) * old_field +
         (launch.dt / kEps0) * inverse_permittivity * curl) /
        (1.0f + beta);
  }
}

}  // namespace

int BeamzLaunchHopper(void* raw_stream, const BeamzLaunch& launch) {
  auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
  const dim3 threads(kTileX, kTileY, kTileZ);
  for (int component = 0; component < 3; ++component) {
    const BeamzBuffer& output = launch.outputs[component];
    const dim3 blocks((output.dims[2] + kTileX - 1) / kTileX,
                      (output.dims[1] + kTileY - 1) / kTileY,
                      (output.dims[0] + kTileZ - 1) / kTileZ);
    UpdateTiled<<<blocks, threads, 0, stream>>>(launch, component);
  }
  return static_cast<int>(cudaPeekAtLastError());
}
