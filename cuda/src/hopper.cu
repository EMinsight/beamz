#include "launch.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace {

constexpr float kEps0 = 8.8541878128e-12f;
constexpr float kMu0 = 1.25663706212e-6f;
constexpr int kTileX = 32;
constexpr int kTileY = 4;
constexpr int kTileZ = 2;
constexpr int kAxis0Elements = (kTileZ + 2) * kTileY * kTileX;
constexpr int kAxis1Elements = kTileZ * (kTileY + 2) * kTileX;
constexpr int kAxis2Elements = kTileZ * kTileY * (kTileX + 2);
constexpr int kMaxSharedElements = kAxis0Elements;

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

__device__ __forceinline__ int DirectionalElements(int axis) {
  return axis == 0 ? kAxis0Elements
                   : (axis == 1 ? kAxis1Elements : kAxis2Elements);
}

__device__ __forceinline__ int DirectionalOffset(int axis, int z, int y,
                                                 int x) {
  if (axis == 0) return (z * kTileY + y) * kTileX + x;
  if (axis == 1) return (z * (kTileY + 2) + y) * kTileX + x;
  return (z * kTileY + y) * (kTileX + 2) + x;
}

__device__ __forceinline__ void StageDirectional(
    float* tile, const BeamzBuffer& source, int axis, int index, int base_z,
    int base_y, int base_x) {
  int value = index;
  const int width = axis == 2 ? kTileX + 2 : kTileX;
  const int height = axis == 1 ? kTileY + 2 : kTileY;
  const int local_x = value % width;
  value /= width;
  const int local_y = value % height;
  const int local_z = value / height;
  const int global_x = base_x + local_x - (axis == 2 ? 1 : 0);
  const int global_y = base_y + local_y - (axis == 1 ? 1 : 0);
  const int global_z = base_z + local_z - (axis == 0 ? 1 : 0);
  tile[index] = ReadChecked(source, global_z, global_y, global_x);
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
  int cz = lz + (axis == 0 ? 1 : 0);
  int cy = ly + (axis == 1 ? 1 : 0);
  int cx = lx + (axis == 2 ? 1 : 0);
  int nz = cz, ny = cy, nx = cx;
  if (axis == 0) ++nz;
  if (axis == 1) ++ny;
  if (axis == 2) ++nx;
  return (tile[DirectionalOffset(axis, nz, ny, nx)] -
          tile[DirectionalOffset(axis, cz, cy, cx)]) *
         inv_dx;
}

__device__ __forceinline__ float SharedBackward(const float* tile, int axis,
                                                int lz, int ly, int lx,
                                                float inv_dx) {
  int cz = lz + (axis == 0 ? 1 : 0);
  int cy = ly + (axis == 1 ? 1 : 0);
  int cx = lx + (axis == 2 ? 1 : 0);
  int pz = cz, py = cy, px = cx;
  if (axis == 0) --pz;
  if (axis == 1) --py;
  if (axis == 2) --px;
  return (tile[DirectionalOffset(axis, cz, cy, cx)] -
          tile[DirectionalOffset(axis, pz, py, px)]) *
         inv_dx;
}

__global__ __launch_bounds__(256, 2) void UpdateTiled(BeamzLaunch launch,
                                                       int component) {
  __shared__ float first_tile[kMaxSharedElements];
  __shared__ float second_tile[kMaxSharedElements];
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
  const int first_elements = DirectionalElements(first_axis);
  const int second_elements = DirectionalElements(second_axis);
  const int staged_elements =
      first_elements > second_elements ? first_elements : second_elements;
  for (int index = thread_linear; index < staged_elements;
       index += blockDim.x * blockDim.y * blockDim.z) {
    if (index < first_elements) {
      StageDirectional(first_tile, first, first_axis, index, base_z, base_y,
                       base_x);
    }
    if (index < second_elements) {
      StageDirectional(second_tile, second, second_axis, index, base_z, base_y,
                       base_x);
    }
  }
  __syncthreads();

  const BeamzBuffer& input = launch.inputs[component];
  const BeamzBuffer& output = launch.outputs[component];
  const int x = base_x + threadIdx.x;
  const int y = base_y + threadIdx.y;
  const int z = base_z + threadIdx.z;
  if (x >= output.dims[2] || y >= output.dims[1] || z >= output.dims[0]) return;
  const int lx = threadIdx.x;
  const int ly = threadIdx.y;
  const int lz = threadIdx.z;
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
