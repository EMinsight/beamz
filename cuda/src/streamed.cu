#include "launch.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr float kEps0 = 8.8541878128e-12f;
constexpr float kMu0 = 1.25663706212e-6f;

__device__ __forceinline__ int64_t Elements(const BeamzBuffer& value) {
  if (value.rank == 0) return 1;
  int64_t count = 1;
  for (int axis = 0; axis < value.rank; ++axis) count *= value.dims[axis];
  return count;
}

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

__device__ __forceinline__ void Coordinates(const BeamzBuffer& value,
                                            int64_t linear, int* z, int* y,
                                            int* x) {
  *x = linear % value.dims[2];
  linear /= value.dims[2];
  *y = linear % value.dims[1];
  *z = linear / value.dims[1];
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
  return (Read(value, next_z, next_y, next_x) - Read(value, z, y, x)) *
         inv_dx;
}

__device__ __forceinline__ float BoundaryDifference(
    const BeamzBuffer& value, int axis, int z, int y, int x, int edge_mask,
    float inv_dx) {
  const int coordinate = axis == 0 ? z : (axis == 1 ? y : x);
  const int size = static_cast<int>(value.dims[axis]);
  if (coordinate == 0) {
    const bool metallic = edge_mask & (1 << (2 * axis));
    return metallic ? Read(value, z, y, x) * inv_dx : 0.0f;
  }
  if (coordinate == size) {
    int last_z = z, last_y = y, last_x = x;
    if (axis == 0) last_z = size - 1;
    if (axis == 1) last_y = size - 1;
    if (axis == 2) last_x = size - 1;
    const bool metallic = edge_mask & (1 << (2 * axis + 1));
    return metallic ? -Read(value, last_z, last_y, last_x) * inv_dx : 0.0f;
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

__global__ void UpdateComponent(BeamzLaunch launch, int component) {
  const BeamzBuffer& input = launch.inputs[component];
  const BeamzBuffer& output = launch.outputs[component];
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                         static_cast<int64_t>(threadIdx.x);
  if (linear >= Elements(output)) return;
  int z, y, x;
  Coordinates(output, linear, &z, &y, &x);
  const float inv_dx = 1.0f / launch.resolution;

  const int first_source[2][3] = {{2, 0, 1}, {2, 0, 1}};
  const int second_source[2][3] = {{1, 2, 0}, {1, 2, 0}};
  const int first_axis[2][3] = {{1, 0, 2}, {1, 0, 2}};
  const int second_axis[2][3] = {{0, 2, 1}, {0, 2, 1}};
  const BeamzBuffer& first = launch.inputs[3 + first_source[launch.phase][component]];
  const BeamzBuffer& second = launch.inputs[3 + second_source[launch.phase][component]];
  float derivative0;
  float derivative1;
  if (launch.phase == 0) {
    derivative0 = ForwardDifference(first, first_axis[0][component], z, y, x, inv_dx);
    derivative1 = ForwardDifference(second, second_axis[0][component], z, y, x, inv_dx);
  } else {
    derivative0 = BoundaryDifference(first, first_axis[1][component], z, y, x,
                                     launch.metallic_edges, inv_dx);
    derivative1 = BoundaryDifference(second, second_axis[1][component], z, y, x,
                                     launch.metallic_edges, inv_dx);
  }
  float curl;
  if (launch.nterms) {
    curl = CorrectCpml(derivative0, 2 * component, z, y, x, launch) +
           CorrectCpml(derivative1, 2 * component + 1, z, y, x, launch);
  } else {
    curl = derivative0 - derivative1;
  }

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

int BeamzLaunchStreamed(void* raw_stream, const BeamzLaunch& launch) {
  auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
  constexpr int threads = 256;
  for (int component = 0; component < 3; ++component) {
    int64_t elements = 1;
    const BeamzBuffer& output = launch.outputs[component];
    for (int axis = 0; axis < output.rank; ++axis) elements *= output.dims[axis];
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    UpdateComponent<<<blocks, threads, 0, stream>>>(launch, component);
  }
  return static_cast<int>(cudaPeekAtLastError());
}
