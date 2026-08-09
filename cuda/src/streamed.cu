#include <cuda_runtime_api.h>

#include <cuda_bf16.h>

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
constexpr int kPressureTileZ = 1;
constexpr int kFusedCoreX = 32;
constexpr int kFusedCoreY = 16;
constexpr int kFusedCoreZ = 2;
constexpr int kFusedSharedX = kFusedCoreX + 1;
constexpr int kFusedSharedY = kFusedCoreY + 1;
constexpr int kFusedSharedZ = kFusedCoreZ + 1;
constexpr int kFusedVolume = kFusedSharedX * kFusedSharedY * kFusedSharedZ;
constexpr size_t kFusedSharedBytes = 3 * kFusedVolume * sizeof(float);
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

bool TypedPsiEnabled() {
  const char* value = std::getenv("BEAMZ_CUDA_DISABLE_TYPED_PSI");
  return value == nullptr || value[0] == '\0' || value[0] == '0';
}

bool PrecomputedDftPhasesEnabled() {
  const char* value = std::getenv("BEAMZ_CUDA_DISABLE_PRECOMPUTED_DFT_PHASES");
  return value == nullptr || value[0] == '\0' || value[0] == '0';
}

std::string GraphKey(void* stream, const BeamzLaunch& h_launch,
                     const BeamzLaunch& e_launch, int32_t nsteps,
                     const BeamzSourceLaunch* source = nullptr,
                     const BeamzSourceGroupLaunch* source_groups = nullptr,
                     int32_t source_group_count = 0,
                     const BeamzDftLaunch* monitor = nullptr,
                     const BeamzDftGroupLaunch* monitor_groups = nullptr) {
  std::string key;
  key.reserve(sizeof(stream) + sizeof(nsteps) + 2 * sizeof(BeamzLaunch));
  key.append(reinterpret_cast<const char*>(&stream), sizeof(stream));
  key.append(reinterpret_cast<const char*>(&nsteps), sizeof(nsteps));
  key.append(reinterpret_cast<const char*>(&h_launch), sizeof(h_launch));
  key.append(reinterpret_cast<const char*>(&e_launch), sizeof(e_launch));
  if (source != nullptr) {
    key.append(reinterpret_cast<const char*>(source), sizeof(*source));
  }
  if (source_groups != nullptr && source_group_count > 0) {
    key.append(reinterpret_cast<const char*>(&source_group_count),
               sizeof(source_group_count));
    key.append(reinterpret_cast<const char*>(source_groups),
               sizeof(*source_groups) * source_group_count);
  }
  if (monitor != nullptr) {
    key.append(reinterpret_cast<const char*>(monitor), sizeof(*monitor));
  }
  if (monitor_groups != nullptr) {
    key.append(reinterpret_cast<const char*>(monitor_groups),
               sizeof(*monitor_groups));
  }
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

template <int MetricKind>
__device__ __forceinline__ float MetricScale(const BeamzLaunch& launch,
                                             int axis, int coordinate) {
  if constexpr (MetricKind == 0) {
    return launch.inv_resolution;
  } else if constexpr (MetricKind == 1) {
    return static_cast<const float*>(launch.metrics[axis].data)[0];
  } else {
    return static_cast<const float*>(launch.metrics[axis].data)[coordinate];
  }
}

template <int MetricKind>
__device__ __forceinline__ float ForwardDifference(const BeamzBuffer& value,
                                                   int axis, int z, int y,
                                                   int x,
                                                   const BeamzLaunch& launch) {
  const int coordinate = axis == 0 ? z : (axis == 1 ? y : x);
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
         MetricScale<MetricKind>(launch, axis, coordinate);
}

template <int MetricKind, bool HasMetallicEdges = true>
__device__ __forceinline__ float BoundaryDifference(const BeamzBuffer& value,
                                                    int axis, int z, int y,
                                                    int x, int edge_mask,
                                                    const BeamzLaunch& launch) {
  const int coordinate = axis == 0 ? z : (axis == 1 ? y : x);
  const float inv_dx = MetricScale<MetricKind>(launch, axis, coordinate);
  const int size = static_cast<int>(value.dims[axis]);
  if (coordinate == 0) {
    if constexpr (!HasMetallicEdges) return 0.0f;
    const bool metallic = edge_mask & (1 << (2 * axis));
    return metallic ? Read3D(value, z, y, x) * inv_dx : 0.0f;
  }
  if (coordinate == size) {
    if constexpr (!HasMetallicEdges) return 0.0f;
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

template <int Term, bool UniformCpml = false, int PsiType = -1>
__device__ __forceinline__ float CorrectCpml(float derivative, int z, int y,
                                             int x, const BeamzLaunch& launch) {
  // The 3D compiler emits the six curl terms in this fixed derivative order.
  // CPML coefficient buffers are 1D profiles along their derivative axis.
  constexpr int axes[6] = {1, 0, 0, 2, 2, 1};
  constexpr int axis = axes[Term];
  constexpr float sign = Term % 2 == 0 ? 1.0f : -1.0f;
  int low;
  int high;
  if constexpr (UniformCpml) {
    low = high = launch.uniform_cpml_thickness;
  } else {
    const auto* descriptor =
        static_cast<const int32_t*>(launch.inputs[12].data);
    low = descriptor[Term * 5 + 2];
    high = descriptor[Term * 5 + 3];
  }
  const BeamzBuffer& target = launch.outputs[Term / 2];
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
  const int coefficient_base = 13 + 3 * Term;
  const int psi_base = 13 + 3 * launch.nterms;
  const BeamzBuffer& psi_input = launch.inputs[psi_base + Term];
  const BeamzBuffer& psi_output = launch.outputs[3 + Term];
  const int psi_offset = (pz * static_cast<int>(psi_output.dims[1]) + py) *
                             static_cast<int>(psi_output.dims[2]) +
                         px;
  float old_psi;
  if constexpr (PsiType == kBeamzBF16) {
    old_psi = __bfloat162float(
        static_cast<const __nv_bfloat16*>(psi_input.data)[psi_offset]);
  } else if constexpr (PsiType == kBeamzF32) {
    old_psi = static_cast<const float*>(psi_input.data)[psi_offset];
  } else {
    old_psi = psi_input.element_type == kBeamzBF16
                  ? __bfloat162float(static_cast<const __nv_bfloat16*>(
                                         psi_input.data)[psi_offset])
                  : static_cast<const float*>(psi_input.data)[psi_offset];
  }
  const float next_psi =
      static_cast<const float *>(
          launch.inputs[coefficient_base + 1].data)[packed] *
          old_psi +
      static_cast<const float*>(launch.inputs[coefficient_base].data)[packed] *
          derivative;
  if constexpr (PsiType == kBeamzBF16) {
    static_cast<__nv_bfloat16*>(psi_output.data)[psi_offset] =
        __float2bfloat16_rn(next_psi);
  } else if constexpr (PsiType == kBeamzF32) {
    static_cast<float*>(psi_output.data)[psi_offset] = next_psi;
  } else if (psi_output.element_type == kBeamzBF16) {
    static_cast<__nv_bfloat16*>(psi_output.data)[psi_offset] =
        __float2bfloat16_rn(next_psi);
  } else {
    static_cast<float*>(psi_output.data)[psi_offset] = next_psi;
  }
  return sign *
         (derivative * static_cast<const float*>(
                           launch.inputs[coefficient_base + 2].data)[packed] +
          next_psi);
}

template <int Phase, int Component, bool Cpml, int MetricKind,
          bool HasMetallicEdges = true, bool UniformCpml = false,
          int PsiType = -1>
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
  bool zero_on_wall = false;
  if constexpr (HasMetallicEdges) {
    if constexpr (constrained) {
      zero_on_wall = on_low_wall || on_high_wall;
    } else {
      for (int axis = 0; axis < 3; ++axis) {
        if (axis == normal_axis) continue;
        const int axis_coordinate = axis == 0 ? z : (axis == 1 ? y : x);
        const int size = static_cast<int>(output.dims[axis]);
        if ((axis_coordinate == 0 &&
             (launch.metallic_edges & (1 << (2 * axis)))) ||
            (axis_coordinate == size - 1 &&
             (launch.metallic_edges & (1 << (2 * axis + 1))))) {
          zero_on_wall = true;
          break;
        }
      }
    }
  }
  if constexpr (!Cpml) {
    if (zero_on_wall) {
      static_cast<float*>(output.data)[linear] = 0.0f;
      return;
    }
  }
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
        ForwardDifference<MetricKind>(first, first_axis[Component], z, y, x,
                                      launch);
    derivative1 =
        ForwardDifference<MetricKind>(second, second_axis[Component], z, y, x,
                                      launch);
  } else {
    derivative0 = BoundaryDifference<MetricKind, HasMetallicEdges>(
        first, first_axis[Component], z, y, x, launch.metallic_edges, launch);
    derivative1 = BoundaryDifference<MetricKind, HasMetallicEdges>(
        second, second_axis[Component], z, y, x, launch.metallic_edges, launch);
  }
  float curl;
  if constexpr (Cpml) {
    curl = CorrectCpml<2 * Component, UniformCpml, PsiType>(
               derivative0, z, y, x, launch) +
           CorrectCpml<2 * Component + 1, UniformCpml, PsiType>(
               derivative1, z, y, x, launch);
  } else {
    curl = derivative0 - derivative1;
  }
  if (zero_on_wall) {
    // CPML memory still evolves where an absorbing face intersects a PEC face;
    // only the constrained field value is masked after the recurrence update.
    static_cast<float*>(output.data)[linear] = 0.0f;
    return;
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

template <int Phase, bool Cpml, int MetricKind, bool HasMetallicEdges = true,
          bool UniformCpml = false, int PsiType = -1>
__global__ void UpdateFusedComponents(BeamzLaunch launch) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  UpdateComponent<Phase, 0, Cpml, MetricKind, HasMetallicEdges, UniformCpml,
                  PsiType>(launch, z, y, x);
  UpdateComponent<Phase, 1, Cpml, MetricKind, HasMetallicEdges, UniformCpml,
                  PsiType>(launch, z, y, x);
  UpdateComponent<Phase, 2, Cpml, MetricKind, HasMetallicEdges, UniformCpml,
                  PsiType>(launch, z, y, x);
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
__global__ void UpdateFusedFullPecScalarComponents(BeamzLaunch launch) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  UpdateFullPecScalarComponent<Phase, 0>(launch, z, y, x);
  UpdateFullPecScalarComponent<Phase, 1>(launch, z, y, x);
  UpdateFullPecScalarComponent<Phase, 2>(launch, z, y, x);
}


__device__ __forceinline__ int FusedOffset(int z, int y, int x) {
  return (z * kFusedSharedY + y) * kFusedSharedX + x;
}

__device__ __forceinline__ bool BufferContains(const BeamzBuffer& value, int z,
                                                int y, int x) {
  return z >= 0 && y >= 0 && x >= 0 && z < value.dims[0] &&
         y < value.dims[1] && x < value.dims[2];
}

template <bool ScalarCoefficients, int MetricKind, int Component>
__device__ __forceinline__ float FusedHValue(const BeamzLaunch& launch,
                                             int z, int y, int x) {
  const BeamzBuffer& output = launch.outputs[Component];
  if (!BufferContains(output, z, y, x)) return 0.0f;
  constexpr int normal_axis = 2 - Component;
  const int coordinate = normal_axis == 0 ? z : (normal_axis == 1 ? y : x);
  if (coordinate == 0 || coordinate == output.dims[normal_axis] - 1) {
    return 0.0f;
  }
  constexpr int first_source[3] = {2, 0, 1};
  constexpr int second_source[3] = {1, 2, 0};
  constexpr int first_axis[3] = {1, 0, 2};
  constexpr int second_axis[3] = {0, 2, 1};
  const float derivative0 = ForwardDifference<MetricKind>(
      launch.inputs[3 + first_source[Component]], first_axis[Component], z, y,
      x, launch);
  const float derivative1 = ForwardDifference<MetricKind>(
      launch.inputs[3 + second_source[Component]], second_axis[Component], z,
      y, x, launch);
  const int linear = (z * static_cast<int>(output.dims[1]) + y) *
                         static_cast<int>(output.dims[2]) +
                     x;
  const float old_field =
      static_cast<const float*>(launch.inputs[Component].data)[linear];
  const float decay =
      ScalarCoefficients
          ? static_cast<const float*>(launch.inputs[6 + Component].data)[0]
          : Read3D(launch.inputs[6 + Component], z, y, x);
  const float source =
      ScalarCoefficients
          ? static_cast<const float*>(launch.inputs[9 + Component].data)[0]
          : Read3D(launch.inputs[9 + Component], z, y, x);
  return decay * old_field - source * (derivative0 - derivative1);
}

template <bool ScalarCoefficients, int MetricKind, int Component>
__device__ __forceinline__ float FusedEValue(const BeamzLaunch& launch,
                                             const float* h_fields,
                                             int local_z, int local_y,
                                             int local_x, int z, int y, int x) {
  const BeamzBuffer& output = launch.outputs[Component];
  if (!BufferContains(output, z, y, x)) return 0.0f;
  constexpr int normal_axis = 2 - Component;
  const int coordinates[3] = {z, y, x};
  for (int axis = 0; axis < 3; ++axis) {
    if (axis != normal_axis &&
        (coordinates[axis] == 0 ||
         coordinates[axis] == output.dims[axis] - 1)) {
      return 0.0f;
    }
  }
  constexpr int first_source[3] = {2, 0, 1};
  constexpr int second_source[3] = {1, 2, 0};
  constexpr int first_axis[3] = {1, 0, 2};
  constexpr int second_axis[3] = {0, 2, 1};
  auto difference = [&](int source_component, int axis) {
    int neighbor_z = local_z;
    int neighbor_y = local_y;
    int neighbor_x = local_x;
    if (axis == 0) --neighbor_z;
    if (axis == 1) --neighbor_y;
    if (axis == 2) --neighbor_x;
    const float center = h_fields[source_component * kFusedVolume +
                                  FusedOffset(local_z, local_y, local_x)];
    const float neighbor = h_fields[source_component * kFusedVolume +
                                    FusedOffset(neighbor_z, neighbor_y,
                                                neighbor_x)];
    const int coordinate = axis == 0 ? z : (axis == 1 ? y : x);
    return (center - neighbor) *
           MetricScale<MetricKind>(launch, axis, coordinate);
  };
  const float curl = difference(first_source[Component], first_axis[Component]) -
                     difference(second_source[Component],
                                second_axis[Component]);
  const int linear = (z * static_cast<int>(output.dims[1]) + y) *
                         static_cast<int>(output.dims[2]) +
                     x;
  const float old_field =
      static_cast<const float*>(launch.inputs[Component].data)[linear];
  const float decay =
      ScalarCoefficients
          ? static_cast<const float*>(launch.inputs[6 + Component].data)[0]
          : Read3D(launch.inputs[6 + Component], z, y, x);
  const float source =
      ScalarCoefficients
          ? static_cast<const float*>(launch.inputs[9 + Component].data)[0]
          : Read3D(launch.inputs[9 + Component], z, y, x);
  return decay * old_field + source * curl;
}

// Fuse a complete leapfrog timestep without a device-wide barrier.  Each block
// redundantly computes the one-cell low halo of H in shared memory, then uses it
// to update its disjoint E/H core into a frozen out-of-place destination.
template <bool ScalarCoefficients, int MetricKind>
__global__ void FusedFullStepPec(BeamzLaunch h_launch,
                                 BeamzLaunch e_launch) {
  extern __shared__ float h_fields[];
  const int thread = threadIdx.y * blockDim.x + threadIdx.x;
  const int threads = blockDim.x * blockDim.y;
  const int origin_x = blockIdx.x * kFusedCoreX - 1;
  const int origin_y = blockIdx.y * kFusedCoreY - 1;
  const int origin_z = blockIdx.z * kFusedCoreZ - 1;
  for (int index = thread; index < kFusedVolume; index += threads) {
    const int local_x = index % kFusedSharedX;
    const int local_y = (index / kFusedSharedX) % kFusedSharedY;
    const int local_z = index / (kFusedSharedX * kFusedSharedY);
    const int x = origin_x + local_x;
    const int y = origin_y + local_y;
    const int z = origin_z + local_z;
    h_fields[index] =
        FusedHValue<ScalarCoefficients, MetricKind, 0>(h_launch, z, y, x);
    h_fields[kFusedVolume + index] =
        FusedHValue<ScalarCoefficients, MetricKind, 1>(h_launch, z, y, x);
    h_fields[2 * kFusedVolume + index] =
        FusedHValue<ScalarCoefficients, MetricKind, 2>(h_launch, z, y, x);
  }
  __syncthreads();

  const int local_x = threadIdx.x + 1;
  const int local_y = threadIdx.y + 1;
  const int x = origin_x + local_x;
  const int y = origin_y + local_y;
  for (int core_z = 0; core_z < kFusedCoreZ; ++core_z) {
    const int local_z = core_z + 1;
    const int z = origin_z + local_z;
    const int center = FusedOffset(local_z, local_y, local_x);
    for (int component = 0; component < 3; ++component) {
      const BeamzBuffer& h_output = h_launch.outputs[component];
      if (BufferContains(h_output, z, y, x)) {
        const int linear = (z * static_cast<int>(h_output.dims[1]) + y) *
                               static_cast<int>(h_output.dims[2]) +
                           x;
        static_cast<float*>(h_output.data)[linear] =
            h_fields[component * kFusedVolume + center];
      }
    }
    const float e0 =
        FusedEValue<ScalarCoefficients, MetricKind, 0>(
            e_launch, h_fields, local_z, local_y, local_x, z, y, x);
    const float e1 =
        FusedEValue<ScalarCoefficients, MetricKind, 1>(
            e_launch, h_fields, local_z, local_y, local_x, z, y, x);
    const float e2 =
        FusedEValue<ScalarCoefficients, MetricKind, 2>(
            e_launch, h_fields, local_z, local_y, local_x, z, y, x);
    const float values[3] = {e0, e1, e2};
    for (int component = 0; component < 3; ++component) {
      const BeamzBuffer& e_output = e_launch.outputs[component];
      if (BufferContains(e_output, z, y, x)) {
        const int linear = (z * static_cast<int>(e_output.dims[1]) + y) *
                               static_cast<int>(e_output.dims[2]) +
                           x;
        static_cast<float*>(e_output.data)[linear] = values[component];
      }
    }
  }
}

template <int MetricKind, bool HasMetallicEdges, bool UniformCpml>
void LaunchFusedUpdate(cudaStream_t stream, const BeamzLaunch& launch,
                       dim3 blocks, dim3 threads) {
  int psi_type = -1;
  if (launch.nterms != 0 && TypedPsiEnabled()) {
    constexpr int psi_input_base = 13 + 3 * 6;
    psi_type = launch.outputs[3].element_type;
    for (int term = 0; term < 6; ++term) {
      if (launch.inputs[psi_input_base + term].element_type != psi_type ||
          launch.outputs[3 + term].element_type != psi_type) {
        psi_type = -1;
        break;
      }
    }
  }
  if (launch.phase == 0 && launch.nterms == 0) {
    UpdateFusedComponents<0, false, MetricKind, HasMetallicEdges, UniformCpml>
        <<<blocks, threads, 0, stream>>>(launch);
  } else if (launch.phase == 0) {
    if (psi_type == kBeamzBF16) {
      UpdateFusedComponents<0, true, MetricKind, HasMetallicEdges, UniformCpml,
                            kBeamzBF16>
          <<<blocks, threads, 0, stream>>>(launch);
    } else if (psi_type == kBeamzF32) {
      UpdateFusedComponents<0, true, MetricKind, HasMetallicEdges, UniformCpml,
                            kBeamzF32>
          <<<blocks, threads, 0, stream>>>(launch);
    } else {
      UpdateFusedComponents<0, true, MetricKind, HasMetallicEdges, UniformCpml>
          <<<blocks, threads, 0, stream>>>(launch);
    }
  } else if (launch.nterms == 0) {
    UpdateFusedComponents<1, false, MetricKind, HasMetallicEdges, UniformCpml>
        <<<blocks, threads, 0, stream>>>(launch);
  } else {
    if (psi_type == kBeamzBF16) {
      UpdateFusedComponents<1, true, MetricKind, HasMetallicEdges, UniformCpml,
                            kBeamzBF16>
          <<<blocks, threads, 0, stream>>>(launch);
    } else if (psi_type == kBeamzF32) {
      UpdateFusedComponents<1, true, MetricKind, HasMetallicEdges, UniformCpml,
                            kBeamzF32>
          <<<blocks, threads, 0, stream>>>(launch);
    } else {
      UpdateFusedComponents<1, true, MetricKind, HasMetallicEdges, UniformCpml>
          <<<blocks, threads, 0, stream>>>(launch);
    }
  }
}

template <int MetricKind, bool HasMetallicEdges>
void LaunchFusedUpdateForBoundary(cudaStream_t stream,
                                  const BeamzLaunch& launch, dim3 blocks,
                                  dim3 threads) {
  if (launch.nterms != 0 && launch.uniform_cpml_thickness > 0) {
    LaunchFusedUpdate<MetricKind, HasMetallicEdges, true>(stream, launch,
                                                          blocks, threads);
  } else {
    LaunchFusedUpdate<MetricKind, HasMetallicEdges, false>(stream, launch,
                                                           blocks, threads);
  }
}

__global__ void ApplySourceSlab(BeamzLaunch e_launch,
                                BeamzSourceLaunch source, int step_offset) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (z >= source.coefficient.dims[0] || y >= source.coefficient.dims[1] ||
      x >= source.coefficient.dims[2]) {
    return;
  }
  const int target_z = source.starts[0] + z;
  const int target_y = source.starts[1] + y;
  const int target_x = source.starts[2] + x;
  const BeamzBuffer& target = e_launch.outputs[source.component];
  const int target_offset =
      (target_z * static_cast<int>(target.dims[1]) + target_y) *
          static_cast<int>(target.dims[2]) +
      target_x;
  const int coefficient_offset =
      (z * static_cast<int>(source.coefficient.dims[1]) + y) *
          static_cast<int>(source.coefficient.dims[2]) +
      x;
  int waveform_index =
      static_cast<const int32_t*>(source.current_step.data)[0] + step_offset;
  waveform_index = waveform_index < 0 ? 0 : waveform_index;
  waveform_index =
      waveform_index >= source.waveform.dims[0]
          ? static_cast<int>(source.waveform.dims[0]) - 1
          : waveform_index;
  const float amplitude =
      static_cast<const float*>(source.waveform.data)[waveform_index];
  static_cast<float*>(target.data)[target_offset] +=
      static_cast<const float*>(source.coefficient.data)[coefficient_offset] *
      amplitude;
}

__device__ __forceinline__ bool SourceCellConstrained(
    const BeamzBuffer& target, int component, int phase, int metallic_edges,
    int z, int y, int x) {
  const int normal_axis = 2 - component;
  const int coordinates[3] = {z, y, x};
  if (phase == 0) {
    const int coordinate = coordinates[normal_axis];
    return (coordinate == 0 &&
            (metallic_edges & (1 << (2 * normal_axis)))) ||
           (coordinate == target.dims[normal_axis] - 1 &&
            (metallic_edges & (1 << (2 * normal_axis + 1))));
  }
  for (int axis = 0; axis < 3; ++axis) {
    if (axis == normal_axis) continue;
    const int coordinate = coordinates[axis];
    if ((coordinate == 0 && (metallic_edges & (1 << (2 * axis)))) ||
        (coordinate == target.dims[axis] - 1 &&
         (metallic_edges & (1 << (2 * axis + 1))))) {
      return true;
    }
  }
  return false;
}

__global__ void ApplySourceGroup(BeamzBuffer target,
                                 BeamzSourceGroupLaunch group,
                                 int source_index, int step_offset,
                                 int metallic_edges) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (z >= group.coefficients.dims[1] ||
      y >= group.coefficients.dims[2] ||
      x >= group.coefficients.dims[3]) {
    return;
  }
  const auto* starts = static_cast<const int32_t*>(group.starts.data);
  const int target_z = starts[3 * source_index] + z;
  const int target_y = starts[3 * source_index + 1] + y;
  const int target_x = starts[3 * source_index + 2] + x;
  if (target_z < 0 || target_z >= target.dims[0] || target_y < 0 ||
      target_y >= target.dims[1] || target_x < 0 ||
      target_x >= target.dims[2]) {
    return;
  }
  // Sources injected after a field update are followed by PEC restoration in the
  // canonical step. Skipping those constrained additions is equivalent because
  // the native field update has already written zero to every constrained cell.
  if (group.timing != 0 &&
      SourceCellConstrained(target, group.component,
                            group.timing == 1 ? 0 : 1, metallic_edges,
                            target_z, target_y, target_x)) {
    return;
  }
  int waveform_index =
      static_cast<const int32_t*>(group.current_step.data)[0] + step_offset;
  waveform_index = waveform_index < 0 ? 0 : waveform_index;
  waveform_index =
      waveform_index >= group.waveforms.dims[1]
          ? static_cast<int>(group.waveforms.dims[1]) - 1
          : waveform_index;
  const int waveform_offset =
      source_index * static_cast<int>(group.waveforms.dims[1]) +
      waveform_index;
  const int coefficient_offset =
      ((source_index * static_cast<int>(group.coefficients.dims[1]) + z) *
           static_cast<int>(group.coefficients.dims[2]) +
       y) *
          static_cast<int>(group.coefficients.dims[3]) +
      x;
  const int target_offset =
      (target_z * static_cast<int>(target.dims[1]) + target_y) *
          static_cast<int>(target.dims[2]) +
      target_x;
  static_cast<float*>(target.data)[target_offset] +=
      static_cast<const float*>(group.coefficients.data)[coefficient_offset] *
      static_cast<const float*>(group.waveforms.data)[waveform_offset];
}

__global__ void PreparePlaneDftPhases(BeamzLaunch e_launch,
                                      BeamzDftLaunch monitor,
                                      int nsteps) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = nsteps * monitor.frequency_count;
  if (linear >= count) return;
  const int step = linear / monitor.frequency_count;
  const int frequency = linear - step * monitor.frequency_count;
  const float time = static_cast<const float*>(monitor.time.data)[0] +
                     static_cast<float>(step + 1) * e_launch.dt;
  const float theta =
      6.2831853071795864769f *
      static_cast<const float*>(monitor.frequencies.data)[frequency] * time;
  float phase_sin;
  float phase_cos;
  sincosf(theta, &phase_sin, &phase_cos);
  static_cast<float*>(monitor.phase_cos.data)[linear] = phase_cos;
  static_cast<float*>(monitor.phase_sin.data)[linear] = phase_sin;
}

__global__ void AccumulatePlaneDft(BeamzLaunch h_launch,
                                   BeamzLaunch e_launch,
                                   BeamzDftLaunch monitor, int step_offset,
                                   bool precomputed_phase) {
  const int point = blockIdx.x * blockDim.x + threadIdx.x;
  const int frequency = blockIdx.y * blockDim.y + threadIdx.y;
  const int component = blockIdx.z * blockDim.z + threadIdx.z;
  if (point >= monitor.point_count || frequency >= monitor.frequency_count ||
      component >= 6) {
    return;
  }
  if (component == 0 && point == 0) {
    static_cast<float*>(monitor.dft_weight.data)[frequency] += 1.0f;
  }
  const float mask =
      static_cast<const float*>(monitor.component_mask.data)[component];
  if (mask == 0.0f) return;

  const BeamzBuffer& indices = monitor.indices[component];
  const BeamzBuffer& weights = monitor.weights[component];
  const BeamzBuffer& field = component < 3 ? e_launch.outputs[component]
                                           : h_launch.outputs[component - 3];
  float sample = 0.0f;
  const int neighbors = static_cast<int>(indices.dims[1]);
  for (int neighbor = 0; neighbor < neighbors; ++neighbor) {
    const int gather_offset = point * neighbors + neighbor;
    const int field_offset =
        static_cast<const int32_t*>(indices.data)[gather_offset];
    sample += static_cast<const float*>(field.data)[field_offset] *
              static_cast<const float*>(weights.data)[gather_offset];
  }

  float phase_sin;
  float phase_cos;
  if (precomputed_phase) {
    const int phase_offset = step_offset * monitor.frequency_count + frequency;
    phase_cos = static_cast<const float*>(monitor.phase_cos.data)[phase_offset];
    phase_sin = static_cast<const float*>(monitor.phase_sin.data)[phase_offset];
  } else {
    const float t = static_cast<const float*>(monitor.time.data)[0] +
                    static_cast<float>(step_offset + 1) * e_launch.dt;
    const float theta =
        6.2831853071795864769f *
        static_cast<const float*>(monitor.frequencies.data)[frequency] * t;
    sincosf(theta, &phase_sin, &phase_cos);
  }
  const int accumulator_offset =
      (component * static_cast<int>(monitor.dft_re.dims[2]) + frequency) *
          static_cast<int>(monitor.dft_re.dims[3]) +
      point;
  static_cast<float*>(monitor.dft_re.data)[accumulator_offset] +=
      sample * phase_cos;
  static_cast<float*>(monitor.dft_im.data)[accumulator_offset] +=
      sample * phase_sin;
}

__global__ void AccumulateDftGroups(BeamzLaunch h_launch,
                                    BeamzLaunch e_launch,
                                    BeamzDftGroupLaunch monitors,
                                    int step_offset) {
  const int point = blockIdx.x * blockDim.x + threadIdx.x;
  const int frequency = blockIdx.y * blockDim.y + threadIdx.y;
  const int lane = blockIdx.z * blockDim.z + threadIdx.z;
  const int monitor = lane / 6;
  const int component = lane % 6;
  if (monitor >= monitors.monitor_count) return;

  const auto* counts = static_cast<const int32_t*>(monitors.counts.data);
  const int frequency_count = counts[5 * monitor];
  const int point_count = counts[5 * monitor + 1];
  const int interval = counts[5 * monitor + 2] > 0
                           ? counts[5 * monitor + 2]
                           : 1;
  const int value_offset = counts[5 * monitor + 3];
  const int weight_offset = counts[5 * monitor + 4];
  if (frequency >= frequency_count) return;

  const int absolute_step =
      static_cast<const int32_t*>(monitors.current_step.data)[0] + step_offset;
  if (absolute_step % interval != 0) return;
  const float time = static_cast<const float*>(monitors.time.data)[0] +
                     static_cast<float>(step_offset + 1) * e_launch.dt;
  const auto* windows = static_cast<const float*>(monitors.windows.data);
  const float start = windows[3 * monitor];
  const float end = windows[3 * monitor + 1];
  if (time < start || time > end) return;

  const auto* codes = static_cast<const int32_t*>(monitors.codes.data);
  const int max_frequency_count = static_cast<int>(monitors.frequencies.dims[1]);
  float window = 0.0f;
  float phase_sin = 0.0f;
  float phase_cos = 0.0f;
  if (threadIdx.x == 0) {
    window = 1.0f;
    if (codes[2 * monitor] == 1 && isfinite(end) && end > start) {
      const float tau =
          fminf(fmaxf((time - start) / (end - start), 0.0f), 1.0f);
      window = 0.5f * (1.0f - cosf(6.2831853071795864769f * tau));
    }
    const float frequency_hz =
        static_cast<const float*>(monitors.frequencies.data)
            [monitor * max_frequency_count + frequency];
    sincosf(6.2831853071795864769f * frequency_hz * time, &phase_sin,
            &phase_cos);
  }
  window = __shfl_sync(0xffffffff, window, 0);
  phase_sin = __shfl_sync(0xffffffff, phase_sin, 0);
  phase_cos = __shfl_sync(0xffffffff, phase_cos, 0);
  if (point >= point_count) return;
  if (component == 0 && point == 0) {
    static_cast<float*>(monitors.dft_weight.data)
        [weight_offset + frequency] += window;
  }
  const float mask = static_cast<const float*>(monitors.component_masks.data)
      [monitor * 6 + component];
  if (mask == 0.0f) return;

  const int max_points = static_cast<int>(monitors.indices.dims[2]);
  const int neighbors = static_cast<int>(monitors.indices.dims[3]);
  const int plan_base = ((monitor * 6 + component) * max_points + point) *
                        neighbors;
  const BeamzBuffer& field = component < 3 ? e_launch.outputs[component]
                                           : h_launch.outputs[component - 3];
  float sample = 0.0f;
  for (int neighbor = 0; neighbor < neighbors; ++neighbor) {
    const int gather_offset = plan_base + neighbor;
    const int field_offset =
        static_cast<const int32_t*>(monitors.indices.data)[gather_offset];
    sample += static_cast<const float*>(field.data)[field_offset] *
              static_cast<const float*>(monitors.weights.data)[gather_offset];
  }

  float scale = window;
  if (codes[2 * monitor + 1] == 1) {
    const float length_unit = windows[3 * monitor + 2];
    scale *= e_launch.dt * static_cast<float>(interval) * 299792458.0f /
             length_unit / sqrtf(6.2831853071795864769f);
  }
  const int accumulator_offset =
      value_offset +
      (component * frequency_count + frequency) * point_count +
      point;
  static_cast<float*>(monitors.dft_re.data)[accumulator_offset] +=
      scale * sample * phase_cos;
  static_cast<float*>(monitors.dft_im.data)[accumulator_offset] +=
      scale * sample * phase_sin;
}

}  // namespace

int BeamzLaunchStreamed(void* raw_stream, const BeamzLaunch& launch) {
  if (launch.metric_kind < 0 || launch.metric_kind > 2) {
    return cudaErrorInvalidValue;
  }
  const int input_count = launch.nterms == 0 ? 12 : 13 + 4 * launch.nterms;
  const int output_count = 3 + launch.nterms;
  for (int index = 0; index < input_count; ++index) {
    if (!FitsIntOffsets(launch.inputs[index])) return cudaErrorInvalidValue;
  }
  for (int index = 0; index < output_count; ++index) {
    if (!FitsIntOffsets(launch.outputs[index])) return cudaErrorInvalidValue;
  }
  for (int axis = 0; axis < 3; ++axis) {
    const BeamzBuffer& metric = launch.metrics[axis];
    if (!FitsIntOffsets(metric)) return cudaErrorInvalidValue;
    if ((launch.metric_kind == 1 && metric.rank != 0) ||
        (launch.metric_kind == 2 &&
         (metric.rank != 1 || metric.dims[0] < 1))) {
      return cudaErrorInvalidValue;
    }
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
  const int tile_z =
      launch.nterms != 0 || launch.metric_kind != 0 ? kPressureTileZ : kTileZ;
  const dim3 threads(kTileX, kTileY, tile_z);
  const dim3 fused_blocks((max_x + kTileX - 1) / kTileX, y_blocks,
                          (max_z + tile_z - 1) / tile_z);
  const bool scalar_coefficients =
      launch.nterms == 0 && launch.inputs[6].rank == 0 &&
      launch.inputs[7].rank == 0 && launch.inputs[8].rank == 0 &&
      launch.inputs[9].rank == 0 && launch.inputs[10].rank == 0 &&
      launch.inputs[11].rank == 0;
  if (launch.metric_kind == 0 && scalar_coefficients &&
      launch.metallic_edges == 63) {
    if (launch.phase == 0) {
      UpdateFusedFullPecScalarComponents<0>
          <<<fused_blocks, threads, 0, stream>>>(launch);
    } else {
      UpdateFusedFullPecScalarComponents<1>
          <<<fused_blocks, threads, 0, stream>>>(launch);
    }
  } else if (launch.metric_kind == 0) {
    if (launch.metallic_edges == 0) {
      LaunchFusedUpdateForBoundary<0, false>(stream, launch, fused_blocks,
                                              threads);
    } else {
      LaunchFusedUpdateForBoundary<0, true>(stream, launch, fused_blocks,
                                             threads);
    }
  } else if (launch.metric_kind == 1) {
    if (launch.metallic_edges == 0) {
      LaunchFusedUpdateForBoundary<1, false>(stream, launch, fused_blocks,
                                              threads);
    } else {
      LaunchFusedUpdateForBoundary<1, true>(stream, launch, fused_blocks,
                                             threads);
    }
  } else {
    if (launch.metallic_edges == 0) {
      LaunchFusedUpdateForBoundary<2, false>(stream, launch, fused_blocks,
                                              threads);
    } else {
      LaunchFusedUpdateForBoundary<2, true>(stream, launch, fused_blocks,
                                             threads);
    }
  }
  return static_cast<int>(cudaPeekAtLastError());
}

int LaunchStreamedGraph(void* raw_stream, const BeamzLaunch& h_launch,
                        const BeamzLaunch& e_launch,
                        const BeamzSourceLaunch* source,
                        const BeamzSourceGroupLaunch* source_groups,
                        int32_t source_group_count,
                        const BeamzDftLaunch* monitor,
                        const BeamzDftGroupLaunch* monitor_groups,
                        int32_t nsteps) {
  if (nsteps < 1 || h_launch.phase != 0 || e_launch.phase != 1 ||
      h_launch.nterms != e_launch.nterms ||
      (h_launch.nterms != 0 && h_launch.nterms != 6)) {
    return cudaErrorInvalidValue;
  }
  auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
  const std::string graph_key =
      GraphKey(raw_stream, h_launch, e_launch, nsteps, source, source_groups,
               source_group_count, monitor, monitor_groups);
  GraphCache& cache = CachedGraphs();
  const bool cache_enabled = GraphCacheEnabled();
  if (cache_enabled) {
    std::lock_guard<std::mutex> lock(cache.mutex);
    const auto cached = cache.entries.find(graph_key);
    if (cached != cache.entries.end()) {
      return static_cast<int>(cudaGraphLaunch(cached->second, stream));
    }
  }
  auto launch_steps = [&]() {
    cudaError_t launch_error = cudaSuccess;
    const bool precomputed_monitor_phase =
        monitor != nullptr && PrecomputedDftPhasesEnabled() &&
        monitor->phase_cos.rank == 2 && monitor->phase_sin.rank == 2 &&
        monitor->phase_cos.dims[0] >= nsteps &&
        monitor->phase_sin.dims[0] >= nsteps &&
        monitor->phase_cos.dims[1] >= monitor->frequency_count &&
        monitor->phase_sin.dims[1] >= monitor->frequency_count;
    if (precomputed_monitor_phase) {
      constexpr int phase_threads = 128;
      const int phase_count = nsteps * monitor->frequency_count;
      PreparePlaneDftPhases<<<
          (phase_count + phase_threads - 1) / phase_threads, phase_threads, 0,
          stream>>>(e_launch, *monitor, nsteps);
      launch_error = cudaPeekAtLastError();
      if (launch_error != cudaSuccess) return launch_error;
    }
    auto launch_source_groups = [&](int timing, int32_t step) {
      for (int32_t group_index = 0; group_index < source_group_count;
           ++group_index) {
        const BeamzSourceGroupLaunch& group = source_groups[group_index];
        if (group.timing != timing || group.coefficients.dims[0] == 0) continue;
        const BeamzLaunch& target_launch = timing == 1 ? h_launch : e_launch;
        const BeamzBuffer& target = target_launch.outputs[group.component];
        const dim3 source_threads(kTileX, kTileY, kTileZ);
        const dim3 source_blocks(
            (group.coefficients.dims[3] + kTileX - 1) / kTileX,
            (group.coefficients.dims[2] + kTileY - 1) / kTileY,
            (group.coefficients.dims[1] + kTileZ - 1) / kTileZ);
        for (int32_t source_index = 0;
             source_index < group.coefficients.dims[0]; ++source_index) {
          ApplySourceGroup<<<source_blocks, source_threads, 0, stream>>>(
              target, group, source_index, step, h_launch.metallic_edges);
          launch_error = cudaPeekAtLastError();
          if (launch_error != cudaSuccess) return;
        }
      }
    };
    for (int32_t step = 0; step < nsteps; ++step) {
      if (source != nullptr) {
        const dim3 source_threads(kTileX, kTileY, kTileZ);
        const dim3 source_blocks(
            (source->coefficient.dims[2] + kTileX - 1) / kTileX,
            (source->coefficient.dims[1] + kTileY - 1) / kTileY,
            (source->coefficient.dims[0] + kTileZ - 1) / kTileZ);
        ApplySourceSlab<<<source_blocks, source_threads, 0, stream>>>(
            e_launch, *source, step);
        launch_error = cudaPeekAtLastError();
        if (launch_error != cudaSuccess) break;
      }
      if (source_groups != nullptr) {
        launch_source_groups(0, step);
        if (launch_error != cudaSuccess) break;
      }
      launch_error =
          static_cast<cudaError_t>(BeamzLaunchStreamed(raw_stream, h_launch));
      if (launch_error != cudaSuccess) break;
      if (source_groups != nullptr) {
        launch_source_groups(1, step);
        if (launch_error != cudaSuccess) break;
      }
      launch_error =
          static_cast<cudaError_t>(BeamzLaunchStreamed(raw_stream, e_launch));
      if (launch_error != cudaSuccess) break;
      if (source_groups != nullptr) {
        launch_source_groups(2, step);
        if (launch_error != cudaSuccess) break;
      }
      if (monitor != nullptr) {
        const dim3 monitor_threads(32, 3, 2);
        const dim3 monitor_blocks(
            (monitor->point_count + monitor_threads.x - 1) / monitor_threads.x,
            (monitor->frequency_count + monitor_threads.y - 1) /
                monitor_threads.y,
            (6 + monitor_threads.z - 1) / monitor_threads.z);
        AccumulatePlaneDft<<<monitor_blocks, monitor_threads, 0, stream>>>(
            h_launch, e_launch, *monitor, step, precomputed_monitor_phase);
        launch_error = cudaPeekAtLastError();
        if (launch_error != cudaSuccess) break;
      }
      if (monitor_groups != nullptr) {
        const dim3 monitor_threads(32, 2, 2);
        const dim3 monitor_blocks(
            (monitor_groups->indices.dims[2] + monitor_threads.x - 1) /
                monitor_threads.x,
            (monitor_groups->frequencies.dims[1] + monitor_threads.y - 1) /
                monitor_threads.y,
            (monitor_groups->monitor_count * 6 + monitor_threads.z - 1) /
                monitor_threads.z);
        AccumulateDftGroups<<<monitor_blocks, monitor_threads, 0, stream>>>(
            h_launch, e_launch, *monitor_groups, step);
        launch_error = cudaPeekAtLastError();
        if (launch_error != cudaSuccess) break;
      }
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
  if (error == cudaSuccess && cache_enabled) {
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
    // Keep the executable protected from concurrent eviction until the launch has
    // been submitted to its stream.
    error = cudaGraphLaunch(executable, stream);
  }
  if (error == cudaSuccess && !cache_enabled) {
    error = cudaGraphLaunch(executable, stream);
  }
  if (!cache_enabled && executable != nullptr) {
    cudaGraphExecDestroy(executable);
  }
  if (graph != nullptr) cudaGraphDestroy(graph);
  return error;
}

int BeamzLaunchStreamedSteps(void* raw_stream, const BeamzLaunch& h_launch,
                             const BeamzLaunch& e_launch, int32_t nsteps) {
  return LaunchStreamedGraph(raw_stream, h_launch, e_launch, nullptr, nullptr,
                             0, nullptr, nullptr, nsteps);
}

int BeamzLaunchTemporalSteps(void* raw_stream, const BeamzLaunch& h_ab,
                             const BeamzLaunch& e_ab,
                             const BeamzLaunch& h_ba,
                             const BeamzLaunch& e_ba, int32_t nsteps) {
  if (nsteps < 4 || h_ab.phase != 0 || e_ab.phase != 1 || h_ba.phase != 0 ||
      e_ba.phase != 1 || h_ab.nterms != 0 || e_ab.nterms != 0 ||
      h_ba.nterms != 0 || e_ba.nterms != 0 || h_ab.metric_kind < 0 ||
      h_ab.metric_kind > 2 || e_ab.metric_kind != h_ab.metric_kind ||
      h_ba.metric_kind != h_ab.metric_kind ||
      e_ba.metric_kind != h_ab.metric_kind || h_ab.metallic_edges != 63 ||
      e_ab.metallic_edges != 63 || h_ba.metallic_edges != 63 ||
      e_ba.metallic_edges != 63) {
    return cudaErrorInvalidValue;
  }
  bool scalar_coefficients = true;
  for (int material = 0; material < 6; ++material) {
    const int h_rank = h_ab.inputs[6 + material].rank;
    const int e_rank = e_ab.inputs[6 + material].rank;
    if ((h_rank != 0 && h_rank != 3) || (e_rank != 0 && e_rank != 3)) {
      return cudaErrorInvalidValue;
    }
    scalar_coefficients &= h_rank == 0 && e_rank == 0;
  }

  auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
  cudaError_t error = cudaSuccess;

  int64_t max_x = 0, max_y = 0, max_z = 0;
  for (int component = 0; component < 3; ++component) {
    const BeamzBuffer& h = h_ab.outputs[component];
    const BeamzBuffer& e = e_ab.outputs[component];
    max_x = h.dims[2] > max_x ? h.dims[2] : max_x;
    max_y = h.dims[1] > max_y ? h.dims[1] : max_y;
    max_z = h.dims[0] > max_z ? h.dims[0] : max_z;
    max_x = e.dims[2] > max_x ? e.dims[2] : max_x;
    max_y = e.dims[1] > max_y ? e.dims[1] : max_y;
    max_z = e.dims[0] > max_z ? e.dims[0] : max_z;
  }
  const dim3 threads(kFusedCoreX, kFusedCoreY);
  const dim3 blocks((max_x + kFusedCoreX - 1) / kFusedCoreX,
                    (max_y + kFusedCoreY - 1) / kFusedCoreY,
                    (max_z + kFusedCoreZ - 1) / kFusedCoreZ);

  BeamzLaunch h_tail = h_ab;
  BeamzLaunch e_tail = e_ab;
  for (int component = 0; component < 3; ++component) {
    h_tail.outputs[component] = h_ba.outputs[component];
    e_tail.inputs[3 + component] = h_tail.outputs[component];
    e_tail.outputs[component] = e_ba.outputs[component];
  }

  std::string graph_key = "fused-full-step";
  graph_key += GraphKey(raw_stream, h_ab, e_ab, nsteps);
  graph_key.append(reinterpret_cast<const char*>(&h_ba), sizeof(h_ba));
  graph_key.append(reinterpret_cast<const char*>(&e_ba), sizeof(e_ba));
  GraphCache& cache = CachedGraphs();
  const bool cache_enabled = GraphCacheEnabled();
  if (cache_enabled) {
    std::lock_guard<std::mutex> lock(cache.mutex);
    const auto cached = cache.entries.find(graph_key);
    if (cached != cache.entries.end()) {
      return static_cast<int>(cudaGraphLaunch(cached->second, stream));
    }
  }

  auto launch_steps = [&]() {
    cudaError_t launch_error = cudaSuccess;
    auto launch_fused = [&](const BeamzLaunch& h_launch,
                            const BeamzLaunch& e_launch) {
      if (h_launch.metric_kind == 0) {
        if (scalar_coefficients) {
          FusedFullStepPec<true, 0>
              <<<blocks, threads, kFusedSharedBytes, stream>>>(h_launch,
                                                               e_launch);
        } else {
          FusedFullStepPec<false, 0>
              <<<blocks, threads, kFusedSharedBytes, stream>>>(h_launch,
                                                               e_launch);
        }
      } else if (h_launch.metric_kind == 1) {
        if (scalar_coefficients) {
          FusedFullStepPec<true, 1>
              <<<blocks, threads, kFusedSharedBytes, stream>>>(h_launch,
                                                               e_launch);
        } else {
          FusedFullStepPec<false, 1>
              <<<blocks, threads, kFusedSharedBytes, stream>>>(h_launch,
                                                               e_launch);
        }
      } else if (scalar_coefficients) {
        FusedFullStepPec<true, 2>
            <<<blocks, threads, kFusedSharedBytes, stream>>>(h_launch,
                                                             e_launch);
      } else {
        FusedFullStepPec<false, 2>
            <<<blocks, threads, kFusedSharedBytes, stream>>>(h_launch,
                                                             e_launch);
      }
      return cudaPeekAtLastError();
    };
    const int32_t step_pairs = nsteps / 2;
    for (int32_t pair = 0; pair < step_pairs; ++pair) {
      launch_error = launch_fused(h_ab, e_ab);
      if (launch_error != cudaSuccess) return launch_error;
      launch_error = launch_fused(h_ba, e_ba);
      if (launch_error != cudaSuccess) return launch_error;
    }
    for (int32_t step = 2 * step_pairs; step < nsteps; ++step) {
      launch_error =
          static_cast<cudaError_t>(BeamzLaunchStreamed(raw_stream, h_tail));
      if (launch_error != cudaSuccess) return launch_error;
      launch_error =
          static_cast<cudaError_t>(BeamzLaunchStreamed(raw_stream, e_tail));
      if (launch_error != cudaSuccess) return launch_error;
    }
    return launch_error;
  };

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (error != cudaSuccess) {
    (void)cudaGetLastError();
    return static_cast<int>(launch_steps());
  }
  error = launch_steps();
  const cudaError_t end_error = cudaStreamEndCapture(stream, &graph);
  if (error == cudaSuccess) error = end_error;
  if (error == cudaSuccess) error = cudaGraphInstantiate(&executable, graph, 0);
  if (error == cudaSuccess && cache_enabled) {
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
    error = cudaGraphLaunch(executable, stream);
  }
  if (error == cudaSuccess && !cache_enabled) {
    error = cudaGraphLaunch(executable, stream);
  }
  if (!cache_enabled && executable != nullptr) {
    cudaGraphExecDestroy(executable);
  }
  if (graph != nullptr) cudaGraphDestroy(graph);
  return static_cast<int>(error);
}

int BeamzLaunchStreamedSourceSteps(void* raw_stream,
                                   const BeamzLaunch& h_launch,
                                   const BeamzLaunch& e_launch,
                                   const BeamzSourceLaunch& source,
                                   int32_t nsteps) {
  if (source.coefficient.rank != 3 || source.waveform.rank != 1 ||
      source.current_step.rank != 0 || source.component < 0 ||
      source.component > 2 || source.waveform.dims[0] < 1 ||
      !FitsIntOffsets(source.coefficient) || !FitsIntOffsets(source.waveform)) {
    return cudaErrorInvalidValue;
  }
  const BeamzBuffer& target = e_launch.outputs[source.component];
  for (int axis = 0; axis < 3; ++axis) {
    if (source.starts[axis] < 0 ||
        source.starts[axis] + source.coefficient.dims[axis] >
            target.dims[axis]) {
      return cudaErrorInvalidValue;
    }
  }
  return LaunchStreamedGraph(raw_stream, h_launch, e_launch, &source, nullptr,
                             0, nullptr, nullptr, nsteps);
}

int BeamzLaunchStreamedSourceGroupSteps(
    void* raw_stream, const BeamzLaunch& h_launch,
    const BeamzLaunch& e_launch,
    const BeamzSourceGroupLaunch* source_groups, int32_t source_group_count,
    int32_t nsteps) {
  if (source_groups == nullptr || source_group_count != 9) {
    return cudaErrorInvalidValue;
  }
  for (int32_t index = 0; index < source_group_count; ++index) {
    const BeamzSourceGroupLaunch& group = source_groups[index];
    if (group.component < 0 || group.component > 2 || group.timing < 0 ||
        group.timing > 2 || group.coefficients.rank != 4 ||
        group.waveforms.rank != 2 || group.starts.rank != 2 ||
        group.current_step.rank != 0 ||
        group.coefficients.dims[0] != group.waveforms.dims[0] ||
        group.coefficients.dims[0] != group.starts.dims[0] ||
        group.starts.dims[1] != 3 || group.waveforms.dims[1] < 1 ||
        !FitsIntOffsets(group.coefficients) ||
        !FitsIntOffsets(group.waveforms) || !FitsIntOffsets(group.starts)) {
      return cudaErrorInvalidValue;
    }
  }
  return LaunchStreamedGraph(raw_stream, h_launch, e_launch, nullptr,
                             source_groups, source_group_count, nullptr,
                             nullptr, nsteps);
}

int BeamzLaunchStreamedProgramSteps(
    void* raw_stream, const BeamzLaunch& h_launch,
    const BeamzLaunch& e_launch,
    const BeamzSourceGroupLaunch* source_groups, int32_t source_group_count,
    const BeamzDftGroupLaunch& monitors, int32_t nsteps) {
  if (source_groups == nullptr || source_group_count != 9 ||
      monitors.monitor_count < 1 || monitors.indices.rank != 4 ||
      monitors.weights.rank != 4 || monitors.frequencies.rank != 2 ||
      monitors.component_masks.rank != 2 || monitors.counts.rank != 2 ||
      monitors.codes.rank != 2 || monitors.windows.rank != 2 ||
      monitors.dft_re.rank != 1 || monitors.dft_im.rank != 1 ||
      monitors.dft_weight.rank != 1 || monitors.time.rank != 0 ||
      monitors.current_step.rank != 0 ||
      monitors.indices.dims[0] < monitors.monitor_count ||
      monitors.indices.dims[1] != 6 ||
      monitors.weights.dims[0] != monitors.indices.dims[0] ||
      monitors.weights.dims[1] != monitors.indices.dims[1] ||
      monitors.weights.dims[2] != monitors.indices.dims[2] ||
      monitors.weights.dims[3] != monitors.indices.dims[3] ||
      monitors.frequencies.dims[0] < monitors.monitor_count ||
      monitors.component_masks.dims[0] < monitors.monitor_count ||
      monitors.component_masks.dims[1] != 6 ||
      monitors.counts.dims[0] < monitors.monitor_count ||
      monitors.counts.dims[1] != 5 ||
      monitors.codes.dims[0] < monitors.monitor_count ||
      monitors.codes.dims[1] != 2 ||
      monitors.windows.dims[0] < monitors.monitor_count ||
      monitors.windows.dims[1] != 3 ||
      monitors.dft_re.dims[0] < 1 ||
      monitors.dft_im.dims[0] != monitors.dft_re.dims[0] ||
      monitors.dft_weight.dims[0] < 1) {
    return cudaErrorInvalidValue;
  }
  for (int32_t index = 0; index < source_group_count; ++index) {
    const BeamzSourceGroupLaunch& group = source_groups[index];
    if (group.component < 0 || group.component > 2 || group.timing < 0 ||
        group.timing > 2 || group.coefficients.rank != 4 ||
        group.waveforms.rank != 2 || group.starts.rank != 2 ||
        group.current_step.rank != 0 ||
        group.coefficients.dims[0] != group.waveforms.dims[0] ||
        group.coefficients.dims[0] != group.starts.dims[0] ||
        group.starts.dims[1] != 3 || group.waveforms.dims[1] < 1 ||
        !FitsIntOffsets(group.coefficients) ||
        !FitsIntOffsets(group.waveforms) || !FitsIntOffsets(group.starts)) {
      return cudaErrorInvalidValue;
    }
  }
  const BeamzBuffer monitor_buffers[] = {
      monitors.indices,         monitors.weights, monitors.frequencies,
      monitors.component_masks, monitors.counts,  monitors.codes,
      monitors.windows,         monitors.dft_re,   monitors.dft_im,
      monitors.dft_weight};
  for (const BeamzBuffer& buffer : monitor_buffers) {
    if (!FitsIntOffsets(buffer)) return cudaErrorInvalidValue;
  }
  return LaunchStreamedGraph(raw_stream, h_launch, e_launch, nullptr,
                             source_groups, source_group_count, nullptr,
                             &monitors, nsteps);
}

int BeamzLaunchStreamedSourceMonitorSteps(
    void* raw_stream, const BeamzLaunch& h_launch,
    const BeamzLaunch& e_launch, const BeamzSourceLaunch& source,
    const BeamzDftLaunch& monitor, int32_t nsteps) {
  if (monitor.frequency_count < 1 || monitor.point_count < 1 ||
      monitor.frequencies.rank != 1 || monitor.component_mask.rank != 1 ||
      monitor.dft_re.rank != 4 || monitor.dft_im.rank != 4 ||
      monitor.dft_weight.rank != 2 || monitor.time.rank != 0 ||
      monitor.phase_cos.rank != 2 || monitor.phase_sin.rank != 2 ||
      monitor.phase_cos.dims[0] < nsteps ||
      monitor.phase_sin.dims[0] < nsteps ||
      monitor.phase_cos.dims[1] < monitor.frequency_count ||
      monitor.phase_sin.dims[1] < monitor.frequency_count ||
      !FitsIntOffsets(monitor.phase_cos) ||
      !FitsIntOffsets(monitor.phase_sin)) {
    return cudaErrorInvalidValue;
  }
  for (int component = 0; component < 6; ++component) {
    if (monitor.indices[component].rank != 2 ||
        monitor.weights[component].rank != 2 ||
        monitor.indices[component].dims[0] < monitor.point_count ||
        monitor.weights[component].dims[0] < monitor.point_count ||
        monitor.indices[component].dims[1] !=
            monitor.weights[component].dims[1]) {
      return cudaErrorInvalidValue;
    }
  }
  if (source.coefficient.rank != 3 || source.waveform.rank != 1 ||
      source.current_step.rank != 0 || source.component < 0 ||
      source.component > 2 || source.waveform.dims[0] < 1) {
    return cudaErrorInvalidValue;
  }
  const BeamzBuffer& target = e_launch.outputs[source.component];
  for (int axis = 0; axis < 3; ++axis) {
    if (source.starts[axis] < 0 ||
        source.starts[axis] + source.coefficient.dims[axis] >
            target.dims[axis]) {
      return cudaErrorInvalidValue;
    }
  }
  return LaunchStreamedGraph(raw_stream, h_launch, e_launch, &source, nullptr,
                             0, &monitor, nullptr, nsteps);
}
