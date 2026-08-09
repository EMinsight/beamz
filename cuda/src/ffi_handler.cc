#include "ffi_handler.h"

#include <cstdint>
#include <string>

#include "launch.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

constexpr float kEps0 = 8.8541878128e-12f;
constexpr float kMu0 = 1.25663706212e-6f;

void SetBoundaryCode(BeamzLaunch* launch, int32_t code) {
  launch->metallic_edges = code & 0x3f;
  launch->uniform_cpml_thickness = code >> 8;
}

ffi::Error DecodeBuffer(const ffi::AnyBuffer& value, BeamzBuffer* output) {
  if (value.element_type() != ffi::DataType::F32 &&
      value.element_type() != ffi::DataType::S32 &&
      value.element_type() != ffi::DataType::BF16) {
    return ffi::Error::InvalidArgument(
        "BeamZ CUDA accepts f32, bf16, and s32 buffers");
  }
  const auto dims = value.dimensions();
  if (dims.size() > 4) {
    return ffi::Error::InvalidArgument(
        "BeamZ CUDA accepts buffers of rank <= 4");
  }
  output->data = value.untyped_data();
  output->rank = static_cast<int32_t>(dims.size());
  output->element_type = value.element_type() == ffi::DataType::F32
                             ? kBeamzF32
                         : value.element_type() == ffi::DataType::BF16
                             ? kBeamzBF16
                             : kBeamzS32;
  output->dims[0] = output->dims[1] = output->dims[2] = output->dims[3] = 1;
  for (size_t index = 0; index < dims.size(); ++index) {
    output->dims[index] = dims[index];
  }
  return ffi::Error::Success();
}

using Launcher = int (*)(void*, const BeamzLaunch&);

ffi::Error Dispatch(Launcher launcher, void* stream, ffi::RemainingArgs args,
                    ffi::RemainingRets rets, int32_t abi_version, int32_t phase,
                    int32_t nterms, float dt, float resolution,
                    int32_t metallic_edges, int32_t metric_kind) {
  if (abi_version != BEAMZ_CUDA_ABI_VERSION) {
    return ffi::Error::InvalidArgument("beamz_cuda ABI version mismatch");
  }
  if (phase < 0 || phase > 1 || (nterms != 0 && nterms != 6) ||
      metric_kind < 0 || metric_kind > 2) {
    return ffi::Error::InvalidArgument(
        "invalid BeamZ CUDA phase or CPML term count");
  }
  const size_t payload_count = 13 + 4 * static_cast<size_t>(nterms);
  const size_t output_count = 3 + static_cast<size_t>(nterms);
  BeamzLaunch launch{};
  launch.abi_version = abi_version;
  launch.phase = phase;
  launch.nterms = nterms;
  launch.metric_kind = metric_kind;
  launch.dt = dt;
  launch.resolution = resolution;
  launch.inv_resolution = 1.0f / resolution;
  launch.dt_over_eps = dt / kEps0;
  launch.dt_over_mu = dt / kMu0;
  SetBoundaryCode(&launch, metallic_edges);
  for (size_t index = 0; index < payload_count; ++index) {
    auto decoded = args.get<ffi::AnyBuffer>(index);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(*decoded, &launch.inputs[index]);
        error.failure()) {
      return error;
    }
  }
  for (size_t axis = 0; axis < 3; ++axis) {
    auto decoded = args.get<ffi::AnyBuffer>(payload_count + axis);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(*decoded, &launch.metrics[axis]);
        error.failure()) {
      return error;
    }
  }
  for (size_t index = 0; index < output_count; ++index) {
    auto decoded = rets.get<ffi::AnyBuffer>(index);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(**decoded, &launch.outputs[index]);
        error.failure()) {
      return error;
    }
  }
  const int error = launcher(stream, launch);
  return error == 0 ? ffi::Error::Success()
                    : ffi::Error::Internal("BeamZ CUDA kernel launch failed: " +
                                           std::to_string(error));
}

ffi::Error StreamedHandler(void* stream, ffi::RemainingArgs args,
                           ffi::RemainingRets rets, int32_t abi_version,
                           int32_t phase, int32_t nterms, float dt,
                           float resolution, int32_t metallic_edges,
                           int32_t metric_kind) {
  return Dispatch(BeamzLaunchStreamed, stream, args, rets, abi_version, phase,
                  nterms, dt, resolution, metallic_edges, metric_kind);
}

ffi::Error StreamedStepsHandler(void* stream, ffi::RemainingArgs args,
                                ffi::RemainingRets rets, int32_t abi_version,
                                int32_t nsteps, float dt, float resolution,
                                int32_t metallic_edges, int32_t metric_kind) {
  if (abi_version != BEAMZ_CUDA_ABI_VERSION) {
    return ffi::Error::InvalidArgument("beamz_cuda ABI version mismatch");
  }
  if (nsteps < 1) {
    return ffi::Error::InvalidArgument("BeamZ CUDA step count must be positive");
  }
  if (metric_kind < 0 || metric_kind > 2) {
    return ffi::Error::InvalidArgument("invalid BeamZ CUDA metric kind");
  }
  BeamzBuffer inputs[24]{};
  BeamzBuffer outputs[6]{};
  for (size_t index = 0; index < 24; ++index) {
    auto decoded = args.get<ffi::AnyBuffer>(index);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(*decoded, &inputs[index]); error.failure()) {
      return error;
    }
  }
  for (size_t index = 0; index < 6; ++index) {
    auto decoded = rets.get<ffi::AnyBuffer>(index);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(**decoded, &outputs[index]); error.failure()) {
      return error;
    }
  }

  auto initialize = [&](int32_t phase) {
    BeamzLaunch launch{};
    launch.abi_version = abi_version;
    launch.phase = phase;
    launch.nterms = 0;
    launch.metric_kind = metric_kind;
    launch.dt = dt;
    launch.resolution = resolution;
    launch.inv_resolution = 1.0f / resolution;
    launch.dt_over_eps = dt / kEps0;
    launch.dt_over_mu = dt / kMu0;
    SetBoundaryCode(&launch, metallic_edges);
    return launch;
  };
  BeamzLaunch h_launch = initialize(0);
  BeamzLaunch e_launch = initialize(1);
  for (int component = 0; component < 3; ++component) {
    h_launch.inputs[component] = inputs[component];
    h_launch.inputs[3 + component] = inputs[3 + component];
    h_launch.outputs[component] = outputs[component];
    e_launch.inputs[component] = inputs[3 + component];
    // The E phase consumes the just-updated, aliased magnetic outputs.
    e_launch.inputs[3 + component] = outputs[component];
    e_launch.outputs[component] = outputs[3 + component];
  }
  for (int material = 0; material < 6; ++material) {
    h_launch.inputs[6 + material] = inputs[6 + material];
    e_launch.inputs[6 + material] = inputs[12 + material];
  }
  for (int axis = 0; axis < 3; ++axis) {
    h_launch.metrics[axis] = inputs[18 + axis];
    e_launch.metrics[axis] = inputs[21 + axis];
  }
  const int error =
      BeamzLaunchStreamedSteps(stream, h_launch, e_launch, nsteps);
  return error == 0 ? ffi::Error::Success()
                    : ffi::Error::Internal(
                          "BeamZ CUDA multi-step launch failed: " +
                          std::to_string(error));
}

// Out-of-place temporal updates need a second, XLA-owned field set.  Exposing
// that workspace as aliased results makes device writes visible to XLA instead
// of mutating buffers which the typed FFI contract considers read-only.
ffi::Error TemporalStepsHandler(void* stream, ffi::RemainingArgs args,
                                ffi::RemainingRets rets, int32_t abi_version,
                                int32_t nsteps, float dt, float resolution,
                                int32_t metallic_edges, int32_t metric_kind) {
  if (abi_version != BEAMZ_CUDA_ABI_VERSION) {
    return ffi::Error::InvalidArgument("beamz_cuda ABI version mismatch");
  }
  if (nsteps < 1 || metric_kind < 0 || metric_kind > 2) {
    return ffi::Error::InvalidArgument("invalid BeamZ CUDA temporal attributes");
  }
  BeamzBuffer inputs[30]{};
  BeamzBuffer outputs[12]{};
  for (size_t index = 0; index < 30; ++index) {
    auto decoded = args.get<ffi::AnyBuffer>(index);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(*decoded, &inputs[index]); error.failure()) {
      return error;
    }
  }
  for (size_t index = 0; index < 12; ++index) {
    auto decoded = rets.get<ffi::AnyBuffer>(index);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(**decoded, &outputs[index]); error.failure()) {
      return error;
    }
  }

  auto initialize = [&](int32_t phase) {
    BeamzLaunch launch{};
    launch.abi_version = abi_version;
    launch.phase = phase;
    launch.nterms = 0;
    launch.metric_kind = metric_kind;
    launch.dt = dt;
    launch.resolution = resolution;
    launch.inv_resolution = 1.0f / resolution;
    launch.dt_over_eps = dt / kEps0;
    launch.dt_over_mu = dt / kMu0;
    SetBoundaryCode(&launch, metallic_edges);
    return launch;
  };
  BeamzLaunch h_ab = initialize(0);
  BeamzLaunch e_ab = initialize(1);
  BeamzLaunch h_ba = initialize(0);
  BeamzLaunch e_ba = initialize(1);
  for (int component = 0; component < 3; ++component) {
    h_ab.inputs[component] = inputs[component];
    h_ab.inputs[3 + component] = inputs[3 + component];
    h_ab.outputs[component] = outputs[6 + component];
    e_ab.inputs[component] = inputs[3 + component];
    e_ab.inputs[3 + component] = outputs[6 + component];
    e_ab.outputs[component] = outputs[9 + component];

    h_ba.inputs[component] = outputs[6 + component];
    h_ba.inputs[3 + component] = outputs[9 + component];
    h_ba.outputs[component] = outputs[component];
    e_ba.inputs[component] = outputs[9 + component];
    e_ba.inputs[3 + component] = outputs[component];
    e_ba.outputs[component] = outputs[3 + component];
  }
  for (int material = 0; material < 6; ++material) {
    h_ab.inputs[6 + material] = inputs[12 + material];
    h_ba.inputs[6 + material] = inputs[12 + material];
    e_ab.inputs[6 + material] = inputs[18 + material];
    e_ba.inputs[6 + material] = inputs[18 + material];
  }
  for (int axis = 0; axis < 3; ++axis) {
    h_ab.metrics[axis] = h_ba.metrics[axis] = inputs[24 + axis];
    e_ab.metrics[axis] = e_ba.metrics[axis] = inputs[27 + axis];
  }
  const int error = BeamzLaunchTemporalSteps(stream, h_ab, e_ab, h_ba, e_ba,
                                             nsteps);
  return error == 0 ? ffi::Error::Success()
                    : ffi::Error::Internal(
                          "BeamZ CUDA temporal workspace launch failed: " +
                          std::to_string(error));
}

ffi::Error StreamedCpmlStepsHandler(
    void* stream, ffi::RemainingArgs args, ffi::RemainingRets rets,
    int32_t abi_version, int32_t nsteps, float dt, float resolution,
    int32_t metallic_edges, int32_t metric_kind) {
  if (abi_version != BEAMZ_CUDA_ABI_VERSION) {
    return ffi::Error::InvalidArgument("beamz_cuda ABI version mismatch");
  }
  if (nsteps < 1 || metric_kind < 0 || metric_kind > 2) {
    return ffi::Error::InvalidArgument("BeamZ CUDA step count must be positive");
  }
  BeamzBuffer inputs[74]{};
  BeamzBuffer outputs[18]{};
  for (size_t index = 0; index < 74; ++index) {
    auto decoded = args.get<ffi::AnyBuffer>(index);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(*decoded, &inputs[index]); error.failure()) {
      return error;
    }
  }
  for (size_t index = 0; index < 18; ++index) {
    auto decoded = rets.get<ffi::AnyBuffer>(index);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(**decoded, &outputs[index]); error.failure()) {
      return error;
    }
  }

  auto initialize = [&](int32_t phase) {
    BeamzLaunch launch{};
    launch.abi_version = abi_version;
    launch.phase = phase;
    launch.nterms = 6;
    launch.metric_kind = metric_kind;
    launch.dt = dt;
    launch.resolution = resolution;
    launch.inv_resolution = 1.0f / resolution;
    launch.dt_over_eps = dt / kEps0;
    launch.dt_over_mu = dt / kMu0;
    SetBoundaryCode(&launch, metallic_edges);
    return launch;
  };
  BeamzLaunch h_launch = initialize(0);
  BeamzLaunch e_launch = initialize(1);
  for (int component = 0; component < 3; ++component) {
    h_launch.inputs[component] = inputs[component];
    h_launch.inputs[3 + component] = inputs[3 + component];
    h_launch.outputs[component] = outputs[component];
    e_launch.inputs[component] = inputs[3 + component];
    e_launch.inputs[3 + component] = outputs[component];
    e_launch.outputs[component] = outputs[3 + component];
  }
  for (int index = 0; index < 31; ++index) {
    h_launch.inputs[6 + index] = inputs[6 + index];
    e_launch.inputs[6 + index] = inputs[37 + index];
  }
  for (int term = 0; term < 6; ++term) {
    h_launch.outputs[3 + term] = outputs[6 + term];
    e_launch.outputs[3 + term] = outputs[12 + term];
  }
  for (int axis = 0; axis < 3; ++axis) {
    h_launch.metrics[axis] = inputs[68 + axis];
    e_launch.metrics[axis] = inputs[71 + axis];
  }
  const int error =
      BeamzLaunchStreamedSteps(stream, h_launch, e_launch, nsteps);
  return error == 0 ? ffi::Error::Success()
                    : ffi::Error::Internal(
                          "BeamZ CUDA multi-step CPML launch failed: " +
                          std::to_string(error));
}

ffi::Error StreamedSourceCpmlStepsHandler(
    void* stream, ffi::RemainingArgs args, ffi::RemainingRets rets,
    int32_t abi_version, int32_t nsteps, float dt, float resolution,
    int32_t metallic_edges, int32_t source_component, int32_t source_start_z,
    int32_t source_start_y, int32_t source_start_x, int32_t metric_kind,
    int32_t cpml_enabled) {
  if (abi_version != BEAMZ_CUDA_ABI_VERSION || nsteps < 1 ||
      source_component < 0 || source_component > 2 || metric_kind < 0 ||
      metric_kind > 2 || cpml_enabled < 0 || cpml_enabled > 1) {
    return ffi::Error::InvalidArgument(
        "invalid BeamZ CUDA source-graph attributes");
  }
  BeamzBuffer inputs[77]{};
  BeamzBuffer outputs[18]{};
  const size_t graph_input_count = cpml_enabled ? 74 : 24;
  const size_t graph_output_count = cpml_enabled ? 18 : 6;
  for (size_t index = 0; index < graph_input_count + 3; ++index) {
    auto decoded = args.get<ffi::AnyBuffer>(index);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(*decoded, &inputs[index]); error.failure()) {
      return error;
    }
  }
  for (size_t index = 0; index < graph_output_count; ++index) {
    auto decoded = rets.get<ffi::AnyBuffer>(index);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(**decoded, &outputs[index]); error.failure()) {
      return error;
    }
  }

  auto initialize = [&](int32_t phase) {
    BeamzLaunch launch{};
    launch.abi_version = abi_version;
    launch.phase = phase;
    launch.nterms = cpml_enabled ? 6 : 0;
    launch.metric_kind = metric_kind;
    launch.dt = dt;
    launch.resolution = resolution;
    launch.inv_resolution = 1.0f / resolution;
    launch.dt_over_eps = dt / kEps0;
    launch.dt_over_mu = dt / kMu0;
    SetBoundaryCode(&launch, metallic_edges);
    return launch;
  };
  BeamzLaunch h_launch = initialize(0);
  BeamzLaunch e_launch = initialize(1);
  for (int component = 0; component < 3; ++component) {
    h_launch.inputs[component] = inputs[component];
    h_launch.inputs[3 + component] = inputs[3 + component];
    h_launch.outputs[component] = outputs[component];
    e_launch.inputs[component] = inputs[3 + component];
    e_launch.inputs[3 + component] = outputs[component];
    e_launch.outputs[component] = outputs[3 + component];
  }
  if (cpml_enabled) {
    for (int index = 0; index < 31; ++index) {
      h_launch.inputs[6 + index] = inputs[6 + index];
      e_launch.inputs[6 + index] = inputs[37 + index];
    }
    for (int term = 0; term < 6; ++term) {
      h_launch.outputs[3 + term] = outputs[6 + term];
      e_launch.outputs[3 + term] = outputs[12 + term];
    }
    for (int axis = 0; axis < 3; ++axis) {
      h_launch.metrics[axis] = inputs[68 + axis];
      e_launch.metrics[axis] = inputs[71 + axis];
    }
  } else {
    for (int material = 0; material < 6; ++material) {
      h_launch.inputs[6 + material] = inputs[6 + material];
      e_launch.inputs[6 + material] = inputs[12 + material];
    }
    for (int axis = 0; axis < 3; ++axis) {
      h_launch.metrics[axis] = inputs[18 + axis];
      e_launch.metrics[axis] = inputs[21 + axis];
    }
  }
  BeamzSourceLaunch source{};
  source.coefficient = inputs[graph_input_count];
  source.waveform = inputs[graph_input_count + 1];
  source.current_step = inputs[graph_input_count + 2];
  source.component = source_component;
  source.starts[0] = source_start_z;
  source.starts[1] = source_start_y;
  source.starts[2] = source_start_x;
  const int error = BeamzLaunchStreamedSourceSteps(
      stream, h_launch, e_launch, source, nsteps);
  return error == 0 ? ffi::Error::Success()
                    : ffi::Error::Internal(
                          "BeamZ CUDA source CPML graph launch failed: " +
                          std::to_string(error));
}

ffi::Error StreamedSourceGroupsCpmlStepsHandler(
    void* stream, ffi::RemainingArgs args, ffi::RemainingRets rets,
    int32_t abi_version, int32_t nsteps, float dt, float resolution,
    int32_t metallic_edges, int32_t metric_kind, int32_t cpml_enabled,
    int32_t coincident_source_group_mask) {
  constexpr int32_t kSourceGroupCount = 9;
  if (abi_version != BEAMZ_CUDA_ABI_VERSION || nsteps < 1 ||
      metric_kind < 0 || metric_kind > 2 || cpml_enabled < 0 ||
      cpml_enabled > 1 || coincident_source_group_mask < 0 ||
      coincident_source_group_mask >= (1 << kSourceGroupCount)) {
    return ffi::Error::InvalidArgument(
        "invalid BeamZ CUDA source-group graph attributes");
  }
  BeamzBuffer inputs[102]{};
  BeamzBuffer outputs[18]{};
  const size_t graph_input_count = cpml_enabled ? 74 : 24;
  const size_t graph_output_count = cpml_enabled ? 18 : 6;
  constexpr size_t source_input_count = 3 * kSourceGroupCount + 1;
  for (size_t index = 0; index < graph_input_count + source_input_count;
       ++index) {
    auto decoded = args.get<ffi::AnyBuffer>(index);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(*decoded, &inputs[index]); error.failure()) {
      return error;
    }
  }
  for (size_t index = 0; index < graph_output_count; ++index) {
    auto decoded = rets.get<ffi::AnyBuffer>(index);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(**decoded, &outputs[index]); error.failure()) {
      return error;
    }
  }

  auto initialize = [&](int32_t phase) {
    BeamzLaunch launch{};
    launch.abi_version = abi_version;
    launch.phase = phase;
    launch.nterms = cpml_enabled ? 6 : 0;
    launch.metric_kind = metric_kind;
    launch.dt = dt;
    launch.resolution = resolution;
    launch.inv_resolution = 1.0f / resolution;
    launch.dt_over_eps = dt / kEps0;
    launch.dt_over_mu = dt / kMu0;
    SetBoundaryCode(&launch, metallic_edges);
    return launch;
  };
  BeamzLaunch h_launch = initialize(0);
  BeamzLaunch e_launch = initialize(1);
  for (int component = 0; component < 3; ++component) {
    h_launch.inputs[component] = inputs[component];
    h_launch.inputs[3 + component] = inputs[3 + component];
    h_launch.outputs[component] = outputs[component];
    e_launch.inputs[component] = inputs[3 + component];
    e_launch.inputs[3 + component] = outputs[component];
    e_launch.outputs[component] = outputs[3 + component];
  }
  if (cpml_enabled) {
    for (int index = 0; index < 31; ++index) {
      h_launch.inputs[6 + index] = inputs[6 + index];
      e_launch.inputs[6 + index] = inputs[37 + index];
    }
    for (int term = 0; term < 6; ++term) {
      h_launch.outputs[3 + term] = outputs[6 + term];
      e_launch.outputs[3 + term] = outputs[12 + term];
    }
    for (int axis = 0; axis < 3; ++axis) {
      h_launch.metrics[axis] = inputs[68 + axis];
      e_launch.metrics[axis] = inputs[71 + axis];
    }
  } else {
    for (int material = 0; material < 6; ++material) {
      h_launch.inputs[6 + material] = inputs[6 + material];
      e_launch.inputs[6 + material] = inputs[12 + material];
    }
    for (int axis = 0; axis < 3; ++axis) {
      h_launch.metrics[axis] = inputs[18 + axis];
      e_launch.metrics[axis] = inputs[21 + axis];
    }
  }

  BeamzSourceGroupLaunch groups[kSourceGroupCount]{};
  const BeamzBuffer& current_step =
      inputs[graph_input_count + 3 * kSourceGroupCount];
  for (int32_t index = 0; index < kSourceGroupCount; ++index) {
    groups[index].coefficients = inputs[graph_input_count + 3 * index];
    groups[index].waveforms = inputs[graph_input_count + 3 * index + 1];
    groups[index].starts = inputs[graph_input_count + 3 * index + 2];
    groups[index].current_step = current_step;
    groups[index].timing = index / 3;
    groups[index].component = index % 3;
    groups[index].coincident =
        (coincident_source_group_mask & (1 << index)) != 0;
  }
  const int error = BeamzLaunchStreamedSourceGroupSteps(
      stream, h_launch, e_launch, groups, kSourceGroupCount, nsteps);
  return error == 0
             ? ffi::Error::Success()
             : ffi::Error::Internal(
                   "BeamZ CUDA source-group CPML graph launch failed: " +
                   std::to_string(error));
}

// A second XLA-owned field bank lets the CUDA implementation freeze every
// timestep's inputs.  The native scheduler can then fuse the CPML-free core
// without racing another block that still needs the old magnetic halo.
ffi::Error TemporalSourceGroupsCpmlStepsHandler(
    void* stream, ffi::RemainingArgs args, ffi::RemainingRets rets,
    int32_t abi_version, int32_t nsteps, float dt, float resolution,
    int32_t metallic_edges, int32_t metric_kind,
    int32_t coincident_source_group_mask) {
  constexpr int32_t kSourceGroupCount = 9;
  constexpr size_t kGraphInputCount = 74;
  constexpr size_t kWorkspaceInputCount = 6;
  constexpr size_t kSourceInputCount = 3 * kSourceGroupCount + 1;
  constexpr size_t kInputCount =
      kGraphInputCount + kWorkspaceInputCount + kSourceInputCount;
  constexpr size_t kOutputCount = 24;
  if (abi_version != BEAMZ_CUDA_ABI_VERSION || nsteps < 1 ||
      metric_kind < 0 || metric_kind > 2 ||
      coincident_source_group_mask < 0 ||
      coincident_source_group_mask >= (1 << kSourceGroupCount)) {
    return ffi::Error::InvalidArgument(
        "invalid BeamZ CUDA temporal CPML source-group attributes");
  }
  BeamzBuffer inputs[kInputCount]{};
  BeamzBuffer outputs[kOutputCount]{};
  for (size_t index = 0; index < kInputCount; ++index) {
    auto decoded = args.get<ffi::AnyBuffer>(index);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(*decoded, &inputs[index]); error.failure()) {
      return error;
    }
  }
  for (size_t index = 0; index < kOutputCount; ++index) {
    auto decoded = rets.get<ffi::AnyBuffer>(index);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(**decoded, &outputs[index]); error.failure()) {
      return error;
    }
  }

  auto initialize = [&](int32_t phase) {
    BeamzLaunch launch{};
    launch.abi_version = abi_version;
    launch.phase = phase;
    launch.nterms = 6;
    launch.metric_kind = metric_kind;
    launch.dt = dt;
    launch.resolution = resolution;
    launch.inv_resolution = 1.0f / resolution;
    launch.dt_over_eps = dt / kEps0;
    launch.dt_over_mu = dt / kMu0;
    SetBoundaryCode(&launch, metallic_edges);
    return launch;
  };
  BeamzLaunch h_ab = initialize(0);
  BeamzLaunch e_ab = initialize(1);
  BeamzLaunch h_ba = initialize(0);
  BeamzLaunch e_ba = initialize(1);
  for (int component = 0; component < 3; ++component) {
    // A occupies outputs 0..5 and B occupies outputs 6..11.  Both output
    // groups alias their corresponding input buffers in the typed FFI call.
    h_ab.inputs[component] = outputs[component];
    h_ab.inputs[3 + component] = outputs[3 + component];
    h_ab.outputs[component] = outputs[6 + component];
    e_ab.inputs[component] = outputs[3 + component];
    e_ab.inputs[3 + component] = outputs[6 + component];
    e_ab.outputs[component] = outputs[9 + component];

    h_ba.inputs[component] = outputs[6 + component];
    h_ba.inputs[3 + component] = outputs[9 + component];
    h_ba.outputs[component] = outputs[component];
    e_ba.inputs[component] = outputs[9 + component];
    e_ba.inputs[3 + component] = outputs[component];
    e_ba.outputs[component] = outputs[3 + component];
  }
  for (int index = 0; index < 31; ++index) {
    h_ab.inputs[6 + index] = h_ba.inputs[6 + index] = inputs[6 + index];
    e_ab.inputs[6 + index] = e_ba.inputs[6 + index] = inputs[37 + index];
  }
  for (int term = 0; term < 6; ++term) {
    h_ab.outputs[3 + term] = h_ba.outputs[3 + term] = outputs[12 + term];
    e_ab.outputs[3 + term] = e_ba.outputs[3 + term] = outputs[18 + term];
  }
  for (int axis = 0; axis < 3; ++axis) {
    h_ab.metrics[axis] = h_ba.metrics[axis] = inputs[68 + axis];
    e_ab.metrics[axis] = e_ba.metrics[axis] = inputs[71 + axis];
  }

  BeamzSourceGroupLaunch groups[kSourceGroupCount]{};
  constexpr size_t kSourceOffset = kGraphInputCount + kWorkspaceInputCount;
  const BeamzBuffer& current_step =
      inputs[kSourceOffset + 3 * kSourceGroupCount];
  for (int32_t index = 0; index < kSourceGroupCount; ++index) {
    groups[index].coefficients = inputs[kSourceOffset + 3 * index];
    groups[index].waveforms = inputs[kSourceOffset + 3 * index + 1];
    groups[index].starts = inputs[kSourceOffset + 3 * index + 2];
    groups[index].current_step = current_step;
    groups[index].timing = index / 3;
    groups[index].component = index % 3;
    groups[index].coincident =
        (coincident_source_group_mask & (1 << index)) != 0;
  }
  const int error = BeamzLaunchTemporalCpmlSourceGroupSteps(
      stream, h_ab, e_ab, h_ba, e_ba, groups, kSourceGroupCount, nullptr,
      nsteps);
  return error == 0
             ? ffi::Error::Success()
             : ffi::Error::Internal(
                   "BeamZ CUDA temporal source-group CPML launch failed: " +
                   std::to_string(error));
}

ffi::Error TemporalProgramCpmlStepsHandler(
    void* stream, ffi::RemainingArgs args, ffi::RemainingRets rets,
    int32_t abi_version, int32_t nsteps, float dt, float resolution,
    int32_t metallic_edges, int32_t metric_kind, int32_t monitor_count,
    int32_t coincident_source_group_mask) {
  constexpr int32_t kSourceGroupCount = 9;
  constexpr size_t kGraphInputCount = 74;
  constexpr size_t kWorkspaceInputCount = 6;
  constexpr size_t kSourceInputCount = 3 * kSourceGroupCount;
  constexpr size_t kMonitorInputCount = 12;
  constexpr size_t kInputCount = kGraphInputCount + kWorkspaceInputCount +
                                 kSourceInputCount + kMonitorInputCount;
  constexpr size_t kOutputCount = 27;
  if (abi_version != BEAMZ_CUDA_ABI_VERSION || nsteps < 1 ||
      metric_kind < 0 || metric_kind > 2 || monitor_count < 1 ||
      coincident_source_group_mask < 0 ||
      coincident_source_group_mask >= (1 << kSourceGroupCount)) {
    return ffi::Error::InvalidArgument(
        "invalid BeamZ CUDA temporal CPML program attributes");
  }
  BeamzBuffer inputs[kInputCount]{};
  BeamzBuffer outputs[kOutputCount]{};
  for (size_t index = 0; index < kInputCount; ++index) {
    auto decoded = args.get<ffi::AnyBuffer>(index);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(*decoded, &inputs[index]); error.failure()) {
      return error;
    }
  }
  for (size_t index = 0; index < kOutputCount; ++index) {
    auto decoded = rets.get<ffi::AnyBuffer>(index);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(**decoded, &outputs[index]); error.failure()) {
      return error;
    }
  }

  auto initialize = [&](int32_t phase) {
    BeamzLaunch launch{};
    launch.abi_version = abi_version;
    launch.phase = phase;
    launch.nterms = 6;
    launch.metric_kind = metric_kind;
    launch.dt = dt;
    launch.resolution = resolution;
    launch.inv_resolution = 1.0f / resolution;
    launch.dt_over_eps = dt / kEps0;
    launch.dt_over_mu = dt / kMu0;
    SetBoundaryCode(&launch, metallic_edges);
    return launch;
  };
  BeamzLaunch h_ab = initialize(0);
  BeamzLaunch e_ab = initialize(1);
  BeamzLaunch h_ba = initialize(0);
  BeamzLaunch e_ba = initialize(1);
  for (int component = 0; component < 3; ++component) {
    h_ab.inputs[component] = outputs[component];
    h_ab.inputs[3 + component] = outputs[3 + component];
    h_ab.outputs[component] = outputs[6 + component];
    e_ab.inputs[component] = outputs[3 + component];
    e_ab.inputs[3 + component] = outputs[6 + component];
    e_ab.outputs[component] = outputs[9 + component];

    h_ba.inputs[component] = outputs[6 + component];
    h_ba.inputs[3 + component] = outputs[9 + component];
    h_ba.outputs[component] = outputs[component];
    e_ba.inputs[component] = outputs[9 + component];
    e_ba.inputs[3 + component] = outputs[component];
    e_ba.outputs[component] = outputs[3 + component];
  }
  for (int index = 0; index < 31; ++index) {
    h_ab.inputs[6 + index] = h_ba.inputs[6 + index] = inputs[6 + index];
    e_ab.inputs[6 + index] = e_ba.inputs[6 + index] = inputs[37 + index];
  }
  for (int term = 0; term < 6; ++term) {
    h_ab.outputs[3 + term] = h_ba.outputs[3 + term] = outputs[12 + term];
    e_ab.outputs[3 + term] = e_ba.outputs[3 + term] = outputs[18 + term];
  }
  for (int axis = 0; axis < 3; ++axis) {
    h_ab.metrics[axis] = h_ba.metrics[axis] = inputs[68 + axis];
    e_ab.metrics[axis] = e_ba.metrics[axis] = inputs[71 + axis];
  }

  constexpr size_t kSourceOffset = kGraphInputCount + kWorkspaceInputCount;
  constexpr size_t kMonitorOffset = kSourceOffset + kSourceInputCount;
  const BeamzBuffer& current_step = inputs[kMonitorOffset + 11];
  BeamzSourceGroupLaunch groups[kSourceGroupCount]{};
  for (int32_t index = 0; index < kSourceGroupCount; ++index) {
    groups[index].coefficients = inputs[kSourceOffset + 3 * index];
    groups[index].waveforms = inputs[kSourceOffset + 3 * index + 1];
    groups[index].starts = inputs[kSourceOffset + 3 * index + 2];
    groups[index].current_step = current_step;
    groups[index].timing = index / 3;
    groups[index].component = index % 3;
    groups[index].coincident =
        (coincident_source_group_mask & (1 << index)) != 0;
  }
  BeamzDftGroupLaunch monitors{};
  monitors.indices = inputs[kMonitorOffset];
  monitors.weights = inputs[kMonitorOffset + 1];
  monitors.frequencies = inputs[kMonitorOffset + 2];
  monitors.component_masks = inputs[kMonitorOffset + 3];
  monitors.counts = inputs[kMonitorOffset + 4];
  monitors.codes = inputs[kMonitorOffset + 5];
  monitors.windows = inputs[kMonitorOffset + 6];
  monitors.dft_re = outputs[24];
  monitors.dft_im = outputs[25];
  monitors.dft_weight = outputs[26];
  monitors.time = inputs[kMonitorOffset + 10];
  monitors.current_step = current_step;
  monitors.monitor_count = monitor_count;

  const int error = BeamzLaunchTemporalCpmlSourceGroupSteps(
      stream, h_ab, e_ab, h_ba, e_ba, groups, kSourceGroupCount, &monitors,
      nsteps);
  return error == 0
             ? ffi::Error::Success()
             : ffi::Error::Internal(
                   "BeamZ CUDA temporal CPML program launch failed: " +
                   std::to_string(error));
}

ffi::Error StreamedProgramCpmlStepsHandler(
    void* stream, ffi::RemainingArgs args, ffi::RemainingRets rets,
    int32_t abi_version, int32_t nsteps, float dt, float resolution,
    int32_t metallic_edges, int32_t metric_kind, int32_t cpml_enabled,
    int32_t monitor_count, int32_t coincident_source_group_mask) {
  constexpr int32_t kSourceGroupCount = 9;
  if (abi_version != BEAMZ_CUDA_ABI_VERSION || nsteps < 1 ||
      metric_kind < 0 || metric_kind > 2 || cpml_enabled < 0 ||
      cpml_enabled > 1 || monitor_count < 1 ||
      coincident_source_group_mask < 0 ||
      coincident_source_group_mask >= (1 << kSourceGroupCount)) {
    return ffi::Error::InvalidArgument(
        "invalid BeamZ CUDA program graph attributes");
  }
  BeamzBuffer inputs[113]{};
  BeamzBuffer outputs[21]{};
  const size_t graph_input_count = cpml_enabled ? 74 : 24;
  const size_t graph_output_count = cpml_enabled ? 18 : 6;
  constexpr size_t source_input_count = 3 * kSourceGroupCount;
  constexpr size_t monitor_input_count = 12;
  for (size_t index = 0;
       index < graph_input_count + source_input_count + monitor_input_count;
       ++index) {
    auto decoded = args.get<ffi::AnyBuffer>(index);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(*decoded, &inputs[index]); error.failure()) {
      return error;
    }
  }
  for (size_t index = 0; index < graph_output_count + 3; ++index) {
    auto decoded = rets.get<ffi::AnyBuffer>(index);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(**decoded, &outputs[index]); error.failure()) {
      return error;
    }
  }

  auto initialize = [&](int32_t phase) {
    BeamzLaunch launch{};
    launch.abi_version = abi_version;
    launch.phase = phase;
    launch.nterms = cpml_enabled ? 6 : 0;
    launch.metric_kind = metric_kind;
    launch.dt = dt;
    launch.resolution = resolution;
    launch.inv_resolution = 1.0f / resolution;
    launch.dt_over_eps = dt / kEps0;
    launch.dt_over_mu = dt / kMu0;
    SetBoundaryCode(&launch, metallic_edges);
    return launch;
  };
  BeamzLaunch h_launch = initialize(0);
  BeamzLaunch e_launch = initialize(1);
  for (int component = 0; component < 3; ++component) {
    h_launch.inputs[component] = inputs[component];
    h_launch.inputs[3 + component] = inputs[3 + component];
    h_launch.outputs[component] = outputs[component];
    e_launch.inputs[component] = inputs[3 + component];
    e_launch.inputs[3 + component] = outputs[component];
    e_launch.outputs[component] = outputs[3 + component];
  }
  if (cpml_enabled) {
    for (int index = 0; index < 31; ++index) {
      h_launch.inputs[6 + index] = inputs[6 + index];
      e_launch.inputs[6 + index] = inputs[37 + index];
    }
    for (int term = 0; term < 6; ++term) {
      h_launch.outputs[3 + term] = outputs[6 + term];
      e_launch.outputs[3 + term] = outputs[12 + term];
    }
    for (int axis = 0; axis < 3; ++axis) {
      h_launch.metrics[axis] = inputs[68 + axis];
      e_launch.metrics[axis] = inputs[71 + axis];
    }
  } else {
    for (int material = 0; material < 6; ++material) {
      h_launch.inputs[6 + material] = inputs[6 + material];
      e_launch.inputs[6 + material] = inputs[12 + material];
    }
    for (int axis = 0; axis < 3; ++axis) {
      h_launch.metrics[axis] = inputs[18 + axis];
      e_launch.metrics[axis] = inputs[21 + axis];
    }
  }

  const size_t monitor_start = graph_input_count + source_input_count;
  const BeamzBuffer& current_step = inputs[monitor_start + 11];
  BeamzSourceGroupLaunch groups[kSourceGroupCount]{};
  for (int32_t index = 0; index < kSourceGroupCount; ++index) {
    groups[index].coefficients = inputs[graph_input_count + 3 * index];
    groups[index].waveforms = inputs[graph_input_count + 3 * index + 1];
    groups[index].starts = inputs[graph_input_count + 3 * index + 2];
    groups[index].current_step = current_step;
    groups[index].timing = index / 3;
    groups[index].component = index % 3;
    groups[index].coincident =
        (coincident_source_group_mask & (1 << index)) != 0;
  }
  BeamzDftGroupLaunch monitors{};
  monitors.indices = inputs[monitor_start];
  monitors.weights = inputs[monitor_start + 1];
  monitors.frequencies = inputs[monitor_start + 2];
  monitors.component_masks = inputs[monitor_start + 3];
  monitors.counts = inputs[monitor_start + 4];
  monitors.codes = inputs[monitor_start + 5];
  monitors.windows = inputs[monitor_start + 6];
  monitors.dft_re = inputs[monitor_start + 7];
  monitors.dft_im = inputs[monitor_start + 8];
  monitors.dft_weight = inputs[monitor_start + 9];
  monitors.time = inputs[monitor_start + 10];
  monitors.current_step = current_step;
  monitors.monitor_count = monitor_count;

  const int error = BeamzLaunchStreamedProgramSteps(
      stream, h_launch, e_launch, groups, kSourceGroupCount, monitors, nsteps);
  return error == 0
             ? ffi::Error::Success()
             : ffi::Error::Internal(
                   "BeamZ CUDA program graph launch failed: " +
                   std::to_string(error));
}

ffi::Error StreamedSourceMonitorCpmlStepsHandler(
    void* stream, ffi::RemainingArgs args, ffi::RemainingRets rets,
    int32_t abi_version, int32_t nsteps, float dt, float resolution,
    int32_t metallic_edges, int32_t source_component, int32_t source_start_z,
    int32_t source_start_y, int32_t source_start_x, int32_t frequency_count,
    int32_t point_count, int32_t metric_kind, int32_t cpml_enabled) {
  if (abi_version != BEAMZ_CUDA_ABI_VERSION || nsteps < 1 ||
      source_component < 0 || source_component > 2 || frequency_count < 1 ||
      point_count < 1 || metric_kind < 0 || metric_kind > 2 ||
      cpml_enabled < 0 || cpml_enabled > 1) {
    return ffi::Error::InvalidArgument(
        "invalid BeamZ CUDA source-monitor graph attributes");
  }
  BeamzBuffer inputs[97]{};
  BeamzBuffer outputs[23]{};
  const size_t graph_input_count = cpml_enabled ? 74 : 24;
  const size_t graph_output_count = cpml_enabled ? 18 : 6;
  for (size_t index = 0; index < graph_input_count + 23; ++index) {
    auto decoded = args.get<ffi::AnyBuffer>(index);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(*decoded, &inputs[index]); error.failure()) {
      return error;
    }
  }
  for (size_t index = 0; index < graph_output_count + 5; ++index) {
    auto decoded = rets.get<ffi::AnyBuffer>(index);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(**decoded, &outputs[index]); error.failure()) {
      return error;
    }
  }

  auto initialize = [&](int32_t phase) {
    BeamzLaunch launch{};
    launch.abi_version = abi_version;
    launch.phase = phase;
    launch.nterms = cpml_enabled ? 6 : 0;
    launch.metric_kind = metric_kind;
    launch.dt = dt;
    launch.resolution = resolution;
    launch.inv_resolution = 1.0f / resolution;
    launch.dt_over_eps = dt / kEps0;
    launch.dt_over_mu = dt / kMu0;
    SetBoundaryCode(&launch, metallic_edges);
    return launch;
  };
  BeamzLaunch h_launch = initialize(0);
  BeamzLaunch e_launch = initialize(1);
  for (int component = 0; component < 3; ++component) {
    h_launch.inputs[component] = inputs[component];
    h_launch.inputs[3 + component] = inputs[3 + component];
    h_launch.outputs[component] = outputs[component];
    e_launch.inputs[component] = inputs[3 + component];
    e_launch.inputs[3 + component] = outputs[component];
    e_launch.outputs[component] = outputs[3 + component];
  }
  if (cpml_enabled) {
    for (int index = 0; index < 31; ++index) {
      h_launch.inputs[6 + index] = inputs[6 + index];
      e_launch.inputs[6 + index] = inputs[37 + index];
    }
    for (int term = 0; term < 6; ++term) {
      h_launch.outputs[3 + term] = outputs[6 + term];
      e_launch.outputs[3 + term] = outputs[12 + term];
    }
    for (int axis = 0; axis < 3; ++axis) {
      h_launch.metrics[axis] = inputs[68 + axis];
      e_launch.metrics[axis] = inputs[71 + axis];
    }
  } else {
    for (int material = 0; material < 6; ++material) {
      h_launch.inputs[6 + material] = inputs[6 + material];
      e_launch.inputs[6 + material] = inputs[12 + material];
    }
    for (int axis = 0; axis < 3; ++axis) {
      h_launch.metrics[axis] = inputs[18 + axis];
      e_launch.metrics[axis] = inputs[21 + axis];
    }
  }
  BeamzSourceLaunch source{};
  source.coefficient = inputs[graph_input_count];
  source.waveform = inputs[graph_input_count + 1];
  source.current_step = inputs[graph_input_count + 2];
  source.component = source_component;
  source.starts[0] = source_start_z;
  source.starts[1] = source_start_y;
  source.starts[2] = source_start_x;
  BeamzDftLaunch monitor{};
  for (int component = 0; component < 6; ++component) {
    monitor.indices[component] = inputs[graph_input_count + 3 + component];
    monitor.weights[component] = inputs[graph_input_count + 9 + component];
  }
  monitor.frequencies = inputs[graph_input_count + 15];
  monitor.component_mask = inputs[graph_input_count + 16];
  monitor.dft_re = outputs[graph_output_count];
  monitor.dft_im = outputs[graph_output_count + 1];
  monitor.dft_weight = outputs[graph_output_count + 2];
  monitor.time = inputs[graph_input_count + 20];
  monitor.phase_cos = outputs[graph_output_count + 3];
  monitor.phase_sin = outputs[graph_output_count + 4];
  monitor.frequency_count = frequency_count;
  monitor.point_count = point_count;
  const int error = BeamzLaunchStreamedSourceMonitorSteps(
      stream, h_launch, e_launch, source, monitor, nsteps);
  return error == 0 ? ffi::Error::Success()
                    : ffi::Error::Internal(
                          "BeamZ CUDA source-monitor graph launch failed: " +
                          std::to_string(error));
}

ffi::Error HopperHandler(void* stream, ffi::RemainingArgs args,
                         ffi::RemainingRets rets, int32_t abi_version,
                         int32_t phase, int32_t nterms, float dt,
                         float resolution, int32_t metallic_edges,
                         int32_t metric_kind) {
  return Dispatch(BeamzLaunchHopper, stream, args, rets, abi_version, phase,
                  nterms, dt, resolution, metallic_edges, metric_kind);
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(beamz_cuda_streamed, StreamedHandler,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<void*>>()
                                  .RemainingArgs()
                                  .RemainingRets()
                                  .Attr<int32_t>("abi_version")
                                  .Attr<int32_t>("phase")
                                  .Attr<int32_t>("nterms")
                                  .Attr<float>("dt")
                                  .Attr<float>("resolution")
                                  .Attr<int32_t>("metallic_edges")
                                  .Attr<int32_t>("metric_kind"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(beamz_cuda_streamed_steps, StreamedStepsHandler,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<void*>>()
                                  .RemainingArgs()
                                  .RemainingRets()
                                  .Attr<int32_t>("abi_version")
                                  .Attr<int32_t>("nsteps")
                                  .Attr<float>("dt")
                                  .Attr<float>("resolution")
                                  .Attr<int32_t>("metallic_edges")
                                  .Attr<int32_t>("metric_kind"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(beamz_cuda_temporal_steps, TemporalStepsHandler,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<void*>>()
                                  .RemainingArgs()
                                  .RemainingRets()
                                  .Attr<int32_t>("abi_version")
                                  .Attr<int32_t>("nsteps")
                                  .Attr<float>("dt")
                                  .Attr<float>("resolution")
                                  .Attr<int32_t>("metallic_edges")
                                  .Attr<int32_t>("metric_kind"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    beamz_cuda_streamed_cpml_steps, StreamedCpmlStepsHandler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<void*>>()
        .RemainingArgs()
        .RemainingRets()
        .Attr<int32_t>("abi_version")
        .Attr<int32_t>("nsteps")
        .Attr<float>("dt")
        .Attr<float>("resolution")
        .Attr<int32_t>("metallic_edges")
        .Attr<int32_t>("metric_kind"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    beamz_cuda_streamed_source_cpml_steps, StreamedSourceCpmlStepsHandler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<void*>>()
        .RemainingArgs()
        .RemainingRets()
        .Attr<int32_t>("abi_version")
        .Attr<int32_t>("nsteps")
        .Attr<float>("dt")
        .Attr<float>("resolution")
        .Attr<int32_t>("metallic_edges")
        .Attr<int32_t>("source_component")
        .Attr<int32_t>("source_start_z")
        .Attr<int32_t>("source_start_y")
        .Attr<int32_t>("source_start_x")
        .Attr<int32_t>("metric_kind")
        .Attr<int32_t>("cpml_enabled"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    beamz_cuda_streamed_source_groups_cpml_steps,
    StreamedSourceGroupsCpmlStepsHandler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<void*>>()
        .RemainingArgs()
        .RemainingRets()
        .Attr<int32_t>("abi_version")
        .Attr<int32_t>("nsteps")
        .Attr<float>("dt")
        .Attr<float>("resolution")
        .Attr<int32_t>("metallic_edges")
        .Attr<int32_t>("metric_kind")
        .Attr<int32_t>("cpml_enabled")
        .Attr<int32_t>("coincident_source_group_mask"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    beamz_cuda_temporal_source_groups_cpml_steps,
    TemporalSourceGroupsCpmlStepsHandler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<void*>>()
        .RemainingArgs()
        .RemainingRets()
        .Attr<int32_t>("abi_version")
        .Attr<int32_t>("nsteps")
        .Attr<float>("dt")
        .Attr<float>("resolution")
        .Attr<int32_t>("metallic_edges")
        .Attr<int32_t>("metric_kind")
        .Attr<int32_t>("coincident_source_group_mask"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    beamz_cuda_temporal_program_cpml_steps, TemporalProgramCpmlStepsHandler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<void*>>()
        .RemainingArgs()
        .RemainingRets()
        .Attr<int32_t>("abi_version")
        .Attr<int32_t>("nsteps")
        .Attr<float>("dt")
        .Attr<float>("resolution")
        .Attr<int32_t>("metallic_edges")
        .Attr<int32_t>("metric_kind")
        .Attr<int32_t>("monitor_count")
        .Attr<int32_t>("coincident_source_group_mask"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    beamz_cuda_streamed_program_cpml_steps, StreamedProgramCpmlStepsHandler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<void*>>()
        .RemainingArgs()
        .RemainingRets()
        .Attr<int32_t>("abi_version")
        .Attr<int32_t>("nsteps")
        .Attr<float>("dt")
        .Attr<float>("resolution")
        .Attr<int32_t>("metallic_edges")
        .Attr<int32_t>("metric_kind")
        .Attr<int32_t>("cpml_enabled")
        .Attr<int32_t>("monitor_count")
        .Attr<int32_t>("coincident_source_group_mask"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    beamz_cuda_streamed_source_monitor_cpml_steps,
    StreamedSourceMonitorCpmlStepsHandler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<void*>>()
        .RemainingArgs()
        .RemainingRets()
        .Attr<int32_t>("abi_version")
        .Attr<int32_t>("nsteps")
        .Attr<float>("dt")
        .Attr<float>("resolution")
        .Attr<int32_t>("metallic_edges")
        .Attr<int32_t>("source_component")
        .Attr<int32_t>("source_start_z")
        .Attr<int32_t>("source_start_y")
        .Attr<int32_t>("source_start_x")
        .Attr<int32_t>("frequency_count")
        .Attr<int32_t>("point_count")
        .Attr<int32_t>("metric_kind")
        .Attr<int32_t>("cpml_enabled"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(beamz_cuda_hopper, HopperHandler,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<void*>>()
                                  .RemainingArgs()
                                  .RemainingRets()
                                  .Attr<int32_t>("abi_version")
                                  .Attr<int32_t>("phase")
                                  .Attr<int32_t>("nterms")
                                  .Attr<float>("dt")
                                  .Attr<float>("resolution")
                                  .Attr<int32_t>("metallic_edges")
                                  .Attr<int32_t>("metric_kind"));
