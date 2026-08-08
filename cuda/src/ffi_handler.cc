#include "ffi_handler.h"

#include <cstdint>
#include <string>

#include "launch.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

constexpr float kEps0 = 8.8541878128e-12f;
constexpr float kMu0 = 1.25663706212e-6f;

ffi::Error DecodeBuffer(const ffi::AnyBuffer& value, BeamzBuffer* output) {
  if (value.element_type() != ffi::DataType::F32 &&
      value.element_type() != ffi::DataType::S32) {
    return ffi::Error::InvalidArgument(
        "BeamZ CUDA accepts f32 and s32 buffers");
  }
  const auto dims = value.dimensions();
  if (dims.size() > 4) {
    return ffi::Error::InvalidArgument(
        "BeamZ CUDA accepts buffers of rank <= 4");
  }
  output->data = value.untyped_data();
  output->rank = static_cast<int32_t>(dims.size());
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
  launch.metallic_edges = metallic_edges;
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
    launch.metallic_edges = metallic_edges;
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
    launch.metallic_edges = metallic_edges;
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
    launch.metallic_edges = metallic_edges;
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
  BeamzBuffer inputs[95]{};
  BeamzBuffer outputs[21]{};
  const size_t graph_input_count = cpml_enabled ? 74 : 24;
  const size_t graph_output_count = cpml_enabled ? 18 : 6;
  for (size_t index = 0; index < graph_input_count + 21; ++index) {
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
    launch.metallic_edges = metallic_edges;
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
