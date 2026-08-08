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
  if (dims.size() > 3) {
    return ffi::Error::InvalidArgument(
        "BeamZ CUDA accepts buffers of rank <= 3");
  }
  output->data = value.untyped_data();
  output->rank = static_cast<int32_t>(dims.size());
  output->dims[0] = output->dims[1] = output->dims[2] = 1;
  for (size_t index = 0; index < dims.size(); ++index) {
    output->dims[index] = dims[index];
  }
  return ffi::Error::Success();
}

using Launcher = int (*)(void*, const BeamzLaunch&);

ffi::Error Dispatch(Launcher launcher, void* stream, ffi::RemainingArgs args,
                    ffi::RemainingRets rets, int32_t abi_version, int32_t phase,
                    int32_t nterms, float dt, float resolution,
                    int32_t metallic_edges) {
  if (abi_version != BEAMZ_CUDA_ABI_VERSION) {
    return ffi::Error::InvalidArgument("beamz_cuda ABI version mismatch");
  }
  if (phase < 0 || phase > 1 || (nterms != 0 && nterms != 6)) {
    return ffi::Error::InvalidArgument(
        "invalid BeamZ CUDA phase or CPML term count");
  }
  const size_t input_count = 13 + 4 * static_cast<size_t>(nterms);
  const size_t output_count = 3 + static_cast<size_t>(nterms);
  BeamzLaunch launch{};
  launch.abi_version = abi_version;
  launch.phase = phase;
  launch.nterms = nterms;
  launch.dt = dt;
  launch.resolution = resolution;
  launch.inv_resolution = 1.0f / resolution;
  launch.dt_over_eps = dt / kEps0;
  launch.dt_over_mu = dt / kMu0;
  launch.metallic_edges = metallic_edges;
  for (size_t index = 0; index < input_count; ++index) {
    auto decoded = args.get<ffi::AnyBuffer>(index);
    if (!decoded) return decoded.error();
    if (auto error = DecodeBuffer(*decoded, &launch.inputs[index]);
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
                           float resolution, int32_t metallic_edges) {
  return Dispatch(BeamzLaunchStreamed, stream, args, rets, abi_version, phase,
                  nterms, dt, resolution, metallic_edges);
}

ffi::Error HopperHandler(void* stream, ffi::RemainingArgs args,
                         ffi::RemainingRets rets, int32_t abi_version,
                         int32_t phase, int32_t nterms, float dt,
                         float resolution, int32_t metallic_edges) {
  return Dispatch(BeamzLaunchHopper, stream, args, rets, abi_version, phase,
                  nterms, dt, resolution, metallic_edges);
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
                                  .Attr<int32_t>("metallic_edges"));

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
                                  .Attr<int32_t>("metallic_edges"));
