#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "abi_layout.h"
#include "ffi_handler.h"

namespace nb = nanobind;
using namespace beamz::cuda::abi;

NB_MODULE(beamz_cuda, module) {
  module.attr("__version__") = kPackageVersion;
  module.attr("__abi_version__") = kAbiVersion;
  module.def("registrations", []() {
    nb::dict registrations;
    registrations[kStreamedTarget] =
        nb::capsule(reinterpret_cast<void*>(beamz_cuda_streamed));
    registrations[kStreamedStepsTarget] =
        nb::capsule(reinterpret_cast<void*>(beamz_cuda_streamed_steps));
    registrations[kTemporalStepsTarget] =
        nb::capsule(reinterpret_cast<void*>(beamz_cuda_temporal_steps));
    registrations[kStreamedCpmlStepsTarget] =
        nb::capsule(reinterpret_cast<void*>(beamz_cuda_streamed_cpml_steps));
    registrations[kStreamedSourceGroupsCpmlStepsTarget] = nb::capsule(
        reinterpret_cast<void*>(
            beamz_cuda_streamed_source_groups_cpml_steps));
    registrations[kTemporalSourceGroupsCpmlStepsTarget] = nb::capsule(
        reinterpret_cast<void*>(
            beamz_cuda_temporal_source_groups_cpml_steps));
    registrations[kTemporalProgramCpmlStepsTarget] = nb::capsule(
        reinterpret_cast<void*>(beamz_cuda_temporal_program_cpml_steps));
    registrations[kStreamedProgramCpmlStepsTarget] = nb::capsule(
        reinterpret_cast<void*>(beamz_cuda_streamed_program_cpml_steps));
    registrations[kHopperTarget] =
        nb::capsule(reinterpret_cast<void*>(beamz_cuda_hopper));
    return registrations;
  });
}
