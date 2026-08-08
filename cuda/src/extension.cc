#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "ffi_handler.h"

namespace nb = nanobind;

NB_MODULE(beamz_cuda, module) {
  module.attr("__version__") = "0.1.0";
  module.def("registrations", []() {
    nb::dict registrations;
    registrations["beamz_cuda_streamed"] =
        nb::capsule(reinterpret_cast<void*>(beamz_cuda_streamed));
    registrations["beamz_cuda_streamed_steps"] =
        nb::capsule(reinterpret_cast<void*>(beamz_cuda_streamed_steps));
    registrations["beamz_cuda_hopper"] =
        nb::capsule(reinterpret_cast<void*>(beamz_cuda_hopper));
    return registrations;
  });
}
