cmake_minimum_required(VERSION 3.24)
project(beamz_cuda LANGUAGES CXX CUDA)

option(
  BEAMZ_CUDA_FAST_MATH
  "Enable CUDA approximate math intrinsics after hardware parity validation"
  OFF
)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import jax; print(jax.ffi.include_dir())"
  OUTPUT_VARIABLE JAX_FFI_INCLUDE
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)

nanobind_add_module(
  beamz_cuda
  STABLE_ABI
  NB_STATIC
  src/extension.cc
  src/ffi_handler.cc
  src/streamed.cu
  src/hopper.cu
)
target_compile_features(beamz_cuda PRIVATE cxx_std_17)
target_include_directories(beamz_cuda PRIVATE "${JAX_FFI_INCLUDE}" src)
target_compile_definitions(beamz_cuda PRIVATE BEAMZ_CUDA_ABI_VERSION=1)
set_target_properties(
  beamz_cuda
  PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;89;90"
    CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(
  beamz_cuda
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--lineinfo>
)
if(BEAMZ_CUDA_FAST_MATH)
  target_compile_options(beamz_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()
install(TARGETS beamz_cuda LIBRARY DESTINATION .)
