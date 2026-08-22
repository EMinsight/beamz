#include "graph.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace {

constexpr size_t kMaxCachedGraphs = 32;

struct GraphCache {
  std::mutex mutex;
  std::unordered_map<std::string, cudaGraphExec_t> entries;
};

GraphCache& CachedGraphs() {
  // CUDA contexts may already be gone when static destructors run. Retain this
  // deliberately bounded cache until process teardown instead.
  static auto* cache = new GraphCache();
  return *cache;
}

template <typename T>
void Append(std::string* key, const T& value) {
  key->append(reinterpret_cast<const char*>(&value), sizeof(value));
}

void AppendBuffer(std::string* key, const BeamzBuffer& value) {
  Append(key, value.data);
  Append(key, value.rank);
  Append(key, value.element_type);
  for (int axis = 0; axis < 4; ++axis) Append(key, value.dims[axis]);
}

void AppendLaunch(std::string* key, const BeamzLaunch& value) {
  Append(key, value.abi_version);
  Append(key, value.cuda_flags);
  Append(key, value.phase);
  Append(key, value.nterms);
  Append(key, value.metric_kind);
  Append(key, value.dt);
  Append(key, value.resolution);
  Append(key, value.inv_resolution);
  Append(key, value.dt_over_eps);
  Append(key, value.dt_over_mu);
  Append(key, value.metallic_edges);
  Append(key, value.uniform_cpml_thickness);
  for (const BeamzBuffer& buffer : value.inputs) AppendBuffer(key, buffer);
  for (const BeamzBuffer& buffer : value.metrics) AppendBuffer(key, buffer);
  for (const BeamzBuffer& buffer : value.outputs) AppendBuffer(key, buffer);
}

void AppendSourceGroup(std::string* key,
                       const BeamzSourceGroupLaunch& value) {
  AppendBuffer(key, value.coefficients);
  AppendBuffer(key, value.waveforms);
  AppendBuffer(key, value.starts);
  AppendBuffer(key, value.current_step);
  Append(key, value.component);
  Append(key, value.timing);
  Append(key, value.coincident);
}

void AppendMonitors(std::string* key, const BeamzDftGroupLaunch& value) {
  AppendBuffer(key, value.indices);
  AppendBuffer(key, value.weights);
  AppendBuffer(key, value.frequencies);
  AppendBuffer(key, value.component_masks);
  AppendBuffer(key, value.counts);
  AppendBuffer(key, value.codes);
  AppendBuffer(key, value.windows);
  AppendBuffer(key, value.dft_re);
  AppendBuffer(key, value.dft_im);
  AppendBuffer(key, value.dft_weight);
  AppendBuffer(key, value.time);
  AppendBuffer(key, value.current_step);
  Append(key, value.monitor_count);
}

}  // namespace

std::string BeamzGraphKey(const char* schedule, void* stream,
                          const BeamzProgramLaunch& program) {
  std::string key(schedule);
  key.push_back('\0');
  Append(&key, stream);
  Append(&key, program.field_bank_count);
  Append(&key, program.nsteps);
  AppendLaunch(&key, program.h_ab);
  AppendLaunch(&key, program.e_ab);
  if (program.field_bank_count == 2) {
    AppendLaunch(&key, program.h_ba);
    AppendLaunch(&key, program.e_ba);
  }
  Append(&key, program.source_group_count);
  for (int32_t index = 0; index < program.source_group_count; ++index) {
    AppendSourceGroup(&key, program.source_groups[index]);
  }
  const bool has_monitors = program.monitors != nullptr;
  Append(&key, has_monitors);
  if (has_monitors) AppendMonitors(&key, *program.monitors);
  return key;
}

cudaError_t BeamzLaunchGraph(cudaStream_t stream, const std::string& key,
                             bool cache_enabled,
                             const std::function<cudaError_t()>& enqueue) {
  GraphCache& cache = CachedGraphs();
  if (cache_enabled) {
    std::lock_guard<std::mutex> lock(cache.mutex);
    const auto cached = cache.entries.find(key);
    if (cached != cache.entries.end()) {
      return cudaGraphLaunch(cached->second, stream);
    }
  }

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  cudaError_t error =
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (error != cudaSuccess) {
    (void)cudaGetLastError();
    return enqueue();
  }
  error = enqueue();
  const cudaError_t end_error = cudaStreamEndCapture(stream, &graph);
  if (error == cudaSuccess) error = end_error;
  if (error == cudaSuccess) error = cudaGraphInstantiate(&executable, graph, 0);
  if (error == cudaSuccess && cache_enabled) {
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (cache.entries.size() >= kMaxCachedGraphs) {
      for (const auto& entry : cache.entries) {
        cudaGraphExecDestroy(entry.second);
      }
      cache.entries.clear();
    }
    const auto [entry, inserted] = cache.entries.emplace(key, executable);
    if (!inserted) {
      cudaGraphExecDestroy(executable);
      executable = entry->second;
    }
    // Keep the executable protected from concurrent eviction until submission.
    error = cudaGraphLaunch(executable, stream);
  } else if (error == cudaSuccess) {
    error = cudaGraphLaunch(executable, stream);
  }
  if (!cache_enabled && executable != nullptr) {
    cudaGraphExecDestroy(executable);
  }
  if (graph != nullptr) cudaGraphDestroy(graph);
  return error;
}
