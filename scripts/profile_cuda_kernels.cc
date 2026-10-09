// Optional CUPTI activity injection for scheduler comparisons; not a build
// dependency. Built and exercised with the CUDA 13.3 SDK
// (CUpti_ActivityKernel12). See docs/reviews/cuda-mstep-rtx3090.md for
// build/run instructions.
#include <cupti.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

namespace {
struct Stats {
  unsigned long long count = 0, nanoseconds = 0;
  unsigned registers = 0, shared = 0, gx = 0, gy = 0, bx = 0;
};
std::map<std::string, Stats> stats;
std::mutex guard;

void Request(uint8_t **buffer, size_t *size, size_t *max_records) {
  *size = 8 * 1024 * 1024;
  *buffer = static_cast<uint8_t *>(std::malloc(*size));
  *max_records = 0;
}

void Complete(CUcontext context, uint32_t stream, uint8_t *buffer, size_t,
              size_t valid) {
  CUpti_Activity *record = nullptr;
  std::lock_guard<std::mutex> lock(guard);
  while (cuptiActivityGetNextRecord(buffer, valid, &record) == CUPTI_SUCCESS) {
    if (record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)
      continue;
    const auto *kernel =
        reinterpret_cast<const CUpti_ActivityKernel12 *>(record);
    if (!kernel->start || kernel->end < kernel->start)
      continue;
    auto &s = stats[kernel->name];
    ++s.count;
    s.nanoseconds += kernel->end - kernel->start;
    s.registers = kernel->registersPerThread;
    s.shared = kernel->staticSharedMemory + kernel->dynamicSharedMemory;
    s.gx = kernel->gridX;
    s.gy = kernel->gridY;
    s.bx = kernel->blockX;
  }
  size_t dropped = 0;
  if (cuptiActivityGetNumDroppedRecords(context, stream, &dropped) ==
          CUPTI_SUCCESS &&
      dropped)
    std::fprintf(stderr, "CUPTI dropped %zu activity records\n", dropped);
  std::free(buffer);
}

void Finish() {
  cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
  const char *path = std::getenv("BEAMZ_KERNEL_PROFILE_OUT");
  if (!path)
    return;
  FILE *file = std::fopen(path, "w");
  if (!file) {
    std::perror(path);
    return;
  }
  std::fprintf(file, "name,count,total_ms,mean_us,registers,shared_bytes,grid_"
                     "x,grid_y,block_x\n");
  std::lock_guard<std::mutex> lock(guard);
  for (const auto &[name, s] : stats)
    std::fprintf(file, "\"%s\",%llu,%.6f,%.6f,%u,%u,%u,%u,%u\n", name.c_str(),
                 s.count, s.nanoseconds / 1e6, s.nanoseconds / 1e3 / s.count,
                 s.registers, s.shared, s.gx, s.gy, s.bx);
  std::fclose(file);
}
} // namespace

extern "C" int InitializeInjection() {
  if (!std::getenv("BEAMZ_KERNEL_PROFILE_OUT"))
    return 0;
  auto error = cuptiActivityRegisterCallbacks(Request, Complete);
  if (error == CUPTI_SUCCESS)
    error = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  if (error != CUPTI_SUCCESS) {
    const char *message = nullptr;
    cuptiGetResultString(error, &message);
    std::fprintf(stderr, "CUPTI: %s\n", message);
    return 0;
  }
  std::atexit(Finish);
  return 1;
}
