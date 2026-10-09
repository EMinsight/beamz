// Whole-column K=2 tiles, following PR #308's temporal.fut by Pepijn de Vos.
// Shared rings hold H1, E1 and H2; E0 is loaded once into a shared window.
// Only eligible 8x16 quadrants publish H2/E2.
// Boundary/source/monitor work uses the existing CUDA kernels through compact
// BeamzBuffer addressing; this file defines no new numerical conventions.
constexpr int kMstepWindowX = 36, kMstepWindowY = 20;
constexpr int kMstepThreads = kMstepWindowX * kMstepWindowY;
constexpr int kMstepInputOffset = PairStage<32, 16, 3>::offset;
constexpr size_t kMstepSharedBytes =
    (kMstepInputOffset + 3 * kMstepThreads) * sizeof(float);

__device__ __forceinline__ float &MstepInput(float *shared, int c, int y,
                                             int x) {
  return shared[kMstepInputOffset + c * kMstepThreads + y * kMstepWindowX + x];
}
template <bool Packed, int K = 0>
__device__ __forceinline__ void
MstepStages(float *shared, const FusedYeePhase &h, const FusedYeePhase &e,
            const PairFieldOutputs &final, int ox, int oy, int oz, int wave,
            int mask) {
  using Here = PairStage<32, 16, K>;
  constexpr int phase = K % 2, shift = phase == 1 ? 1 : 0;
  constexpr int sign = phase == 0 ? 1 : -1;
  const auto &launch = phase == 0 ? h : e;
  const int z = oz + wave - 2 - K / 2;
  if (wave >= K) {
    for (int i = threadIdx.x; i < Here::sx * Here::sy; i += blockDim.x) {
      const int ix = i % Here::sx, iy = i / Here::sx;
      const int x = ox - 2 + (K + 1) / 2 + ix;
      const int y = oy - 2 + (K + 1) / 2 + iy;
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        float next = 0.f;
        const auto &out = launch.outputs[c];
        if (BufferContains(out, z, y, x)) {
          auto other = [&](int component, int dz, int dy, int dx) {
            if constexpr (K == 0) {
              if (!dz)
                return MstepInput(shared, component, iy + dy, ix + dx);
              const auto &f = h.inputs[3 + component];
              return BufferContains(f, z + dz, y + dy, x + dx)
                         ? Read3D(f, z + dz, y + dy, x + dx)
                         : 0.f;
            } else {
              return PairStage<32, 16, K - 1>::At(
                  shared, component, z + dz, iy + shift + dy, ix + shift + dx);
            }
          };
          auto difference = [&](int component, int axis) {
            const float a = other(component, 0, 0, 0);
            const float b = other(component, axis == 0 ? sign : 0,
                                  axis == 1 ? sign : 0, axis == 2 ? sign : 0);
            return beamz::cuda::yee::ScaleYeeDifference(
                phase == 0 ? b - a : a - b, launch.inv_resolution);
          };
          const auto a = beamz::cuda::yee::FirstCurlTerm(c);
          const auto b = beamz::cuda::yee::SecondCurlTerm(c);
          const float curl = difference(a.source_component, a.derivative_axis) -
                             difference(b.source_component, b.derivative_axis);
          float old;
          if constexpr (K == 1)
            old = MstepInput(shared, c, iy + 1, ix + 1);
          else if constexpr (K == 0)
            old = Read3D(launch.inputs[c], z, y, x);
          else
            old = PairStage<32, 16, K - 2>::At(shared, c, z, iy + 1, ix + 1);
          const int logical = (z * static_cast<int>(out.dims[1]) + y) *
                                  static_cast<int>(out.dims[2]) +
                              x;
          const float decay =
              phase == 1 && Packed ? 1.f : Read(launch.inputs[6 + c], z, y, x);
          const float source =
              phase == 1 && Packed
                  ? beamz::cuda::yee::PackedMaterialSource(
                        launch.inputs[6 + c], launch.inputs[9 + c], logical)
                  : Read(launch.inputs[9 + c], z, y, x);
          next = beamz::cuda::yee::AdvanceYeeField(phase, old, decay, source,
                                                   curl);
        }
        if constexpr (K == 3) {
          const int quadrant = (iy >= 8 ? 2 : 0) + (ix >= 16 ? 1 : 0);
          if (mask & (1 << quadrant)) {
            const auto &ef = final.fields[3 + c];
            const auto &hf = final.fields[c];
            static_cast<float *>(ef.data)[BeamzOffset3D(ef, z, y, x)] = next;
            static_cast<float *>(hf.data)[BeamzOffset3D(hf, z, y, x)] =
                PairStage<32, 16, 2>::At(shared, c, z, iy + 1, ix + 1);
          }
        } else
          Here::At(shared, c, z, iy, ix) = next;
      }
    }
  }
  __syncthreads();
  if constexpr (K < 3)
    MstepStages<Packed, K + 1>(shared, h, e, final, ox, oy, oz, wave, mask);
}

template <bool Packed>
__global__ __launch_bounds__(kMstepThreads) void MstepColumnCore(
    const __grid_constant__ FusedYeePhase h,
    const __grid_constant__ FusedYeePhase e,
    const __grid_constant__ PairFieldOutputs final,
    const __grid_constant__ BeamzBuffer masks,
    const __grid_constant__ BeamzBuffer bounds) {
  const int mask = static_cast<const int *>(
      masks.data)[blockIdx.y * masks.dims[1] + blockIdx.x];
  if (!mask)
    return;
  const int zb = static_cast<const int *>(bounds.data)[0];
  const int zt = static_cast<const int *>(bounds.data)[1];
  extern __shared__ float shared[];
  for (int wave = 0; wave < zt - zb + 3; ++wave) {
    const int y = threadIdx.x / kMstepWindowX;
    const int x = threadIdx.x % kMstepWindowX;
    const int z = zb + wave - 2;
    const int gy = blockIdx.y * 16 - 2 + y;
    const int gx = blockIdx.x * 32 - 2 + x;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const auto &f = h.inputs[3 + c];
      MstepInput(shared, c, y, x) =
          BufferContains(f, z, gy, gx) ? Read3D(f, z, gy, gx) : 0.f;
    }
    __syncthreads();
    MstepStages<Packed>(shared, h, e, final, blockIdx.x * 32, blockIdx.y * 16,
                        zb, wave, mask);
  }
}

template <int Phase, bool Cpml, bool Packed, bool UniformCpml = false>
__global__ void MstepEdgePhase(const __grid_constant__ BeamzLaunch launch,
                               const __grid_constant__ BeamzBuffer cells,
                               int ny, int nx) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cells.dims[0])
    return;
  const int row = static_cast<const int *>(cells.data)[i];
  const int z = row / (ny * nx), y = (row / nx) % ny, x = row % nx;
  UpdateComponent<Phase, 0, Cpml, 0, true, UniformCpml, -1, Packed>(launch, z,
                                                                    y, x);
  UpdateComponent<Phase, 1, Cpml, 0, true, UniformCpml, -1, Packed>(launch, z,
                                                                    y, x);
  UpdateComponent<Phase, 2, Cpml, 0, true, UniformCpml, -1, Packed>(launch, z,
                                                                    y, x);
}
