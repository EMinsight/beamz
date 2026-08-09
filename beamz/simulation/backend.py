"""Execution-backend selection with a lazy optional CUDA extension boundary."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

import jax

ExecutionBackend = Literal[
    "auto",
    "jax",
    "cuda",
    "cuda_streamed",
    "cuda_hopper",
]
ResolvedBackend = Literal["jax", "cuda_streamed", "cuda_hopper"]

_EXTENSION_MODULE = "beamz_cuda"
_REGISTERED_MODULE: ModuleType | None = None
CUDA_ABI_VERSION = 10

# One immutable bitset travels from program compilation through XLA FFI into every
# native launch. Environment variables remain useful for controlled experiments,
# but they are sampled exactly once and therefore cannot change a cached program's
# numerical or scheduling meaning underneath JAX.
CUDA_TYPED_PSI = 1 << 0
CUDA_BATCHED_SOURCE_GROUPS = 1 << 1
CUDA_COINCIDENT_SOURCE_GROUPS = 1 << 2
CUDA_ADAPTIVE_SOURCE_TILES = 1 << 3
CUDA_CPML_CORE_SPLIT = 1 << 4
CUDA_COMBINED_CPML_QUEUE = 1 << 5
CUDA_GRAPH_CACHE = 1 << 7
CUDA_TEMPORAL_PSI = 1 << 8
CUDA_TEMPORAL_CPML = 1 << 9
CUDA_TEMPORAL_YEE = 1 << 10
CUDA_MATERIAL_CODEBOOK = 1 << 11
CUDA_BF16_PSI = 1 << 12

CUDA_DEFAULT_FLAGS = (
    CUDA_TYPED_PSI
    | CUDA_BATCHED_SOURCE_GROUPS
    | CUDA_COINCIDENT_SOURCE_GROUPS
    | CUDA_ADAPTIVE_SOURCE_TILES
    | CUDA_CPML_CORE_SPLIT
    | CUDA_COMBINED_CPML_QUEUE
    | CUDA_GRAPH_CACHE
    | CUDA_TEMPORAL_PSI
    | CUDA_TEMPORAL_CPML
    | CUDA_TEMPORAL_YEE
    | CUDA_MATERIAL_CODEBOOK
)
CUDA_STREAMED_TARGETS = frozenset(
    {
        "beamz_cuda_streamed",
        "beamz_cuda_streamed_steps",
        "beamz_cuda_temporal_steps",
        "beamz_cuda_streamed_cpml_steps",
        "beamz_cuda_streamed_source_groups_cpml_steps",
        "beamz_cuda_temporal_source_groups_cpml_steps",
        "beamz_cuda_temporal_program_cpml_steps",
        "beamz_cuda_streamed_program_cpml_steps",
    }
)
CUDA_HOPPER_TARGET = "beamz_cuda_hopper"


class CudaBackendUnavailable(RuntimeError):
    """An explicitly requested CUDA backend cannot run in this process."""


def _env_enabled(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def cuda_flags_from_env() -> int:
    """Snapshot experimental CUDA policy into one cacheable native bitset."""
    flags = CUDA_DEFAULT_FLAGS
    disable_flags = {
        "BEAMZ_CUDA_DISABLE_TYPED_PSI": CUDA_TYPED_PSI,
        "BEAMZ_CUDA_DISABLE_BATCHED_SOURCE_GROUPS": CUDA_BATCHED_SOURCE_GROUPS,
        "BEAMZ_CUDA_DISABLE_COINCIDENT_SOURCE_GROUPS": CUDA_COINCIDENT_SOURCE_GROUPS,
        "BEAMZ_CUDA_DISABLE_ADAPTIVE_SOURCE_TILES": CUDA_ADAPTIVE_SOURCE_TILES,
        "BEAMZ_CUDA_DISABLE_CPML_CORE_SPLIT": CUDA_CPML_CORE_SPLIT,
        "BEAMZ_CUDA_DISABLE_COMBINED_CPML_QUEUE": CUDA_COMBINED_CPML_QUEUE,
        "BEAMZ_CUDA_DISABLE_GRAPH_CACHE": CUDA_GRAPH_CACHE,
        "BEAMZ_CUDA_DISABLE_TEMPORAL_PSI": CUDA_TEMPORAL_PSI,
        "BEAMZ_CUDA_DISABLE_CPML_TEMPORAL": CUDA_TEMPORAL_CPML,
        "BEAMZ_CUDA_DISABLE_TEMPORAL": CUDA_TEMPORAL_YEE,
        "BEAMZ_CUDA_DISABLE_MATERIAL_CODEBOOK": CUDA_MATERIAL_CODEBOOK,
    }
    for name, flag in disable_flags.items():
        if _env_enabled(name, default=False):
            flags &= ~flag
    precision = os.environ.get("BEAMZ_CUDA_CPML_PSI_PRECISION", "fp32").lower()
    if precision in {"bf16", "bfloat16"}:
        flags |= CUDA_BF16_PSI
    elif precision not in {"fp32", "float32", ""}:
        raise ValueError("BEAMZ_CUDA_CPML_PSI_PRECISION must be 'fp32' or 'bf16'")
    return flags


@dataclass(frozen=True, slots=True)
class CudaBackendStatus:
    """Diagnostic snapshot used by backend selection and bug reports."""

    available: bool
    extension_version: str | None
    abi_version: int | None
    targets: tuple[str, ...]
    gpu_devices: tuple[str, ...]
    compute_capabilities: tuple[int, ...]
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return stable diagnostics suitable for logs and bug reports."""
        return {
            "available": self.available,
            "extension_version": self.extension_version,
            "abi_version": self.abi_version,
            "targets": self.targets,
            "gpu_devices": self.gpu_devices,
            "compute_capabilities": self.compute_capabilities,
            "reason": self.reason,
        }


def normalize_backend(backend: str | None) -> ExecutionBackend:
    value = (
        os.environ.get("BEAMZ_EXECUTION_BACKEND", "auto")
        if backend is None
        else str(backend)
    )
    aliases = {
        "auto": "auto",
        "jax": "jax",
        "xla": "jax",
        "cuda": "cuda",
        "streamed": "cuda_streamed",
        "cuda_streamed": "cuda_streamed",
        "cuda-streamed": "cuda_streamed",
        "hopper": "cuda_hopper",
        "cuda_hopper": "cuda_hopper",
        "cuda-hopper": "cuda_hopper",
    }
    try:
        return aliases[value.strip().lower()]  # type: ignore[return-value]
    except KeyError as exc:
        choices = "auto, jax, cuda, cuda_streamed, cuda_hopper"
        raise ValueError(
            f"Unknown execution backend {value!r}; use one of: {choices}."
        ) from exc


def _load_extension() -> ModuleType:
    return importlib.import_module(_EXTENSION_MODULE)


def _gpu_devices():
    try:
        return tuple(jax.devices("gpu"))
    except Exception:
        return ()


def _compute_capability(device) -> int:
    capability = getattr(device, "compute_capability", None)
    if capability is None:
        stats = getattr(device, "device_stats", lambda: {})() or {}
        capability = stats.get("compute_capability")
    if isinstance(capability, tuple):
        return int(capability[0]) * 10 + int(capability[1])
    if isinstance(capability, str):
        parts = capability.replace("sm_", "").split(".")
        return int(parts[0]) * 10 + int(parts[1]) if len(parts) == 2 else int(parts[0])
    if capability is None:
        # The model name is enough for safe Hopper dispatch when older jaxlib builds
        # do not expose compute capability directly.
        return 90 if "H100" in str(getattr(device, "device_kind", "")).upper() else 0
    return int(capability)


def _validated_registrations(extension: ModuleType) -> dict[str, object]:
    abi_version = getattr(extension, "__abi_version__", None)
    if abi_version != CUDA_ABI_VERSION:
        raise CudaBackendUnavailable(
            "beamz_cuda ABI mismatch: "
            f"runtime requires v{CUDA_ABI_VERSION}, extension provides "
            f"{abi_version!r}"
        )
    registrations = extension.registrations()
    if not isinstance(registrations, dict) or not registrations:
        raise CudaBackendUnavailable("beamz_cuda exposes no FFI registrations")
    targets = {str(name) for name in registrations}
    missing = sorted(CUDA_STREAMED_TARGETS - targets)
    if missing:
        raise CudaBackendUnavailable(
            "beamz_cuda is missing required streamed FFI targets: " + ", ".join(missing)
        )
    return registrations


def register_cuda_ffi_targets(module: ModuleType | None = None) -> tuple[str, ...]:
    """Register extension-provided typed FFI handlers exactly once."""
    global _REGISTERED_MODULE
    extension = _load_extension() if module is None else module
    if _REGISTERED_MODULE is extension:
        return tuple(sorted(_validated_registrations(extension)))
    registrations = _validated_registrations(extension)
    for name, capsule in registrations.items():
        jax.ffi.register_ffi_target(
            str(name),
            capsule,
            platform="CUDA",
            api_version=1,
        )
    _REGISTERED_MODULE = extension
    return tuple(sorted(str(name) for name in registrations))


def cuda_backend_status(*, register: bool = True) -> CudaBackendStatus:
    devices = _gpu_devices()
    device_names = tuple(
        str(getattr(device, "device_kind", device)) for device in devices
    )
    capabilities = tuple(_compute_capability(device) for device in devices)
    if not devices:
        return CudaBackendStatus(
            False,
            None,
            None,
            (),
            device_names,
            capabilities,
            "JAX has no visible CUDA devices",
        )
    try:
        extension = _load_extension()
        registrations = _validated_registrations(extension)
        targets = (
            register_cuda_ffi_targets(extension)
            if register
            else tuple(sorted(registrations))
        )
    except (ImportError, AttributeError, CudaBackendUnavailable) as exc:
        return CudaBackendStatus(
            False,
            None,
            None,
            (),
            device_names,
            capabilities,
            f"optional beamz_cuda extension is unavailable: {exc}",
        )
    version = str(getattr(extension, "__version__", "unknown"))
    return CudaBackendStatus(
        True,
        version,
        CUDA_ABI_VERSION,
        targets,
        device_names,
        capabilities,
    )


def resolve_backend(backend: str | None) -> ResolvedBackend:
    """Resolve public backend policy without silently weakening explicit requests."""
    requested = normalize_backend(backend)
    if requested == "jax":
        return "jax"
    status = cuda_backend_status()
    if requested == "auto" and not status.available:
        return "jax"
    if not status.available:
        raise CudaBackendUnavailable(
            f"CUDA execution backend was requested but is unavailable: {status.reason}. "
            "Install the beamz-cuda wheel matching JAX/CUDA, or use backend='jax'."
        )
    has_streamed = CUDA_STREAMED_TARGETS.issubset(status.targets)
    has_hopper = (
        CUDA_HOPPER_TARGET in status.targets
        and bool(status.compute_capabilities)
        and all(capability >= 90 for capability in status.compute_capabilities)
    )
    if requested == "cuda_hopper":
        if not has_hopper:
            raise CudaBackendUnavailable(
                "cuda_hopper requires the beamz_cuda_hopper target and SM90+ GPUs"
            )
        return "cuda_hopper"
    if requested == "cuda_streamed":
        if not has_streamed:
            raise CudaBackendUnavailable(
                "cuda_streamed target is missing from the beamz_cuda extension"
            )
        return "cuda_streamed"
    if has_streamed:
        return "cuda_streamed"
    if requested == "auto":
        return "jax"
    raise CudaBackendUnavailable("beamz_cuda has no compatible execution target")
