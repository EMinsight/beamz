"""Exercise native gradient writes under Compute Sanitizer without importing JAX.

This Linux harness mirrors launch.h and calls the existing component's exported
C++ launcher. It complements JAX parity tests; it is not a public solver API.
All inputs are initialized host-to-device copies. Gradient allocations are left
uninitialized, then every element is copied back and checked for finite values.
Diagonal field and CPML-memory columns also have independent analytic checks.
"""

import argparse
import ctypes as ct
import json
from itertools import product
from pathlib import Path

import numpy as np


class Buffer(ct.Structure):
    _fields_ = [
        ("data", ct.c_void_p),
        ("rank", ct.c_int32),
        ("element_type", ct.c_int32),
        ("dims", ct.c_int64 * 4),
        ("row_stride", ct.c_int32),
        ("plane_stride", ct.c_int32),
    ]


class Phase(ct.Structure):
    _fields_ = [
        *(
            (name, ct.c_int32)
            for name in ("abi_version", "cuda_flags", "phase", "nterms", "metric_kind")
        ),
        *(
            (name, ct.c_float)
            for name in (
                "dt",
                "resolution",
                "inv_resolution",
                "dt_over_eps",
                "dt_over_mu",
            )
        ),
        ("metallic_edges", ct.c_int32),
        ("uniform_cpml_thickness", ct.c_int32),
        ("inputs", Buffer * 37),
        ("metrics", Buffer * 3),
        ("outputs", Buffer * 9),
        ("shard_geometry", Buffer),
        ("shard_halos", Buffer * 6),
    ]


class Adjoint(ct.Structure):
    _fields_ = [
        ("phase", Phase),
        ("gradient_mask", ct.c_uint64),
        ("gradients", Buffer * 37),
        ("metric_gradients", Buffer * 3),
        ("accumulate_sources", ct.c_bool),
        ("accumulate_material", ct.c_bool),
        ("source_cotangents", Buffer * 3),
        ("source_decays", Buffer * 3),
        ("reverse_iteration", Buffer),
        ("psi_bank_stride", ct.c_int32 * 6),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cudart", required=True, type=Path)
    parser.add_argument(
        "--banked",
        action="store_true",
        help="Exercise aliased source/material gradients and both CPML banks",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    abi = json.loads((root / "cuda/abi_layout.json").read_text())["abi_version"]
    if abi != 27 or ct.sizeof(Phase) != 3184 or ct.sizeof(Adjoint) != 5856:
        raise RuntimeError("Update this harness when launch.h/ABI changes")
    extension = next((root / "beamz").glob("_cuda*.so"))
    native, runtime = ct.CDLL(str(extension)), ct.CDLL(str(args.cudart))
    launch = native._Z18BeamzLaunchAdjointPvRK18BeamzAdjointLaunch
    launch.argtypes, launch.restype = (ct.c_void_p, ct.POINTER(Adjoint)), ct.c_int
    runtime.cudaMalloc.argtypes = (ct.POINTER(ct.c_void_p), ct.c_size_t)
    runtime.cudaMemcpy.argtypes = (ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int)
    runtime.cudaFree.argtypes = (ct.c_void_p,)
    runtime.cudaGetErrorString.argtypes = (ct.c_int,)
    runtime.cudaGetErrorString.restype = ct.c_char_p

    def check(status):
        if status:
            raise RuntimeError(runtime.cudaGetErrorString(status).decode())

    rng = np.random.default_rng(201)
    empty = Buffer(None, 1, 0, (ct.c_int64 * 4)(0, 0, 0, 0))
    e_shapes = ((9, 10, 10), (9, 9, 11), (8, 10, 11))
    h_shapes = ((8, 9, 11), (8, 10, 10), (9, 9, 10))
    cases = 0
    for phase in (0, 1):
        shapes, sources = (h_shapes, e_shapes) if phase == 0 else (e_shapes, h_shapes)
        for cpml in (False, True):
            for walls in (0, 42):
                for layout, mode, bank in product(
                    ("scalar", "profile", "grid"),
                    ("fields", "sources", "all"),
                    (0, 1) if args.banked and cpml else (0,),
                ):
                    allocations = []

                    def allocate(
                        shape, value=None, dtype=np.float32, allocations=allocations
                    ):
                        size = int(np.prod(shape)) if shape else 1
                        pointer = ct.c_void_p()
                        check(
                            runtime.cudaMalloc(
                                ct.byref(pointer), size * np.dtype(dtype).itemsize
                            )
                        )
                        allocations.append(pointer)
                        result = Buffer(
                            pointer,
                            len(shape),
                            int(dtype == np.int32),
                            (ct.c_int64 * 4)(*shape),
                            shape[2] if len(shape) == 3 else 0,
                            shape[1] * shape[2] if len(shape) == 3 else 0,
                        )
                        if value is not None:
                            array = (
                                np.full(shape, value, dtype=dtype)
                                if np.isscalar(value)
                                else np.asarray(value, dtype=dtype)
                            )
                            array = np.ascontiguousarray(array)
                            assert array.size == size
                            check(
                                runtime.cudaMemcpy(
                                    pointer, array.ctypes.data, array.nbytes, 1
                                )
                            )
                        return result

                    work = Adjoint()
                    work.accumulate_sources = args.banked
                    work.accumulate_material = args.banked
                    if args.banked:
                        work.reverse_iteration = allocate((), bank, np.int32)
                        for component, shape in enumerate(sources):
                            work.source_cotangents[component] = allocate(shape, 0.15)
                            work.source_decays[component] = allocate((), 0.93)
                    work.phase.abi_version = abi
                    work.phase.phase = phase
                    work.phase.nterms = 6 if cpml else 0
                    work.phase.resolution = work.phase.inv_resolution = 1.0
                    work.phase.metallic_edges = walls
                    work.phase.uniform_cpml_thickness = 2 if cpml else 0
                    work.phase.dt = 0.1
                    payload = 37 if cpml else 13
                    field_mask = 63 | ((63 << 31) if cpml else 0)
                    work.gradient_mask = (
                        ((1 << payload) - 1) & ~(1 << 12)
                        if mode == "all"
                        else field_mask | ((7 << 9) if mode == "sources" else 0)
                    )
                    cotangents = [
                        rng.normal(size=shape).astype(np.float32) * 0.1
                        for shape in shapes
                    ]
                    for index, shape in enumerate((*shapes, *sources)):
                        work.phase.inputs[index] = allocate(shape, 0.03)
                    for component, shape in enumerate(shapes):
                        coefficient_shape = (
                            ()
                            if layout == "scalar"
                            else ((1, shape[1], 1) if layout == "profile" else shape)
                        )
                        work.phase.inputs[6 + component] = allocate(
                            coefficient_shape, 0.98
                        )
                        work.phase.inputs[9 + component] = allocate(
                            coefficient_shape, 0.2
                        )
                        work.phase.outputs[component] = allocate(
                            shape, cotangents[component]
                        )
                    expected_psi = []
                    masked = []
                    for component, field in enumerate(cotangents):
                        field = field.copy()
                        for axis in range(3):
                            if (phase == 0) != (axis == 2 - component):
                                continue
                            for side in (0, 1):
                                if walls & (1 << (2 * axis + side)):
                                    selection = [slice(None)] * 3
                                    selection[axis] = 0 if side == 0 else -1
                                    field[tuple(selection)] = 0
                        masked.append(field)
                    if cpml:
                        axes = (1, 0, 0, 2, 2, 1)
                        metadata = np.asarray(
                            [
                                (axis, 1 if term % 2 == 0 else -1, 2, 2, 0)
                                for term, axis in enumerate(axes)
                            ],
                            dtype=np.int32,
                        )
                        work.phase.inputs[12] = allocate(
                            metadata.shape, metadata, np.int32
                        )
                        for term, axis in enumerate(axes):
                            profile = [1, 1, 1]
                            profile[axis] = 4
                            for kind, value in enumerate((-0.1, 0.95, 0.8)):
                                work.phase.inputs[13 + 3 * term + kind] = allocate(
                                    tuple(profile), value
                                )
                            shape = list(shapes[term // 2])
                            shape[axis] = 4
                            shape = tuple(shape)
                            psi_cot = rng.normal(size=shape).astype(np.float32) * 0.1
                            work.phase.inputs[31 + term] = allocate(shape, 0.03)
                            if args.banked:
                                backing = allocate((2, *shape))
                                stride = int(np.prod(shape))
                                work.psi_bank_stride[term] = stride
                                logical = Buffer(
                                    backing.data,
                                    3,
                                    0,
                                    (ct.c_int64 * 4)(*shape),
                                    shape[2],
                                    shape[1] * shape[2],
                                )
                                work.phase.outputs[3 + term] = logical
                                check(
                                    runtime.cudaMemcpy(
                                        ct.c_void_p(backing.data + bank * stride * 4),
                                        psi_cot.ctypes.data,
                                        psi_cot.nbytes,
                                        1,
                                    )
                                )
                            else:
                                work.phase.outputs[3 + term] = allocate(shape, psi_cot)
                            field = masked[term // 2]
                            packed_field = np.take(
                                field,
                                [0, 1, field.shape[axis] - 2, field.shape[axis] - 1],
                                axis=axis,
                            )
                            sign = (-1 if phase == 0 else 1) * (-1 if term % 2 else 1)
                            expected_psi.append(
                                np.float32(0.95)
                                * (psi_cot + np.float32(sign * 0.2) * packed_field)
                            )
                    else:
                        work.phase.inputs[12] = allocate(
                            (0, 5), np.empty((0, 5), np.int32), np.int32
                        )
                    for axis in range(3):
                        work.phase.metrics[axis] = allocate((), 1.0)
                        work.metric_gradients[axis] = empty
                    for index in range(payload):
                        if not work.gradient_mask & (1 << index):
                            work.gradients[index] = empty
                            continue
                        value = work.phase.inputs[index]
                        if args.banked and 3 <= index < 6:
                            work.gradients[index] = work.source_cotangents[index - 3]
                        elif args.banked and index >= 31:
                            work.gradients[index] = work.phase.outputs[3 + index - 31]
                        else:
                            work.gradients[index] = allocate(
                                tuple(value.dims[: value.rank]),
                                0.25 if args.banked and 9 <= index < 12 else None,
                            )
                    try:
                        check(launch(None, ct.byref(work)))
                        check(runtime.cudaDeviceSynchronize())
                        for index in range(payload):
                            if not work.gradient_mask & (1 << index):
                                continue
                            value = work.gradients[index]
                            array = np.empty(
                                tuple(value.dims[: value.rank]), dtype=np.float32
                            )
                            check(
                                runtime.cudaMemcpy(
                                    array.ctypes.data,
                                    ct.c_void_p(
                                        value.data
                                        + (
                                            (bank ^ 1)
                                            * work.psi_bank_stride[index - 31]
                                            * 4
                                        )
                                    )
                                    if args.banked and index >= 31
                                    else value.data,
                                    array.nbytes,
                                    2,
                                )
                            )
                            assert np.all(np.isfinite(array)), (
                                phase,
                                cpml,
                                walls,
                                layout,
                                index,
                            )
                            if index < 3:
                                np.testing.assert_allclose(
                                    array,
                                    np.float32(0.98) * masked[index],
                                    rtol=2e-6,
                                    atol=1e-7,
                                )
                            if index >= 31:
                                np.testing.assert_allclose(
                                    array,
                                    expected_psi[index - 31],
                                    rtol=2e-6,
                                    atol=1e-7,
                                )
                            if (
                                args.banked
                                and not cpml
                                and not walls
                                and 9 <= index < 12
                            ):
                                np.testing.assert_allclose(array, 0.25, atol=1e-7)
                        cases += 1
                    finally:
                        for pointer in allocations:
                            check(runtime.cudaFree(pointer))
    print(f"Passed {cases} native gradient-buffer cases (all outputs read).")


if __name__ == "__main__":
    main()
