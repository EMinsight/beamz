from __future__ import annotations

import numpy as np

from beamz.simulation.compile import (
    _pack_cuda_coefficient_ids,
    _pack_cuda_lossless_e_coefficients,
)


def _unpack(table, packed, size):
    words = np.asarray(packed).view(np.uint32)
    shifts = np.arange(4, dtype=np.uint32) * np.uint32(8)
    codes = ((words[:, None] >> shifts) & np.uint32(0xFF)).reshape(-1)[:size]
    return np.asarray(table)[codes]


def test_cuda_coefficient_ids_round_trip_exact_fp32_values():
    values = np.asarray([[[1.0, 2.0, 4.0], [2.0, 1.0, 8.0]]], dtype=np.float32)

    encoded = _pack_cuda_coefficient_ids(values)

    assert encoded is not None
    table, packed = encoded
    np.testing.assert_array_equal(_unpack(table, packed, values.size), values.ravel())
    assert packed.dtype == np.int32
    assert packed.size == 2


def test_cuda_lossless_codebook_requires_scalar_unit_decay(monkeypatch):
    monkeypatch.delenv("BEAMZ_CUDA_DISABLE_MATERIAL_CODEBOOK", raising=False)
    sources = tuple(np.asarray([[[1.0, 2.0, 1.0]]], dtype=np.float32) for _ in range(3))

    packed = _pack_cuda_lossless_e_coefficients(
        tuple(np.asarray(1.0, dtype=np.float32) for _ in range(3)), sources
    )

    assert packed is not None
    tables, ids = packed
    for table, words, source in zip(tables, ids, sources, strict=True):
        np.testing.assert_array_equal(
            _unpack(table, words, source.size), source.ravel()
        )

    assert (
        _pack_cuda_lossless_e_coefficients(
            (np.asarray(0.9, dtype=np.float32), *(np.asarray(1.0) for _ in range(2))),
            sources,
        )
        is None
    )


def test_cuda_material_codebook_debug_disable(monkeypatch):
    monkeypatch.setenv("BEAMZ_CUDA_DISABLE_MATERIAL_CODEBOOK", "1")
    sources = tuple(np.ones((1, 1, 1), dtype=np.float32) for _ in range(3))

    assert (
        _pack_cuda_lossless_e_coefficients(
            tuple(np.asarray(1.0, dtype=np.float32) for _ in range(3)), sources
        )
        is None
    )
