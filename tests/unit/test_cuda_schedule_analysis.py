"""Measured startup costs and reuse can reverse the preferred CUDA schedule."""

import json
from pathlib import Path

import pytest

from scripts.analyze_cuda_schedule_space import load_reports, rank, summarize


def _record(name, preparation, run, *, fresh=True):
    return dict(
        status="ok",
        configuration=dict(name=name, backend="cuda", env={}),
        samples_s=[run * 0.99, run, run * 1.01],
        median_run_s=run,
        cold_preparation_s=preparation,
        first_run_s=run + 0.1,
        fresh_process=fresh,
    )


def _report(records):
    return dict(
        case=dict(name="medium", shape=[128, 256, 512], steps=256),
        device="GPU A",
        jax="0.test",
        extension_sha256="binary-one",
        records=records,
    )


def test_short_cold_and_repeated_runs_choose_different_schedules():
    summary = summarize(
        _report([_record("streamed", 0.4, 0.5), _record("mstep", 0.9, 0.3)])
    )
    assert rank(summary, 256)[0]["configuration"]["name"] == "streamed"
    assert rank(summary, 256, expected_runs=4)[0]["configuration"]["name"] == "mstep"
    assert rank(summary, 256, cached=True)[0]["configuration"]["name"] == "mstep"
    assert rank(summary, 256, cached=True)[0]["seconds"] == pytest.approx(0.3)
    assert rank(summary, 1024)[0]["extrapolated"]
    assert not rank(summary, 256)[0]["extrapolated"]


def test_loading_merges_complementary_sweeps_but_separates_binaries(tmp_path):
    first = _report([_record("streamed", 0.4, 0.5)])
    second = _report([_record("mstep", 0.9, 0.3)])
    other = dict(second, extension_sha256="binary-two")
    for i, report in enumerate((first, second, other)):
        (tmp_path / f"{i}.json").write_text(json.dumps(report))
    reports = load_reports([Path(tmp_path)])
    assert sorted(len(r["records"]) for r in reports) == [1, 2]
    assert sorted(len(r["sources"]) for r in reports) == [1, 2]


def test_shared_sweep_cannot_supply_verified_cold_start(tmp_path):
    (tmp_path / "shared.json").write_text(
        json.dumps(_report([_record("mstep", 0.01, 0.3, fresh=False)]))
    )
    assert load_reports([tmp_path], fresh_only=True) == []


def test_exhaustive_layout_space_is_complete_and_names_are_unique():
    from scripts.benchmark_cuda_schedule_space import configurations

    candidates = configurations(exhaustive_layouts=True)
    assert len(candidates) == len({c["name"] for c in candidates}) == 72
    mstep = [c for c in candidates if c["backend"] == "cuda_mstep"]
    assert len(mstep) == 4 * 5 * 3
    assert {c["name"] for c in configurations()} <= {c["name"] for c in candidates}
