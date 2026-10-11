"""Measured startup costs and reuse can reverse the preferred CUDA schedule."""

import gzip
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


def test_repeated_common_program_setup_cannot_bias_schedule_comparison():
    streamed = _record("streamed", 0.8, 0.5)
    mstep = _record("mstep", 1.0, 0.3)
    mstep["program_setup_s"] = 0.4
    summary = summarize(_report([streamed, mstep]))
    assert rank(summary, 256)[0]["configuration"]["name"] == "mstep"
    measured = next(r for r in summary["configurations"] if r["name"] == "mstep")
    assert measured["cold_preparation_s"] == pytest.approx(0.6)


def test_shared_sweep_cannot_supply_verified_cold_start(tmp_path):
    (tmp_path / "shared.json").write_text(
        json.dumps(_report([_record("mstep", 0.01, 0.3, fresh=False)]))
    )
    assert load_reports([tmp_path], fresh_only=True) == []


def test_resumed_sweep_keeps_each_workers_implementation_identity(tmp_path):
    first = _record("streamed", 0.4, 0.5)
    second = _record("mstep", 0.9, 0.3)
    first["provenance"] = {"extension_sha256": "earlier-binary"}
    report = _report([first, second])
    (tmp_path / "resumed.json").write_text(json.dumps(report))
    reports = load_reports([tmp_path], fresh_only=True)
    assert {r["extension_sha256"] for r in reports} == {
        "earlier-binary",
        "binary-one",
    }
    assert all(len(r["records"]) == 1 for r in reports)


@pytest.mark.parametrize("compressed", [False, True])
def test_archive_is_replayable_and_excluded_pilots_cannot_affect_ranking(
    tmp_path, compressed
):
    report = _report([_record("streamed", 0.4, 0.5)])
    pilot = _report([_record("mstep", 0.0, 0.001)])
    path = tmp_path / ("archive.json.gz" if compressed else "archive.json")
    payload = json.dumps(
        {
            "evidence": [
                dict(source="fresh.json", exclusion=None, report=report),
                dict(source="pilot.json", exclusion="Invalid workload", report=pilot),
            ]
        }
    ).encode()
    path.write_bytes(gzip.compress(payload, mtime=0) if compressed else payload)
    reports = load_reports([path], fresh_only=True)
    assert len(reports) == 1
    assert rank(summarize(reports[0]), 256)[0]["configuration"]["name"] == "streamed"


def test_exhaustive_layout_space_is_complete_and_names_are_unique():
    from scripts.benchmark_cuda_schedule_space import configurations

    candidates = configurations(exhaustive_layouts=True)
    assert len(candidates) == len({c["name"] for c in candidates}) == 72
    mstep = [c for c in candidates if c["backend"] == "cuda_mstep"]
    assert len(mstep) == 4 * 5 * 3
    assert {c["name"] for c in configurations()} <= {c["name"] for c in candidates}


@pytest.mark.parametrize(
    "provenance",
    [
        {"extension_sha256": "different binary"},
        {"implementation_sha256": "different dispatch"},
        {"benchmark_sha256": "different method"},
        {"hardware": {"uuid": "different GPU", "driver": "driver", "power_limit": 300}},
    ],
)
def test_ranking_separates_resumed_profiles(tmp_path, provenance):
    records = [_record("streamed", 0.4, 0.5), _record("streamed", 4.0, 5.0)]
    records[1]["provenance"] = provenance
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report(records)))
    reports = load_reports([path], fresh_only=True)
    rows = [summarize(report)["configurations"][0] for report in reports]
    assert len(rows) == 2
    assert all(row["rounds"] == 1 for row in rows)
    assert {row["cold_preparation_s"] for row in rows} == {0.4, 4.0}
    assert {row["execution_s"] for row in rows} == {0.5, 5.0}


def test_requested_mstep_fallback_is_reported_as_streamed():
    record = _record("mstep-request", 0.4, 0.5)
    record["decision"] = {"selected": "streamed"}
    summary = summarize(_report([record]))
    assert summary["configurations"][0]["actual_schedule"] == "streamed"
