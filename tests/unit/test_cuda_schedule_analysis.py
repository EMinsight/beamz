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


def test_archive_is_replayable_and_excluded_pilots_cannot_affect_ranking(tmp_path):
    report = _report([_record("streamed", 0.4, 0.5)])
    pilot = _report([_record("mstep", 0.0, 0.001)])
    path = tmp_path / "archive.json"
    path.write_text(
        json.dumps(
            {
                "evidence": [
                    dict(source="fresh.json", exclusion=None, report=report),
                    dict(
                        source="pilot.json", exclusion="Invalid workload", report=pilot
                    ),
                ]
            }
        )
    )
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


def test_evidence_export_excludes_pilots_and_reports_streamed_fallback(tmp_path):
    from scripts.export_cuda_schedule_study import rows, write_tables

    record = _record("mstep-request", 0.4, 0.5)
    record.update(
        decision={"selected": "streamed"},
        cold_time_to_result_s=1.0,
        total_time_to_result_s=4.0,
    )
    report = _report([record])
    packet = {
        "evidence": [
            dict(
                source="finalists256/medium.json",
                phase="fresh",
                exclusion=None,
                report=report,
            ),
            dict(
                source="magnetic.json",
                phase="fresh",
                exclusion="Not dense H",
                report=report,
            ),
        ]
    }
    measured = rows(packet)
    assert len(measured) == 1
    assert measured[0]["actual_schedule"] == "streamed"
    assert measured[0]["raw_pipeline_first_result_s"] == 4.0
    write_tables(measured, tmp_path)
    assert "fallback" in (tmp_path / "cuda-schedule-tables.md").read_text()


def _export_report(records):
    for record in records:
        record.update(
            decision={"selected": "streamed"},
            cold_time_to_result_s=record["cold_preparation_s"] + record["first_run_s"],
            total_time_to_result_s=8.0,
        )
    return _report(records)


def _packet(*reports):
    return {
        "evidence": [
            dict(
                source=f"finalists256/{i}.json", phase="fresh", exclusion=None, report=r
            )
            for i, r in enumerate(reports)
        ]
    }


@pytest.mark.parametrize(
    "provenance",
    [
        {"extension_sha256": "different binary"},
        {"implementation_sha256": "different dispatch"},
        {"benchmark_sha256": "different method"},
        {"hardware": {"uuid": "different GPU", "driver": "driver", "power_limit": 300}},
    ],
)
def test_export_and_ranking_separate_resumed_profiles(tmp_path, provenance):
    from scripts.export_cuda_schedule_study import profiles, rows

    records = [_record("streamed", 0.4, 0.5), _record("streamed", 4.0, 5.0)]
    records[1]["provenance"] = provenance
    report = _export_report(records)
    packet = _packet(report)
    measured = rows(packet)
    assert len(measured) == 2
    assert all(r["rounds"] == 1 for r in measured)
    assert {r["preparation_s"] for r in measured} == {0.4, 4.0}
    assert {r["warm_s"] for r in measured} == {0.5, 5.0}
    assert {r["profile"] for r in measured} == set(profiles(packet))
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    assert len(load_reports([path], fresh_only=True)) == 2


def test_table_never_compares_choices_from_different_profiles(tmp_path):
    from scripts.export_cuda_schedule_study import rows, write_tables

    streamed = _export_report([_record("streamed", 0.4, 0.5)])
    mstep = _export_report([_record("mstep-request", 0.1, 0.1)])
    mstep["extension_sha256"] = "different binary"
    mstep["records"][0]["decision"]["selected"] = "mstep"
    measured = rows(_packet(streamed, mstep))
    write_tables(measured, tmp_path)
    table = (tmp_path / "cuda-schedule-tables.md").read_text()
    assert table.count("| medium (") == 2
    assert "streamed (only candidate)" in table
    assert "mstep-request (only candidate)" in table


def test_export_uses_only_fresh_workers_and_profile_spans_horizons():
    from scripts.export_cuda_schedule_study import rows

    report = _export_report(
        [
            _record("streamed", 0.4, 0.5),
            _record("streamed", 0.0, 0.001, fresh=False),
        ]
    )
    later = _export_report([_record("streamed", 0.4, 2.0)])
    later["case"]["steps"] = 1024
    measured = rows(_packet(report, later))
    assert len(measured) == 2
    assert len({r["profile"] for r in measured}) == 1
    assert measured[0]["warm_s"] == 0.5
    assert measured[0]["rounds"] == 1


def test_plot_does_not_connect_different_profile_horizons(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    from scripts.export_cuda_schedule_study import rows, write_plot

    first = _export_report([_record("streamed", 0.4, 0.5)])
    second = _export_report([_record("streamed", 4.0, 5.0)])
    second["case"]["steps"] = 1024
    second["extension_sha256"] = "different binary"
    measurements = rows(_packet(first, second))
    original = Axes.plot
    horizons = []

    def plot(self, x, *args, **kwargs):
        horizons.append(tuple(x))
        return original(self, x, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", plot)
    write_plot(measurements, tmp_path)
    assert sorted(horizons) == [(256,), (1024,)]
    assert (tmp_path / "cuda-schedule-time-to-result.png").exists()
