from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_badge_module():
    script = Path(__file__).parents[2] / "scripts" / "update_coverage_badge.py"
    spec = importlib.util.spec_from_file_location("update_coverage_badge", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_update_badge_writes_only_when_coverage_changes(tmp_path: Path) -> None:
    badge = _load_badge_module()
    coverage = tmp_path / "coverage.xml"
    output = tmp_path / "coverage.svg"
    coverage.write_text('<coverage line-rate="0.843"/>', encoding="utf-8")

    assert badge.update_badge(coverage, output)
    assert 'aria-label="coverage: 84%"' in output.read_text(encoding="utf-8")
    assert not badge.update_badge(coverage, output)
