"""Opt-in regression gate for the controlled RTX 3090 benchmark protocol."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.hardware
@pytest.mark.performance
def test_cuda_streamed_outperforms_origin_main_on_rtx3090(tmp_path):
    """Run only on the controlled 3090 host; the runner fails on a regression."""
    if os.environ.get("BEAMZ_RUN_RTX3090_BENCHMARK") != "1":
        pytest.skip("set BEAMZ_RUN_RTX3090_BENCHMARK=1 on the controlled RTX 3090")
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        (
            sys.executable,
            "scripts/benchmark_rtx3090_cuda.py",
            "--output-dir",
            str(tmp_path),
        ),
        cwd=root,
        capture_output=True,
        text=True,
        timeout=1_800,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert (tmp_path / "rtx3090-cuda-comparison.json").is_file()
    assert (tmp_path / "rtx3090-cuda-comparison.png").is_file()
