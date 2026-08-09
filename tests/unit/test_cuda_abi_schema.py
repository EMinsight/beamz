"""Keep both sides of the optional CUDA FFI on one generated contract."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from beamz.simulation import _cuda_abi as abi

ROOT = Path(__file__).resolve().parents[2]


def test_generated_cuda_abi_files_are_current():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_cuda_abi.py"), "--check"],
        cwd=ROOT,
        check=True,
    )


def test_cuda_abi_layout_relationships_are_explicit():
    schema = json.loads((ROOT / "cuda" / "abi_layout.json").read_text())
    layout = schema["layout"]

    assert layout["cpml_e_input_offset"] == (
        layout["field_count"] + layout["cpml_phase_input_count"]
    )
    assert layout["cpml_e_psi_input_offset"] == (
        layout["cpml_h_psi_input_offset"] + layout["cpml_phase_input_count"]
    )
    assert layout["temporal_field_workspace_input"] == layout["cpml_graph_input_count"]
    assert layout["temporal_psi_workspace_input"] == (
        layout["temporal_field_workspace_input"] + layout["field_count"]
    )
    assert layout["cpml_e_psi_input_offset"] == abi.CPML_E_PSI_INPUT_OFFSET


def test_private_cuda_component_version_matches_abi_schema():
    schema = json.loads((ROOT / "cuda" / "abi_layout.json").read_text())
    wheel_config = (ROOT / "cuda" / "pyproject.toml").read_text()
    cmake_config = (ROOT / "cuda" / "CMakeLists.txt").read_text()
    wheel_version = re.search(r'^version = "([^"]+)"$', wheel_config, re.MULTILINE)

    assert wheel_version is not None
    assert wheel_version.group(1) == schema["component_version"]
    assert 'name = "beamz-cuda-component"' in wheel_config
    assert "Private :: Do Not Upload" in wheel_config
    assert '"beamz==' not in wheel_config
    assert re.search(r"nanobind_add_module\(\s*_cuda\b", cmake_config)
    assert "install(TARGETS _cuda LIBRARY DESTINATION beamz)" in cmake_config
    assert schema["abi_version"] == abi.CUDA_ABI_VERSION
