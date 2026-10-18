#!/usr/bin/env bash
# Run from the repository root after installing the test dependencies.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
.venv/bin/python -m pytest \
  tests/unit/modes \
  tests/unit/test_mode_source_solver.py \
  tests/unit/test_modal_normal_sampling.py \
  tests/unit/test_modal_power_reference.py \
  tests/unit/test_material_blocks.py \
  tests/unit/test_cuda_coefficient_storage.py \
  tests/unit/test_2d_port_orientation.py \
  tests/contracts/test_public_input_validation.py \
  tests/contracts/test_mode_plane_contract.py \
  tests/contracts/test_modal_material_snapshot.py \
  tests/contracts/test_tensor_modal_support.py \
  tests/contracts/test_result_archive.py \
  tests/integration/test_mode_launch_planner.py \
  tests/integration/test_modal_power_contract.py \
  tests/integration/test_differentiable_mode_projection.py \
  tests/integration/test_native_design_rasterization.py \
  tests/integration/test_mode_source_integration.py \
  tests/integration/test_mode_source_monitor_ygrid.py \
  tests/validation/regression/test_reverse_mode_launch.py \
  tests/validation/regression/test_tensor_port_workflow.py \
  tests/validation/analytical/test_guided_cpml_projection.py \
  tests/validation/analytical/test_tensor_constitutive_operator.py \
  tests/validation/analytical/test_tensor_port_maxwell.py \
  tests/validation/convergence/test_tensor_port_convergence.py \
  tests/validation/analytical/test_rectilinear_tensor_operator.py \
  tests/validation/analytical/test_tensor_region_update.py \
  tests/validation/analytical/test_rectilinear_tensor_modes.py \
  tests/validation/regression/test_rectilinear_tensor_workflow.py \
  tests/unit/test_rectilinear_grid.py \
  tests/validation/invariants/test_automatic_termination.py \
  tests/differential/test_tensor_port_mpb.py -q
