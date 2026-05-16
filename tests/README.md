# Test Structure

The suite currently mixes several kinds of tests. New tests should be placed and
marked by contract, not just by subsystem.

## Test categories

- `unit`: Fast tests for public API behavior and pure helper logic.
- `component`: Small real-component tests with limited simulation scope.
- `integration`: End-to-end tests that span multiple subsystems.
- `characterization`: Numerical sweeps and behavior-profiling tests. These are
  useful for research and regression analysis, but they should not be the
  default fast gate.
- `compiled`: Tests that exercise the compiled/JAX execution path.
- `pdk`: Tests that depend on an external PDK or design-kit install.
- `slow`: Tests that are too expensive for the default fast feedback loop.

## Running tests

The pull-request CI gate intentionally runs a marked fast slice:

```bash
pytest tests/ -m "(unit or component or optimization) and not slow and not characterization and not pdk"
```

The full suite, including coverage, is still the release-quality signal and runs
on the weekly/manual GitHub workflow:

```bash
pytest tests/ --cov=beamz --cov-report=term-missing --cov-report=xml
```

## Placement guidance

- Keep API normalization and validation tests in focused files such as
  `test_simulation_api.py`.
- Keep small compiled-engine and monitor tests in compiled/component files.
- Keep expensive physics validation in integration-oriented files.
- Keep broad numerical sweeps and exploratory residual checks in
  characterization files, not mixed into core API tests.

## Rules of thumb

- Prefer public-behavior assertions over private attribute checks.
- Avoid asserting internal helper names or exact implementation wording in
  exception messages.
- Do not freeze known defects as expected behavior.
- Mark heavy tests explicitly so fast PR jobs can exclude them.
