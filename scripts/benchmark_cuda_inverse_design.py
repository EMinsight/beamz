"""Reproduce GPU timings using this checkout's inverse-design notebook setup.

Run from the repository root with the same Python environment as the notebook.
Each backend uses float32, identical source/CPML/DFT physics, the same density
map and fixed mode projector, and the same checkpoint interval. Setup and mode
solving are excluded. Timed device-resident calls are explicitly synchronized.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuda_benchmark_utils import provenance, speedup_interval

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["BEAMZ_NOTEBOOK_BACKEND"] = "cuda"
os.environ["BEAMZ_DOCS_TEST"] = "0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dx-nm", type=int, default=100)
parser.add_argument("--samples", type=int, default=7)
parser.add_argument("--checkpoint", type=int, default=32)
parser.add_argument("--beta", type=float, default=2.0)
parser.add_argument("--parameters-npz", type=Path)
parser.add_argument("--profile", type=Path)
parser.add_argument("--finite-difference", action="store_true")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.samples < 3 or args.dx_nm <= 0:
    parser.error("Use at least three samples and a positive mesh spacing")
notebook_path = Path("examples/notebooks/cuda_autodiff_waveguide_bend.ipynb")
notebook = json.loads(notebook_path.read_text())
namespace = {}


def cell(index):
    return "".join(notebook["cells"][index]["source"])


exec(cell(2), namespace)
jax, jnp, np = (namespace[k] for k in ("jax", "jnp", "np"))
assert jax.default_backend() == "gpu", "Both backends must run on the same GPU"
jax.config.update("jax_enable_compilation_cache", False)
namespace["dx"] = args.dx_nm * 1e-9
namespace["dt"] = 0.8 * namespace["dx"] / (namespace["bz"].LIGHT_SPEED * np.sqrt(3))
namespace["times"] = np.arange(0, namespace["run_time"], namespace["dt"])
namespace["checkpoint_interval"] = args.checkpoint
exec(cell(4), namespace)
exec(cell(6), namespace)
# Ordinary reference/mode setup ends before any differentiable forward call.
exec(cell(8).split("initial_state = solver()", 1)[0], namespace)
print(
    f"Benchmark mesh: {args.dx_nm}nm, {namespace['shape']}, "
    f"{len(namespace['times'])} steps, checkpoint={args.checkpoint}",
    flush=True,
)
parameters = namespace["initial_parameters"]
if args.parameters_npz:
    parameters = jnp.asarray(np.load(args.parameters_npz)["final_parameters"])
    assert parameters.shape == namespace["initial_parameters"].shape
beta = jnp.float32(args.beta)
permittivity = namespace["permittivity"]
project = namespace["project_output"]
denominator = namespace["reference_power"]
solvers = {
    "cuda": namespace["solver"],
    "jax": namespace["simulation"].differentiable(
        backend="jax", checkpoint_interval=args.checkpoint
    ),
}
functions, compiled, report = {}, {}, {}
for backend, solver in solvers.items():

    def objective(p, beta, solver=solver):
        state = solver(permittivity(p, beta))
        amplitude = project(solver.monitor_fields(state, "output"))[0, 1, 0]
        return -(jnp.abs(amplitude) ** 2) / denominator

    functions[backend] = {
        "objective": jax.jit(objective),
        "value_and_grad": jax.jit(jax.value_and_grad(objective)),
    }
    compiled[backend], report[backend] = {}, {}
    for kind, function in functions[backend].items():
        start = time.perf_counter()
        executable = function.lower(parameters, beta).compile()
        compilation = time.perf_counter() - start
        compiled[backend][kind] = executable
        if args.profile:
            path = args.profile / backend / kind
            path.mkdir(parents=True, exist_ok=True)
            (path / "compiled_hlo.txt").write_text(executable.as_text())
        info = executable.memory_analysis()
        report[backend][kind] = {
            "lower_and_compile_s": compilation,
            "temporary_bytes": info.temp_size_in_bytes,
            "samples_s": [],
        }
        # Warm up separately. This also forces initial device allocations.
        for _ in range(3):
            jax.block_until_ready(executable(parameters, beta))
        print(f"Compiled/warmed {backend} {kind}: {compilation:.3f} s", flush=True)
# Confirm correctness outside the timed intervals.
native = compiled["cuda"]["value_and_grad"](parameters, beta)
reference = compiled["jax"]["value_and_grad"](parameters, beta)
jax.block_until_ready((native, reference))
value_error = float(abs(native[0] - reference[0]))
gradient_error = float(
    jnp.linalg.norm(native[1] - reference[1]) / jnp.linalg.norm(reference[1])
)
objective_value = float(native[0])
gradient_norm = float(jnp.linalg.norm(native[1]))
finite_differences = []
if args.finite_difference:
    direction = jnp.asarray(
        np.random.default_rng(893).normal(size=parameters.shape), dtype=jnp.float32
    )
    direction /= jnp.linalg.norm(direction)
    predicted = float(jnp.vdot(native[1], direction))
    for delta in (0.2, 0.1, 0.05):
        high = compiled["cuda"]["objective"](parameters + delta * direction, beta)
        low = compiled["cuda"]["objective"](parameters - delta * direction, beta)
        observed = float((high - low) / (2 * delta))
        finite_differences.append(
            dict(delta=delta, analytic=predicted, observed=observed)
        )
    assert any(
        np.isclose(item["observed"], predicted, rtol=0.03, atol=1e-6)
        for item in finite_differences
    ), finite_differences
np.testing.assert_allclose(native[0], reference[0], rtol=5e-4, atol=1e-6)
assert gradient_error < 0.005
print(f"GPU gradient parity: {gradient_error:.4%}", flush=True)
del native, reference
for kind in ("objective", "value_and_grad"):
    for sample in range(args.samples):
        order = ("cuda", "jax") if sample % 2 == 0 else ("jax", "cuda")
        for backend in order:
            start = time.perf_counter()
            result = compiled[backend][kind](parameters, beta)
            jax.block_until_ready(result)
            elapsed = time.perf_counter() - start
            report[backend][kind]["samples_s"].append(elapsed)
            del result
        print(
            f"Sample {sample + 1}/{args.samples} {kind}: "
            f"CUDA={report['cuda'][kind]['samples_s'][-1]:.4f}s "
            f"JAX={report['jax'][kind]['samples_s'][-1]:.4f}s",
            flush=True,
        )
for backend in report:
    for kind in report[backend]:
        samples = report[backend][kind]["samples_s"]
        report[backend][kind].update(
            median_s=float(np.median(samples)),
            minimum_s=float(np.min(samples)),
            maximum_s=float(np.max(samples)),
        )
if args.profile:
    for backend in ("cuda", "jax"):
        for kind in ("objective", "value_and_grad"):
            with jax.profiler.trace(
                str(args.profile / backend / kind), create_perfetto_trace=True
            ):
                jax.block_until_ready(compiled[backend][kind](parameters, beta))
speedups = {
    kind: report["jax"][kind]["median_s"] / report["cuda"][kind]["median_s"]
    for kind in ("objective", "value_and_grad")
}
metadata = dict(
    jax_version=jax.__version__,
    devices=[str(d) for d in jax.devices()],
    gpu=subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip(),
    dtype="float32",
    mesh_nm=args.dx_nm,
    shape_zyx=namespace["shape"],
    timesteps=len(namespace["times"]),
    run_time_s=namespace["run_time"],
    checkpoint_interval=args.checkpoint,
    warmup_calls=3,
    samples=args.samples,
    notebook_sha256=hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
    gradient_relative_difference=gradient_error,
    objective_absolute_difference=value_error,
    compilation_cache_enabled=False,
    provenance=provenance(),
    objective_value=objective_value,
    gradient_norm=gradient_norm,
    beta=args.beta,
    parameters_npz=str(args.parameters_npz) if args.parameters_npz else None,
    parameters_sha256=hashlib.sha256(np.asarray(parameters).tobytes()).hexdigest(),
)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(
    json.dumps(
        dict(
            metadata=metadata,
            measurements=report,
            cuda_speedup_over_jax=speedups,
            finite_differences=finite_differences,
            speedup_95_percent_intervals={
                kind: speedup_interval(
                    report["cuda"][kind]["samples_s"], report["jax"][kind]["samples_s"]
                )
                for kind in ("objective", "value_and_grad")
            },
        ),
        indent=2,
    )
)
print("CUDA speedup over GPU JAX:", speedups, flush=True)
print("Saved:", args.output, flush=True)
