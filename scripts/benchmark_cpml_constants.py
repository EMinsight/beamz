"""Compare CPML constant storage in two checkouts on one local CUDA GPU.

Requires CUDA-enabled JAX, psutil and nvidia-ml-py. Each checkout needs its native
raster extension; native CUDA cases also require its matching CUDA extension.
Example:
    python scripts/benchmark_cpml_constants.py --baseline-root /path/to/main \
        --output benchmarks/results/cpml-constants
"""

import argparse
import ast
import hashlib
import json
import os
import resource
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path


def _matrix(args):
    cases = [
        ("sphere", n, smoothing, backend)
        for n in (96, 224)
        for backend in ("jax", "cuda_streamed")
        for smoothing in ("volume", "farjadpour_diagonal", "farjadpour_full")
        if backend == "jax" or smoothing != "farjadpour_full"
    ]
    cases += [
        (case, 128, "farjadpour_diagonal", backend)
        for case in ("lossy", "sponge")
        for backend in ("jax", "cuda_streamed")
    ]
    cases += [
        ("ubend", 30, smoothing, "jax")
        for smoothing in ("volume", "farjadpour_diagonal", "farjadpour_full")
    ]
    cases += [("ubend", 30, "farjadpour_diagonal", "cuda_streamed")]
    roots = {"main": args.baseline_root, "pr": args.root}
    env = os.environ.copy()
    env.update(
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        JAX_ENABLE_X64="false",
        JAX_PLATFORMS="cuda",
        BEAMZ_DISABLE_JAX_PERSISTENT_CACHE="1",
    )
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    args.output.mkdir(parents=True, exist_ok=True)
    for case, n, smoothing, backend in cases:
        label = f"{case}-{n}-{smoothing}-{backend}"
        reference = args.output / "references" / label
        for role, root in roots.items():
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--root",
                str(root),
                "--case",
                case,
                "--n",
                str(n),
                "--smoothing",
                smoothing,
                "--backend",
                backend,
                "--reference",
                str(reference),
                "--output",
                str(args.output / f"{label}-{role}.json"),
            ]
            if role == "main":
                command += ["--save-reference"]
            print(role, label, flush=True)
            with (args.output / f"{label}-{role}.log").open("w") as log:
                subprocess.run(
                    command, env=env, check=True, stdout=log, stderr=subprocess.STDOUT
                )


def _measure(a):
    import jax
    import numpy as np
    import psutil
    import pynvml

    import beamz as bz
    from beamz.design import raster
    from beamz.simulation.execute import build_scan, initial_program_state

    assert Path(bz.__file__).resolve().parents[1] == Path(a.root).resolve()
    assert jax.default_backend() == "gpu"
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    peaks = {"gpu_process_bytes": 0, "host_rss_bytes": 0}
    stop = threading.Event()
    proc = psutil.Process()

    def sample():
        while not stop.is_set():
            peaks["host_rss_bytes"] = max(
                peaks["host_rss_bytes"], proc.memory_info().rss
            )
            for v in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if v.pid == os.getpid():
                    peaks["gpu_process_bytes"] = max(
                        peaks["gpu_process_bytes"], v.usedGpuMemory
                    )
            stop.wait(0.05)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    t0 = time.perf_counter()
    if a.case == "ubend":
        notebook_path = (
            Path(__file__).resolve().parents[1]
            / "examples/notebooks/interface_smoothing_choice.ipynb"
        )
        notebook = json.loads(notebook_path.read_text())
        ns = {"bz": bz, "np": np, "um": bz.um, "RasterOptions": raster.RasterOptions}
        # Find the builder by its definition, independent of notebook cell numbering.
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]))
            if not any(
                isinstance(node, ast.FunctionDef) and node.name == "make_sim"
                for node in tree.body
            ):
                continue
            definitions = []
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "preview" for t in node.targets
                ):
                    break
                definitions.append(node)
            exec(
                compile(
                    ast.Module(body=definitions, type_ignores=[]),
                    str(notebook_path),
                    "exec",
                ),
                ns,
            )
            break
        else:
            raise ValueError("Notebook has no make_sim builder")
        sim = ns["make_sim"](a.n * 1e-9, a.smoothing)
    else:
        dx = 100e-9
        n = a.n
        size = (n * dx,) * 3
        dt = 0.5 * dx / (bz.LIGHT_SPEED * np.sqrt(3))
        signal = np.exp(-(((np.arange(a.steps) - 48) / 18) ** 2)) * np.sin(
            2 * np.pi * (bz.LIGHT_SPEED / 1.55e-6) * np.arange(a.steps) * dt
        )
        scene = raster.Scene(
            (
                raster.Material(1.44**2),
                raster.Material(
                    2.0**2, conductivity=50.0 if a.case == "lossy" else 0.0
                ),
            ),
            (
                raster.Object(
                    raster.Sphere(tuple(v / 2 for v in size), size[0] * 0.25), 1
                ),
            ),
        )
        sim = bz.Simulation(
            scene=scene,
            raster_grid=raster.Grid.uniform((0.0, 0.0, 0.0), size, (n,) * 3),
            time=np.arange(a.steps) * dt,
            sources=(
                bz.GaussianSource(
                    position=(0.4 * size[0], 0.5 * size[1], 0.5 * size[2]),
                    width=3 * dx,
                    signal=signal.astype(np.float32),
                ),
            ),
            monitors=(
                bz.FieldMonitor(
                    center=(0.65 * size[0], 0.5 * size[1], 0.5 * size[2]),
                    size=(0, 0.3 * size[1], 0.3 * size[2]),
                    freqs=[bz.LIGHT_SPEED / 1.55e-6],
                    fields=("Ey", "Hz"),
                    name="flux_plane",
                ),
            ),
            boundaries=(
                bz.PML(
                    thickness=10 * dx,
                    formulation="sponge" if a.case == "sponge" else "cpml",
                ),
            ),
            raster_options=raster.RasterOptions(smoothing=a.smoothing),
        )
    program = sim.compile(num_steps=a.steps, backend=a.backend)
    state = initial_program_state(program, t=0, current_step=0, monitor_steps=a.steps)
    jax.block_until_ready(state)
    setup_s = time.perf_counter() - t0
    setup_mem = jax.devices()[0].memory_stats()
    t0 = time.perf_counter()
    executable = (
        build_scan(program, donate_state=False)
        .lower(state, program.coefficients)
        .compile()
    )
    compile_s = time.perf_counter() - t0
    for _ in range(3):
        result = jax.block_until_ready(executable(state, program.coefficients))
        del result
    samples = []
    for _ in range(a.samples):
        t0 = time.perf_counter()
        result = jax.block_until_ready(executable(state, program.coefficients))
        samples.append(time.perf_counter() - t0)
        del result
    result = jax.block_until_ready(executable(state, program.coefficients))
    mem = jax.devices()[0].memory_stats()
    time.sleep(0.06)
    stop.set()
    thread.join()
    validation = []
    ref = Path(a.reference)
    ref.mkdir(parents=True, exist_ok=True)
    for i, leaf in enumerate(jax.tree.leaves(result)):
        actual = np.asarray(jax.device_get(leaf))
        path = ref / f"{i:03}.npy"
        finite = bool(np.isfinite(actual).all())
        if a.save_reference:
            np.save(path, actual)
            expected = actual
        else:
            expected = np.load(path, mmap_mode="r")
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        exact = bool(np.array_equal(actual, expected))
        err = float(np.max(np.abs(actual - expected), initial=0))
        close = bool(np.allclose(actual, expected, rtol=3e-5, atol=3e-6))
        validation.append(
            dict(
                index=i,
                shape=actual.shape,
                dtype=str(actual.dtype),
                sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                finite=finite,
                exact=exact,
                max_abs_error=err,
                close=close,
                max_abs_value=float(np.max(np.abs(actual), initial=0)),
            )
        )
        del actual, expected
    rows = dict(
        case=a.case,
        n=a.n,
        backend=a.backend,
        smoothing=a.smoothing,
        steps=a.steps,
        samples_s=samples,
        median_s=statistics.median(samples),
        gcups=np.prod(program.grid.material_grid.shape)
        * a.steps
        / statistics.median(samples)
        / 1e9,
        setup_s=setup_s,
        compile_s=compile_s,
        setup_memory=setup_mem,
        memory=mem,
        peaks=peaks,
        host_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        grid_shape=program.grid.material_grid.shape,
        root=str(a.root),
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=a.root, text=True
        ).strip(),
        jax=jax.__version__,
        device=str(jax.devices()[0]),
        gpu_name=pynvml.nvmlDeviceGetName(handle),
        driver=pynvml.nvmlSystemGetDriverVersion(),
        validation=validation,
        material_shapes={
            k: list(np.shape(getattr(program.grid, k)))
            for k in ("conductivity", "permeability", "sig_x", "mu_hx")
        },
        cuda_flags=program.config.cuda_flags,
    )
    if a.backend == "cuda_streamed":
        import beamz._cuda as ext

        rows["extension"] = {
            "sha256": hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest(),
            "version": ext.__version__,
            "abi": ext.__abi_version__,
        }
    Path(a.output).write_text(json.dumps(rows, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: rows[k]
                for k in (
                    "case",
                    "n",
                    "backend",
                    "smoothing",
                    "median_s",
                    "setup_s",
                    "compile_s",
                    "peaks",
                )
            }
        ),
        flush=True,
    )
    assert all(v["finite"] and v["close"] for v in validation), (
        "numerical parity failed"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--baseline-root", type=Path, help="Run the full paired matrix")
    parser.add_argument(
        "--output", type=Path, required=True, help="Matrix directory or worker JSON"
    )
    parser.add_argument(
        "--case", choices=("sphere", "lossy", "sponge", "ubend"), default="sphere"
    )
    parser.add_argument("--backend", choices=("jax", "cuda_streamed"), default="jax")
    parser.add_argument(
        "--smoothing",
        choices=("volume", "farjadpour_diagonal", "farjadpour_full"),
        default="farjadpour_diagonal",
    )
    parser.add_argument(
        "--n", type=int, default=128, help="Cube side in cells, or U-bend spacing in nm"
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--save-reference", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.output = args.output.resolve()
    if args.baseline_root:
        args.baseline_root = args.baseline_root.resolve()
        _matrix(args)
        return
    if args.reference is None:
        parser.error("--reference is required for a single worker")
    args.steps = args.steps or (4096 if args.case == "ubend" else 256)
    args.samples = args.samples or (5 if args.case == "ubend" else 7)
    sys.path.insert(0, str(args.root))
    _measure(args)


if __name__ == "__main__":
    main()
