"""Cold-process modal workflow timing, memory and numerical exports across revisions.

Run this same script with --root pointing at each isolated checkout. Native CUDA
extensions must match each checkout's ABI. Enable both CPU and CUDA backends.
"""

# Imports follow --root selection so each worker loads the requested checkout.
# ruff: noqa: E402

import argparse
import contextlib
import hashlib
import json
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", required=True)
parser.add_argument("--backend", choices=["jax", "cuda_streamed"], required=True)
parser.add_argument("--scene", choices=["straight", "junction"], required=True)
parser.add_argument("--full", action="store_true")
parser.add_argument("--compile-only", action="store_true")
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path(args.root).resolve()
sys.path.insert(0, str(root))

import jax
import numpy as np
import psutil
import pynvml

import beamz as bz
from beamz.design.raster import RasterOptions
from beamz.simulation.execute import (
    _compiled_source_launch_powers,
    _decode_monitor_results,
    build_program_scan,
    initial_program_state,
)
from beamz.simulation.results import SimulationResults
from beamz.simulation.sharding import place_tree

jax.config.update("jax_enable_compilation_cache", False)
np.random.seed(0)
assert Path(bz.__file__).resolve().is_relative_to(root), bz.__file__
assert jax.devices()[0].platform == "gpu"
assert jax.devices("cpu"), "CPU setup must be available"
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
process = psutil.Process()
phases = {}
phase = "initialization"
peaks = {}
stop = threading.Event()


def sample():
    gpu = next(
        (
            p.usedGpuMemory
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if p.pid == os.getpid()
        ),
        0,
    )
    old = peaks.setdefault(phase, {"host_rss": 0, "gpu_process": 0})
    old["host_rss"] = max(old["host_rss"], process.memory_info().rss)
    old["gpu_process"] = max(old["gpu_process"], gpu)


def poll():
    while not stop.wait(0.02):
        sample()


thread = threading.Thread(target=poll, daemon=True)
thread.start()


@contextlib.contextmanager
def measure(name):
    global phase
    phase = name
    sample()
    start = time.perf_counter()
    yield
    sample()
    print(f"Completed {name} in {time.perf_counter() - start:.3f} s", flush=True)
    phases[name] = dict(
        seconds=time.perf_counter() - start,
        sampled_peak=peaks[name].copy(),
        host_cumulative_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        gpu_allocator=jax.devices()[0].memory_stats(),
    )


um = bz.um
frequency = bz.LIGHT_SPEED / (1.55 * um)
fwidth = 0.1 * frequency
frequencies = np.linspace(frequency - 2 * fwidth, frequency + 2 * fwidth, 17)
size = np.array((8, 4, 3) if args.scene == "straight" else (8, 7, 6)) * um
pml = (0.6 if args.scene == "straight" else 1.25) * um
source_x = -size[0] / 2 + pml + 0.75 * um
plane = (0, 2 * um, 1.5 * um)
mode_spec = bz.ModeSpec(
    # The narrow full-tensor input guide has one safely confined mode across
    # this band. Radiation-mode tensor ports are outside the supported scope.
    num_modes=1 if args.full else 3,
    target_neff=0.98 * 3.48,
    num_freqs=7,
    polarization=None if args.full else "te",
)

with measure("geometry_and_raster"):
    silicon = bz.Material(permittivity=3.48**2)
    design = bz.Design(background=bz.Material(permittivity=1.0))
    design += bz.Box(
        center=(0, 0, -size[2] / 2),
        size=(bz.inf, bz.inf, size[2]),
        material=bz.Material(permittivity=1.45**2),
    )
    w = 0.45 * um
    half = size[0] / 2
    if args.scene == "straight":
        vertices = [(-half, -w / 2), (half, -w / 2), (half, w / 2), (-half, w / 2)]
    else:
        wide = 1.4 * um
        vertices = [
            (-half, -w / 2),
            (0, -w / 2),
            (0, -wide / 2),
            (half, -wide / 2),
            (half, wide / 2),
            (0, wide / 2),
            (0, w / 2),
            (-half, w / 2),
        ]
    design += bz.Polygon(
        vertices=vertices, z=0, depth=0.22 * um, sidewall_angle=10, material=silicon
    )
    grid = (
        bz.GridSpec.uniform(40e-9, courant=0.8)
        if args.scene == "straight"
        else bz.GridSpec.auto(
            wavelength=1.55 * um, min_steps_per_wvl=10, max_scale=1.25, courant=0.8
        )
    )
    source = bz.ModeSource(
        center=(source_x, 0, 0),
        size=plane,
        direction="+",
        source_time=bz.GaussianPulse(freq0=frequency, fwidth=fwidth, offset=0.75),
        mode_spec=mode_spec,
    )
    monitors = [
        bz.FieldMonitor(
            center=(0, 0, 0.11 * um),
            size=(size[0], size[1], 0),
            freqs=[frequency],
            fields=("Ey",),
            name="field",
        )
    ]
    for name, x in [("near", source_x + 0.5 * um), ("out", -source_x)]:
        monitors += [
            bz.ModeMonitor(
                center=(x, 0, 0),
                size=plane,
                freqs=frequencies,
                mode_spec=mode_spec,
                name=name,
            ),
            bz.FluxMonitor(
                center=(x, 0, 0), size=plane, freqs=frequencies, name=name + "_flux"
            ),
        ]
    sim = bz.Simulation(
        domain=size,
        grid_spec=grid,
        design=design,
        sources=[source],
        monitors=monitors,
        boundaries=[bz.PML(thickness=pml, formulation="cpml")],
        run_time=250e-15,
        setup_device="cpu",
        raster_options=RasterOptions(
            smoothing="farjadpour_full" if args.full else "farjadpour_diagonal"
        ),
    )
    material = sim._material_grid(progress=False)

with measure("plan_and_source_modes"):
    plan = sim.compile(backend=args.backend)
    jax.block_until_ready(plan.coefficients)
with measure("device_placement"):
    state = place_tree(plan, initial_program_state(plan, t=0.0, current_step=0))
    coeffs = place_tree(plan, plan.coefficients)
    jax.block_until_ready((state, coeffs))
with measure("jit"):
    executable = build_program_scan(plan).lower(state, coeffs).compile()
coefficient_layout = {
    prefix + c: {
        "type": type(value).__name__,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "bytes": value.nbytes,
    }
    for prefix in ("e_permittivity_", "e_source_", "e_inverse_diagonal_")
    for c in "xyz"
    for value in (getattr(plan.coefficients, prefix + c),)
}
if args.compile_only:
    stop.set()
    thread.join()
    report = dict(
        root=str(root),
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        backend=args.backend,
        scene=args.scene,
        full=args.full,
        compile_only=True,
        coefficient_layout=coefficient_layout,
        phases=phases,
    )
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    sys.exit(0)
with measure("first_run"):
    output = jax.block_until_ready(executable(state, coeffs))
with measure("warmup"):
    output = jax.block_until_ready(executable(state, coeffs))
samples = []
with measure("steady_runs"):
    for _ in range(args.repeats):
        start = time.perf_counter()
        output = jax.block_until_ready(executable(state, coeffs))
        samples.append(time.perf_counter() - start)
with measure("results_and_monitor_modes"):
    result = SimulationResults.from_run(
        sim,
        runtime_fields=plan.grid,
        monitor_results=_decode_monitor_results(sim, plan, output),
        completed_steps=int(output.current_step),
        store_full_materials=False,
        source_launch_powers=_compiled_source_launch_powers(plan, len(sim.sources)),
    )
    arrays = {
        name: np.asarray(getattr(output, name))
        for name in ("ex", "ey", "ez", "hx", "hy", "hz", "dft_vec_re", "dft_vec_im")
    }
    for name in ("near", "out"):
        modal = result.mode(name)
        arrays[name + "_amps"] = np.asarray(modal.amps)
        arrays[name + "_projected_flux"] = np.asarray(modal.modal_flux)
        arrays[name + "_modal_flux"] = np.asarray(modal.flux)
        arrays[name + "_flux"] = np.asarray(result[name + "_flux"].flux)
    for name, value in arrays.items():
        assert np.isfinite(value).all(), name
stop.set()
thread.join()
analysis = executable.memory_analysis()
report = dict(
    root=str(root),
    revision=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip(),
    backend=args.backend,
    scene=args.scene,
    full=args.full,
    shape=list(sim.grid.shape),
    steps=plan.config.num_steps,
    dt=plan.config.dt,
    frequencies=frequencies.tolist(),
    polarization=mode_spec.polarization,
    source_profiles=7,
    monitor_modes=mode_spec.num_modes,
    sidewall_angle=10,
    setup_device="cpu",
    coefficient_layout=coefficient_layout,
    phases=phases,
    samples=samples,
    median_seconds_per_step=float(np.median(samples)) / plan.config.num_steps,
    executable_memory={
        name: getattr(analysis, name)
        for name in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "temp_size_in_bytes",
            "alias_size_in_bytes",
        )
    },
    jax=jax.__version__,
    device=jax.devices()[0].device_kind,
    environment={
        key: os.environ.get(key)
        for key in (
            "JAX_PLATFORMS",
            "XLA_PYTHON_CLIENT_PREALLOCATE",
            "OPENBLAS_NUM_THREADS",
        )
    },
)
if args.backend == "cuda_streamed":
    import beamz._cuda as extension

    report["cuda_extension"] = dict(
        version=extension.__version__,
        sha256=hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
    )
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(output_path.with_suffix(".npz"), **arrays)
output_path.write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
