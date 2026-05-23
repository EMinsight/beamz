from pathlib import Path

import numpy as np

from beamz import (
    LIGHT_SPEED,
    PML,
    Design,
    GaussianSource,
    Material,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
    um,
)
from beamz.visual.data import signal_plot_data


def _make_snapshot_sim():
    wl = 1.55 * um
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=2, safety_factor=0.95, points_per_wavelength=8
    )
    domain = 4.0 * wl
    steps = 12
    t = np.arange(0, steps * dt, dt)
    freq = LIGHT_SPEED / wl
    signal = ramped_cosine(
        t,
        amplitude=1.0,
        frequency=freq,
        ramp_duration=2 / freq,
        t_max=t[-1] * 0.4,
    )
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))
    source = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    return Simulation(
        design=design,
        sources=[source],
        boundaries=[PML(thickness=1.0 * wl)],
        time=t,
        resolution=dx,
    )


def test_simulation_run_collects_snapshot_layout_and_field_payload():
    seen = []
    sim = _make_snapshot_sim()

    results = sim.run(
        snapshot_field="Ez",
        snapshot_interval=4,
        snapshot_callback=seen.append,
        store_snapshots=True,
        progress=False,
    )

    assert len(seen) == 3
    snapshot = seen[0]
    assert snapshot["field_name"] == "Ez"
    assert snapshot["units"] == "V/µm"
    assert snapshot["layout"]["design"]["width"] == sim.design.width
    assert np.max(np.abs(snapshot["field"])) > 0.0

    assert results is not None
    assert len(results.snapshots) == 3
    assert results["snapshots"][0]["step"] == 4


def test_signal_plot_data_scales_picoseconds():
    payload = signal_plot_data(np.array([0.0, 1.0]), np.array([0.0, 2.0e-12]))

    assert payload["time_unit"] == "ps"
    assert np.allclose(payload["t_scaled"], np.array([0.0, 2.0]))


def test_matplotlib_backend_isolated_to_visual_mpl():
    root = Path(__file__).resolve().parents[1] / "beamz"
    offenders = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel == "visual/mpl.py":
            continue
        text = path.read_text()
        if "import matplotlib" in text or "from matplotlib" in text:
            offenders.append(rel)

    assert offenders == []


def test_import_beamz_does_not_import_pyplot():
    import subprocess
    import sys

    code = "import sys, beamz; print('matplotlib.pyplot' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_simulation_run_accepts_animate_live_kwargs(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from beamz.visual import mpl as mpl_backend

    rendered_steps = []

    def fake_snapshot_figure(snapshot, **kwargs):
        rendered_steps.append(snapshot["step"])
        fig, ax = plt.subplots()
        return fig, ax

    class FakePyplot:
        @staticmethod
        def show(*args, **kwargs):
            return None

        @staticmethod
        def pause(*args, **kwargs):
            return None

    monkeypatch.setattr(mpl_backend, "snapshot_figure", fake_snapshot_figure)
    monkeypatch.setattr(mpl_backend, "_pyplot", lambda: FakePyplot)

    sim = _make_snapshot_sim()
    results = sim.run(
        animate_live="Ez",
        animation_interval=4,
        store_snapshots=False,
        progress=False,
    )

    assert results is None
    assert rendered_steps == [4, 8, 12]
    plt.close("all")


def test_simulation_run_accepts_fixed_live_cmap_limits(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from beamz.visual import mpl as mpl_backend

    rendered_limits = []

    def fake_snapshot_figure(snapshot, **kwargs):
        rendered_limits.append((kwargs.get("vmin"), kwargs.get("vmax")))
        fig, ax = plt.subplots()
        return fig, ax

    class FakePyplot:
        @staticmethod
        def show(*args, **kwargs):
            return None

        @staticmethod
        def pause(*args, **kwargs):
            return None

    monkeypatch.setattr(mpl_backend, "snapshot_figure", fake_snapshot_figure)
    monkeypatch.setattr(mpl_backend, "_pyplot", lambda: FakePyplot)

    sim = _make_snapshot_sim()
    sim.run(
        animate_live="Ez",
        animation_interval=4,
        cmap_limits=(-0.25, 0.25),
        store_snapshots=False,
        progress=False,
    )

    assert rendered_limits == [(-0.25, 0.25), (-0.25, 0.25), (-0.25, 0.25)]
    plt.close("all")


def test_simulation_run_dynamic_live_cmap_limits_are_default(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from beamz.visual import mpl as mpl_backend

    rendered_limits = []

    def fake_snapshot_figure(snapshot, **kwargs):
        rendered_limits.append((kwargs.get("vmin"), kwargs.get("vmax")))
        fig, ax = plt.subplots()
        return fig, ax

    class FakePyplot:
        @staticmethod
        def show(*args, **kwargs):
            return None

        @staticmethod
        def pause(*args, **kwargs):
            return None

    monkeypatch.setattr(mpl_backend, "snapshot_figure", fake_snapshot_figure)
    monkeypatch.setattr(mpl_backend, "_pyplot", lambda: FakePyplot)

    sim = _make_snapshot_sim()
    sim.run(
        animate_live="Ez",
        animation_interval=4,
        cmap_limits="dynamic",
        store_snapshots=False,
        progress=False,
    )

    assert rendered_limits == [(None, None), (None, None), (None, None)]
    plt.close("all")


def test_simulation_animation_convenience_methods_forward_kwargs(monkeypatch):
    calls = []
    sim = _make_snapshot_sim()

    def fake_run(**kwargs):
        calls.append(kwargs)
        return "result"

    monkeypatch.setattr(sim, "run", fake_run)

    assert sim.animate("Hy", animation_interval=2, cmap_limits=(-1.0, 1.0)) == "result"
    assert sim.save_video("out.mp4", field="Ez", video_fps=24) == "result"

    assert calls[0] == {
        "animation_interval": 2,
        "cmap_limits": (-1.0, 1.0),
        "animate_live": "Hy",
    }
    assert calls[1] == {
        "video_fps": 24,
        "save_video": "out.mp4",
        "video_field": "Ez",
    }
