import numpy as np

import beamz as bz


def test_centered_structure_simulation_constructor_builds_design_and_time():
    si = bz.Medium(permittivity=12.0)
    sio2 = bz.Medium(permittivity=2.0)
    grid_spec = bz.GridSpec.auto(min_steps_per_wvl=10, wavelength=1.55 * bz.um)

    sim = bz.Simulation(
        size=(4 * bz.um, 3 * bz.um, 2 * bz.um),
        grid_spec=grid_spec,
        structures=[
            bz.Structure(
                geometry=bz.Box(center=(0, 0, -1 * bz.um), size=(bz.inf, bz.inf, 2 * bz.um)),
                medium=sio2,
            ),
            bz.Structure(
                geometry=bz.Box(
                    center=(0, 0, 0.11 * bz.um),
                    size=(4 * bz.um, 0.45 * bz.um, 0.22 * bz.um),
                ),
                medium=si,
            ),
        ],
        sources=[],
        monitors=[],
        run_time=2e-15,
    )

    assert sim.design.width == 4 * bz.um
    assert sim.design.depth == 2 * bz.um
    assert sim.resolution < 1.55 * bz.um / 10
    assert sim.time.size >= 2


def test_semantic_monitor_wrappers_create_dft_planes_and_shift_with_simulation():
    freqs = np.array([1.0, 2.0])
    monitor = bz.FluxMonitor(
        center=(1.0, 0.0, 0.0),
        size=(0.0, 2.0, 3.0),
        freqs=freqs,
        name="flux",
    )

    sim = bz.Simulation(
        size=(10.0, 8.0, 6.0),
        structures=[],
        sources=[],
        monitors=[monitor],
        resolution=1.0,
        time=np.array([0.0, 1e-15]),
    )

    shifted = sim.monitors[0]
    assert shifted.name == "flux"
    assert shifted.dft_enabled
    np.testing.assert_allclose(shifted.get_dft_frequencies(), freqs)
    assert shifted.start[0] == 6.0
    assert shifted.end[0] == 6.0


def test_simulation_copy_update_is_reset_configuration_copy():
    sim = bz.Simulation(
        size=(2.0, 1.0),
        structures=[],
        sources=[],
        monitors=[],
        resolution=0.5,
        time=np.array([0.0, 1e-15]),
    )
    sim.current_step = 1

    copied = sim.copy(update={"sources": []})

    assert copied is not sim
    assert copied.current_step == 0
    assert copied.sources == []
