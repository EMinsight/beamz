"""Same-run plane/aperture convergence probe from beamzorg/beamz#309.

Run from the repository root, preferably with CUDA JAX:
    python -m tests.characterization.sbend_monitor_case --mesh 10 --output probe.json

The geometry is the issue's standalone export of GDS_FDTD sbend_dontfabme.
No GDS reader, PDK, or reference-engine installation is required.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
from pathlib import Path

import numpy as np

import beamz as bz
from beamz.analysis import s_parameters
from beamz.devices.sources.time import gaussian_band_pulse


class Probe:
    def __init__(
        self,
        scene,
        mesh,
        *,
        widths=(1.35, 2.7, 4.0),
        distances=(0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75),
        straight=False,
    ):
        self.scene, self.mesh = scene, mesh
        self.widths, self.distances, self.straight = widths, distances, straight

    def run(self):
        # Copy before replacing geometry for the straight-guide control.
        scene = json.loads(json.dumps(self.scene))
        if self.straight:
            guide = dict(scene["structures"][2])
            guide["vertices_m"] = [
                [-1e-7, 3e-6],
                [7.1e-6, 3e-6],
                [7.1e-6, 3.5e-6],
                [-1e-7, 3.5e-6],
            ]
            scene["structures"] = [guide]
            scene["ports"]["opt2"]["center_m"][1] = scene["ports"]["opt1"]["center_m"][
                1
            ]
        design = bz.Design(
            width=scene["size_m"][0],
            height=scene["size_m"][1],
            depth=scene["size_m"][2],
            material=bz.Material(permittivity=scene["background_epsilon"]),
            structures=[
                bz.Polygon(
                    vertices=s["vertices_m"],
                    z=s["z_m"],
                    depth=s["depth_m"],
                    material=bz.Material(permittivity=s["epsilon"]),
                )
                for s in scene["structures"]
            ],
        )
        dx, dt = bz.dxdt(
            1.55e-6,
            n_max=scene["n_max"],
            dims=3,
            safety_factor=0.999,
            points_per_wavelength=self.mesh,
        )
        frequencies = 299792458 / (np.array([1.6, 1.55, 1.5]) * 1e-6)
        pulse = gaussian_band_pulse(
            frequencies,
            carrier_frequency=299792458 / 1.55e-6,
            dt=dt,
            run_after_sources_uoc=90,
            max_output_distance_um=3,
        )
        mode = bz.ModeSpec(polarization="te")

        def port(original, name, inward_offset_um):
            p = scene["ports"][original]
            center = list(p["center_m"])
            center[0] += (1 if p["direction"] == "+" else -1) * inward_offset_um * 1e-6
            return bz.Port(
                center=tuple(center),
                size=tuple(p["size_m"]),
                direction=p["direction"],
                name=name,
                mode_spec=mode,
            )

        src = port("opt1", "source", -1.4)
        input_port = port("opt1", "input", -0.5)
        outputs, planes = [], {}
        for width in self.widths:
            for distance in self.distances:
                name = f"w{width:g}_d{distance:g}"
                base = port("opt2", name, -distance)
                size = (0, width * 1e-6, width * 2.196 / 2.7 * 1e-6)
                outputs.append(
                    bz.Port(
                        center=base.center,
                        size=size,
                        direction=base.direction,
                        name=name,
                        mode_spec=mode,
                    )
                )
                planes[name] = {
                    "distance_um": distance,
                    "size_um": [v / 1e-6 for v in size],
                }
        source = bz.ModeSource(
            center=src.center,
            size=src.size,
            direction=src.direction,
            mode_spec=mode,
            source_time=bz.SampledSignal(
                values=pulse.signal,
                quadrature=pulse.signal_quadrature,
                dt=dt,
                freq0=299792458 / 1.55e-6,
            ),
        )
        sim = bz.Simulation(
            design=design,
            sources=[source],
            monitors=[p.to_monitor(frequencies) for p in [input_port, *outputs]],
            boundaries=[
                bz.PML(
                    edges=("left", "right", "top", "bottom", "front", "back"),
                    thickness=1e-6,
                )
            ],
            time=pulse.time,
            resolution=dx,
            setup_device="cpu",
        )
        results = sim.run(
            termination=bz.AutoTermination(
                min_steps=int(np.ceil((pulse.source_end_time + pulse.tail_time) / dt)),
                field_decay=1e-4,
            )
        )
        extracted = s_parameters(
            results,
            source_port="input",
            ports=[input_port, *outputs],
            frequencies=frequencies,
        )
        waves = extracted.diagnostics["waves"]
        return {
            "beamz": bz.__version__,
            "jax": importlib.metadata.version("jax"),
            "mesh": self.mesh,
            "dx_nm": dx / 1e-9,
            "wavelength_um": (299792458 / frequencies / 1e-6).tolist(),
            "straight_control": self.straight,
            "planes": planes,
            "s21_db": {
                p.name: (
                    20 * np.log10(np.abs(extracted.s_matrix[(p.name, "input")]))
                ).tolist()
                for p in outputs
            },
            "P_in": np.asarray(extracted.diagnostics["P_in"]).tolist(),
            "valid_mask": np.asarray(extracted.diagnostics["valid_mask"]).tolist(),
            "termination": dataclasses.asdict(results.termination),
            "waves": {
                name: {
                    key: {
                        "real": np.real(values[key]).tolist(),
                        "imag": np.imag(values[key]).tolist(),
                    }
                    for key in [
                        "a_plus",
                        "a_minus",
                        "mode_neff",
                        "projection_residual",
                        "condition_number",
                    ]
                }
                for name, values in waves.items()
            },
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--widths", type=float, nargs="+", default=[1.35, 2.7, 4.0])
    parser.add_argument(
        "--distances",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    )
    parser.add_argument("--straight", action="store_true")
    args = parser.parse_args()
    if args.mesh <= 0 or any(w <= 0 or w > 4.0 for w in args.widths):
        parser.error("mesh must be positive and widths must be in (0, 4] um")
    if any(d < 0.25 or d > 1.75 for d in args.distances):
        parser.error(
            "distances must be in [0.25, 1.75] um to stay in the clear straight lead"
        )
    scene = json.loads(
        (Path(__file__).with_name("fixtures") / "issue_309_sbend.json").read_text()
    )
    record = Probe(
        scene,
        args.mesh,
        widths=args.widths,
        distances=args.distances,
        straight=args.straight,
    ).run()
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
