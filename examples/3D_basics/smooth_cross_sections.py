"""Render vector-accurate 3D simulation setup cross sections.

Run this file from the repository root to regenerate the two images in
``docs/assets``. The images are drawn from design geometry: changing
``resolution`` will not turn the curved ring or material boundaries into pixels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import beamz as bz


def build_simulation() -> tuple[bz.Simulation, float, float]:
    """Build a compact SOI add-drop ring layout with setup devices."""
    air = bz.Material(permittivity=1.0)
    oxide = bz.Material(permittivity=1.45**2)
    silicon = bz.Material(permittivity=3.48**2)
    width, height, depth = 13 * bz.um, 7.2 * bz.um, 3.5 * bz.um
    core_z = 1.5 * bz.um
    center_y = 0.5 * height
    ring_center_y = center_y + 1.5 * bz.um

    design = bz.Design(width=width, height=height, depth=depth, background=air)
    design += bz.Rectangle(
        position=(0.0, 0.0, 0.0),
        width=width,
        height=height,
        depth=core_z,
        material=oxide,
    )
    design += bz.Rectangle(
        position=(0.0, center_y - 0.25 * bz.um, core_z),
        width=width,
        height=0.5 * bz.um,
        depth=0.22 * bz.um,
        material=silicon,
    )
    design += bz.Ring(
        position=(7.2 * bz.um, ring_center_y, core_z),
        inner_radius=1.15 * bz.um,
        outer_radius=1.45 * bz.um,
        depth=0.22 * bz.um,
        material=silicon,
    )

    frequency = bz.LIGHT_SPEED / (1.55 * bz.um)
    source = bz.ModeSource(
        center=(1.1 * bz.um, center_y, core_z + 0.11 * bz.um),
        size=(0.0, 3.2 * bz.um, 0.9 * bz.um),
        source_time=bz.GaussianPulse(freq0=frequency, fwidth=2e13),
        direction="+",
    )
    monitors = (
        bz.FluxMonitor(
            center=(11.9 * bz.um, center_y, core_z + 0.11 * bz.um),
            size=(0.0, 3.2 * bz.um, 0.9 * bz.um),
            freqs=[frequency],
            name="through",
        ),
        bz.FluxMonitor(
            center=(7.2 * bz.um, center_y + 2.95 * bz.um, core_z + 0.11 * bz.um),
            size=(2.4 * bz.um, 0.0, 0.9 * bz.um),
            freqs=[frequency],
            name="drop",
        ),
    )
    return (
        bz.Simulation(
            design=design,
            sources=[source],
            monitors=monitors,
            boundaries=[bz.PML(thickness=0.6 * bz.um)],
            # This is deliberately much coarser than the geometry. The plot below
            # stays smooth because it draws the design, not this numerical mesh.
            resolution=0.45 * bz.um,
            time=np.array([0.0, 1e-15]),
        ),
        core_z + 0.11 * bz.um,
        ring_center_y,
    )


def plot_setup(*, xlim=None):
    """Create the material, PML, source, and monitor setup figure."""
    simulation, core_z, center_y = build_simulation()
    fig, axes = simulation.plot(
        z=core_z,
        y=center_y,
        xlim=xlim,
        figsize=(13, 5.2),
        show=False,
    )
    fig.legend(
        handles=(
            Patch(facecolor="#f7fbff", label="air"),
            Patch(facecolor="#9ecae1", label="SiO₂"),
            Patch(facecolor="#d81b60", label="Si"),
            Patch(facecolor="#666666", alpha=0.22, hatch="///", label="PML"),
            Line2D([], [], color="#2ca02c", lw=3, label="mode source"),
            Line2D([], [], color="#ff9800", lw=2, label="DFT power planes"),
        ),
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        frameon=False,
    )
    return fig, axes


def main() -> None:
    """Generate full-layout and curved-geometry setup previews."""
    output_dir = Path(__file__).resolve().parents[2] / "docs" / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, xlim in (
        ("smooth-simulation-cross-sections.png", None),
        ("smooth-simulation-cross-sections-zoom.png", (4.8, 9.6)),
    ):
        fig, _ = plot_setup(xlim=xlim)
        fig.savefig(output_dir / name, dpi=200, bbox_inches="tight")
        if xlim is None:
            # The layout artists are paths, so this stays vector-sharp in docs,
            # slides, and design reviews as well as in the PNG preview.
            svg_path = output_dir / "smooth-simulation-cross-sections.svg"
            fig.savefig(svg_path, bbox_inches="tight")
            # Matplotlib wraps SVG path data with trailing spaces. Strip only that
            # formatting noise so regenerated assets pass repository whitespace
            # checks without changing their vector paths.
            svg_path.write_text(
                "\n".join(line.rstrip() for line in svg_path.read_text().splitlines())
                + "\n"
            )
        plt.close(fig)


if __name__ == "__main__":
    main()
