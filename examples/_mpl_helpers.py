"""Compatibility wrappers for older examples.

New examples should import plotting helpers from ``beamz.visual.mpl`` or call
object methods such as ``design.show()``, ``grid.show()``, and ``source.show()``.
"""

from beamz.visual.mpl import (
    plot_design,
    plot_grid,
    plot_mode_profile,
    plot_signal,
    save_snapshot_video,
    show_snapshots,
)

__all__ = [
    "plot_design",
    "plot_grid",
    "plot_mode_profile",
    "plot_signal",
    "save_snapshot_video",
    "show_snapshots",
]
