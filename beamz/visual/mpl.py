"""Matplotlib rendering backend for BeamZ plot-data payloads."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from beamz.visual.helpers import get_si_scale_and_label


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def _mpl_types():
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle, PathPatch, Rectangle
    from matplotlib.path import Path as MplPath

    return {
        "Circle": Circle,
        "FFMpegWriter": FFMpegWriter,
        "FuncAnimation": FuncAnimation,
        "LinearSegmentedColormap": LinearSegmentedColormap,
        "MplPath": MplPath,
        "PathPatch": PathPatch,
        "Rectangle": Rectangle,
    }


def get_twilight_zero_cmap():
    """Return the historical BeamZ diverging field colormap."""
    LinearSegmentedColormap = _mpl_types()["LinearSegmentedColormap"]
    colors = [
        (1.0, 1.0, 1.0),
        (0.2, 0.3, 0.8),
        (0.1, 0.1, 0.5),
        (0.1, 0.1, 0.1),
        (0.5, 0.1, 0.1),
        (0.8, 0.3, 0.2),
        (1.0, 1.0, 1.0),
    ]
    return LinearSegmentedColormap.from_list("twilight_zero", colors, N=256)


def resolve_cmap(cmap):
    if cmap == "twilight_zero":
        return get_twilight_zero_cmap()
    return cmap


def resolve_cmap_limits(cmap_limits="dynamic", *, vmin=None, vmax=None):
    """Normalize colormap scaling options to matplotlib ``vmin``/``vmax``."""
    explicit_limits = vmin is not None or vmax is not None
    if cmap_limits is None:
        cmap_limits = "dynamic"

    if isinstance(cmap_limits, str):
        if cmap_limits.lower() != "dynamic":
            raise ValueError("cmap_limits must be 'dynamic' or a (vmin, vmax) pair.")
        return vmin, vmax

    if explicit_limits:
        raise ValueError("Use either cmap_limits or vmin/vmax, not both.")

    try:
        low, high = cmap_limits
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "cmap_limits must be 'dynamic' or a (vmin, vmax) pair."
        ) from exc

    return (
        None if low is None else float(low),
        None if high is None else float(high),
    )


def _maybe_show(fig, *, show):
    if show:
        _pyplot().show()
    return fig


def _draw_polygon(ax, payload):
    vertices = payload["vertices"]
    if not vertices:
        return None

    types = _mpl_types()
    MplPath = types["MplPath"]
    PathPatch = types["PathPatch"]

    coords = []
    codes = []
    coords.extend(vertices)
    coords.append(vertices[0])
    codes.append(MplPath.MOVETO)
    if len(vertices) > 1:
        codes.extend([MplPath.LINETO] * (len(vertices) - 1))
    codes.append(MplPath.CLOSEPOLY)

    for hole in payload.get("interiors", []):
        if not hole:
            continue
        coords.extend(hole)
        coords.append(hole[0])
        codes.append(MplPath.MOVETO)
        if len(hole) > 1:
            codes.extend([MplPath.LINETO] * (len(hole) - 1))
        codes.append(MplPath.CLOSEPOLY)

    style = payload["style"]
    patch = PathPatch(
        MplPath(np.asarray(coords), np.asarray(codes)),
        facecolor=style.get("facecolor", "none"),
        edgecolor=style.get("edgecolor", "black"),
        alpha=style.get("alpha", 1.0),
        linestyle=style.get("linestyle", "-"),
    )
    ax.add_patch(patch)
    return patch


def _draw_source(ax, payload):
    style = payload["style"]
    Circle = _mpl_types()["Circle"]
    if payload["shape"] == "gaussian":
        circle = Circle(
            tuple(payload["position"][:2]),
            radius=payload["radius"],
            facecolor=style.get("facecolor", "none"),
            edgecolor=style.get("edgecolor", "orange"),
            linewidth=2,
            alpha=style.get("alpha", 0.8),
            linestyle=style.get("linestyle", "-"),
        )
        ax.add_patch(circle)
        ax.add_patch(
            Circle(
                tuple(payload["position"][:2]),
                radius=max(payload["radius"] * 0.1, 1e-9),
                facecolor=style.get("edgecolor", "orange"),
                edgecolor="none",
                alpha=style.get("alpha", 0.8),
            )
        )
        return circle

    if payload["shape"] != "mode":
        return None

    center = payload["center"]
    half_width = (payload.get("width") or 0.5e-6) / 2.0
    if payload["direction"] in {"+x", "-x"}:
        x = [center[0], center[0]]
        y = [center[1] - half_width, center[1] + half_width]
    else:
        x = [center[0] - half_width, center[0] + half_width]
        y = [center[1], center[1]]

    (line,) = ax.plot(
        x,
        y,
        color=style.get("edgecolor", "crimson"),
        linewidth=3,
        alpha=style.get("alpha", 0.8),
        solid_capstyle="round",
    )

    arrow_length = (payload.get("wavelength") or 0.5e-6) * 0.5
    dx, dy = 0.0, 0.0
    if payload["direction"] == "+x":
        dx = arrow_length
    elif payload["direction"] == "-x":
        dx = -arrow_length
    elif payload["direction"] == "+y":
        dy = arrow_length
    elif payload["direction"] == "-y":
        dy = -arrow_length

    end_x = center[0] + dx
    end_y = center[1] + dy
    ax.plot(
        [center[0], end_x],
        [center[1], end_y],
        color=style.get("edgecolor", "crimson"),
        linewidth=2,
        alpha=style.get("alpha", 0.8),
    )
    marker = {"+x": ">", "-x": "<", "+y": "^", "-y": "v"}.get(payload["direction"], "o")
    ax.plot(
        [end_x],
        [end_y],
        marker=marker,
        markersize=7,
        color=style.get("edgecolor", "crimson"),
        alpha=style.get("alpha", 0.8),
        linestyle="none",
    )
    return line


def _draw_monitor(ax, payload):
    style = payload["style"]
    if payload["shape"] == "line":
        x0, y0 = payload["start"][:2]
        x1, y1 = payload["end"][:2]
        color = style.get("edgecolor", "navy")
        (line,) = ax.plot(
            [x0, x1], [y0, y1], lw=4, color=color, alpha=style.get("alpha", 1.0)
        )
        ax.plot(
            [x0, x1],
            [y0, y1],
            lw=1,
            color=color,
            linestyle=style.get("linestyle", "-"),
        )
        return line

    Rectangle = _mpl_types()["Rectangle"]
    x0, y0 = payload["start"][:2]
    width, height = payload["size"][:2]
    rect = Rectangle(
        (x0, y0),
        width,
        height,
        fill=style.get("facecolor", "none") != "none",
        facecolor=style.get("facecolor", "none"),
        edgecolor=style.get("edgecolor", "navy"),
        alpha=style.get("alpha", 1.0) * 0.3,
        linestyle=style.get("linestyle", "-"),
        linewidth=2,
    )
    ax.add_patch(rect)
    if payload.get("position") is not None:
        ax.text(
            payload["position"][0],
            payload["position"][1],
            payload["name"],
            ha="center",
            va="center",
            fontsize=8,
            color=style.get("edgecolor", "navy"),
        )
    return rect


def _draw_boundaries(ax, layout, line_color="gray", line_opacity=0.5):
    Rectangle = _mpl_types()["Rectangle"]
    for boundary in layout.get("boundaries", []):
        for rect in boundary["rectangles"]:
            ax.add_patch(
                Rectangle(
                    rect["origin"],
                    rect["width"],
                    rect["height"],
                    facecolor="none",
                    edgecolor=line_color,
                    linestyle=":",
                    alpha=line_opacity,
                )
            )


def _configure_axes(ax, design_payload):
    unit = design_payload["scale_unit"]
    scale = design_payload["scale_factor"]
    ax.set_xlabel(f"X ({unit})")
    ax.set_ylabel(f"Y ({unit})")
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x * scale:.1f}")
    ax.yaxis.set_major_formatter(lambda y, pos: f"{y * scale:.1f}")


def _figure_axes(ax, *, figsize):
    plt = _pyplot()
    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize)


def plot_design(
    design,
    *,
    sources=None,
    monitors=None,
    ax=None,
    figsize=None,
    show=True,
    title="Design Layout",
):
    """Plot a design layout and optional source/monitor overlays."""
    payload = design.to_plot_data(sources=sources, monitors=monitors)
    if figsize is None:
        figsize = (6.0, 6.0 * float(design.height) / max(float(design.width), 1e-30))
    fig, ax = _figure_axes(ax, figsize=figsize)

    for structure in payload["structures"]:
        _draw_polygon(ax, structure)
    for source in payload["sources"]:
        _draw_source(ax, source)
    for monitor in payload["monitors"]:
        _draw_monitor(ax, monitor)

    ax.set_title(title)
    ax.set_xlim(*payload["xlim"])
    ax.set_ylim(*payload["ylim"])
    ax.set_aspect("equal")
    _configure_axes(ax, payload)
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax


def plot_simulation(
    sim,
    *,
    ax=None,
    figsize=None,
    show=True,
    title="Simulation Layout",
):
    """Plot a simulation layout with sources, monitors, and boundaries."""
    payload = sim.to_plot_data()
    design_payload = payload["design"]
    if figsize is None:
        width = max(float(design_payload["width"]), 1e-30)
        figsize = (6.0, 6.0 * float(design_payload["height"]) / width)
    fig, ax = _figure_axes(ax, figsize=figsize)

    for structure in design_payload["structures"]:
        _draw_polygon(ax, structure)
    for source in design_payload["sources"]:
        _draw_source(ax, source)
    for monitor in design_payload["monitors"]:
        _draw_monitor(ax, monitor)
    _draw_boundaries(ax, payload)

    ax.set_title(title)
    ax.set_xlim(*design_payload["xlim"])
    ax.set_ylim(*design_payload["ylim"])
    ax.set_aspect("equal")
    _configure_axes(ax, design_payload)
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax


def plot_grid(
    grid,
    *,
    field="permittivity",
    z_index=None,
    z_position=None,
    ax=None,
    figsize=None,
    cmap="Grays",
    show=True,
    colorbar=True,
):
    """Plot a rasterized grid field or 3D grid slice."""
    payload = grid.to_plot_data(field=field, z_index=z_index, z_position=z_position)
    design = payload["design"]
    if figsize is None:
        figsize = (6.0, 6.0 * design["height"] / max(design["width"], 1e-30))
    fig, ax = _figure_axes(ax, figsize=figsize)
    im = ax.imshow(
        payload["array"],
        origin="lower",
        cmap=cmap,
        extent=payload["extent"],
    )
    if colorbar:
        fig.colorbar(im, ax=ax, label=field)
    ax.set_title("Rasterized Design Grid")
    scale_factor, scale_unit = get_si_scale_and_label(
        max(design["width"], design["height"], design.get("depth", 0.0))
    )
    _configure_axes(
        ax,
        {
            "scale_factor": scale_factor,
            "scale_unit": scale_unit,
        },
    )
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax


def plot_signal(signals, t, *, ax=None, figsize=(9, 4), show=True, save_path=None):
    """Plot one or more time-domain source signals."""
    from beamz.visual.data import signal_plot_data

    payload = signal_plot_data(signals, t)
    fig, ax = _figure_axes(ax, figsize=figsize)
    for idx, values in enumerate(payload["signals"]):
        kwargs = {"label": f"Signal {idx}"} if len(payload["signals"]) > 1 else {}
        ax.plot(payload["t_scaled"], values, **kwargs)
    ax.set_xlim(*payload["xlim"])
    ax.set_xlabel(f"Time ({payload['time_unit']})")
    ax.set_ylabel("Amplitude")
    ax.set_title("Signal")
    if len(payload["signals"]) > 1:
        ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    _maybe_show(fig, show=show)
    return fig, ax


def plot_mode_profile(
    source,
    *,
    field=None,
    ax=None,
    figsize=(8, 6),
    show=True,
    save_path=None,
):
    """Plot a ModeSource profile."""
    payload = source.mode_profile_data(field=field)
    fig, ax = _figure_axes(ax, figsize=figsize)
    if payload["is_2d"]:
        im = ax.imshow(
            payload["amplitude"],
            origin="lower",
            cmap="magma",
            aspect="auto",
        )
        fig.colorbar(im, ax=ax, label="Absolute Amplitude")
        ax.set_title(
            f"Mode Source 2D Profile: {payload['title']} "
            f"(neff={payload['neff']:.4f})"
        )
        if payload["direction"] in ["+x", "-x"]:
            ax.set_xlabel("Y-axis")
            ax.set_ylabel("Z-axis")
        else:
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Z-axis")
    else:
        ax.plot(payload["amplitude"], "k-")
        ax.set_title(
            f"Mode Source 1D Profile: {payload['title']} "
            f"(neff={payload['neff']:.4f})"
        )
        ax.set_xlabel("Transverse Coordinate (cells)")
        ax.set_ylabel("Absolute Amplitude")
        ax.grid(True)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    _maybe_show(fig, show=show)
    return fig, ax


def plot_monitor_field(
    monitor,
    *,
    field="Ez",
    time_index=-1,
    ax=None,
    figsize=(10, 6),
    cmap="RdBu",
    show=True,
):
    """Plot recorded field data from a monitor."""
    payload = monitor.field_plot_data(field=field, time_index=time_index)
    fig, ax = _figure_axes(ax, figsize=figsize)
    if payload["monitor_type"] == "line":
        ax.plot(payload["x"], np.ravel(payload["array"]), "b-", linewidth=2)
        ax.set_xlabel("Position along monitor line")
        ax.set_ylabel(f"{field} amplitude")
        ax.grid(True, alpha=0.3)
    else:
        im = ax.imshow(payload["array"], cmap=cmap, origin="lower", aspect="auto")
        fig.colorbar(im, ax=ax, label=f"{field} amplitude")
        ax.set_xlabel("X index")
        ax.set_ylabel("Y index")
    ax.set_title(payload["title"])
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax


def plot_monitor_power(
    monitor,
    *,
    log_scale=False,
    db_scale=False,
    ax=None,
    figsize=(10, 6),
    show=True,
):
    """Plot monitor power history."""
    payload = monitor.power_plot_data(log_scale=log_scale, db_scale=db_scale)
    fig, ax = _figure_axes(ax, figsize=figsize)
    ax.plot(payload["time_steps"], payload["power"], "r-", linewidth=2)
    ax.set_yscale(payload["yscale"])
    ax.set_xlabel("Time step")
    ax.set_ylabel(payload["ylabel"])
    ax.set_title(payload["title"])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax


def animate_monitor_fields(
    monitor,
    *,
    field="Ez",
    figsize=(8, 6),
    interval=100,
    save_filename=None,
    show=True,
):
    """Create a matplotlib animation from recorded monitor fields."""
    if (
        not monitor.fields["t"]
        or field not in monitor.fields
        or not monitor.fields[field]
    ):
        raise RuntimeError(f"No data available for field '{field}'.")

    plt = _pyplot()
    FuncAnimation = _mpl_types()["FuncAnimation"]
    fig, ax = plt.subplots(figsize=figsize)

    if monitor.monitor_type == "line":
        (line,) = ax.plot([], [], "b-", linewidth=2)
        ax.set_xlabel("Position along monitor line")
        ax.set_ylabel(f"{field} amplitude")
        all_data = np.concatenate([np.ravel(v) for v in monitor.fields[field]])
        ax.set_xlim(0, len(np.ravel(monitor.fields[field][0])))
        ax.set_ylim(np.min(all_data), np.max(all_data))

        def update(frame):
            field_data = np.ravel(monitor.fields[field][frame])
            line.set_data(range(field_data.size), field_data)
            ax.set_title(f'{field} at t = {monitor.fields["t"][frame]:.2e} s')
            return (line,)

        artists = True
    else:
        field_data = monitor.fields[field][0]
        im = ax.imshow(
            field_data, cmap="RdBu", origin="lower", aspect="auto", animated=True
        )
        fig.colorbar(im, ax=ax, label=f"{field} amplitude")
        ax.set_xlabel("X index")
        ax.set_ylabel("Y index")
        all_data = np.asarray(monitor.fields[field])
        im.set_clim(np.min(all_data), np.max(all_data))

        def update(frame):
            im.set_array(monitor.fields[field][frame])
            ax.set_title(f'{field} at t = {monitor.fields["t"][frame]:.2e} s')
            return (im,)

        artists = True

    anim = FuncAnimation(
        fig,
        update,
        frames=len(monitor.fields["t"]),
        interval=interval,
        blit=artists,
        repeat=True,
    )
    if save_filename:
        anim.save(save_filename, writer="pillow", fps=max(1, 1000 // interval))
    _maybe_show(fig, show=show)
    return anim


def _draw_scale_bar(ax, design_payload, wavelength=None, fontsize=10):
    width = design_payload["width"]
    height = design_payload["height"]
    scale_factor = design_payload["scale_factor"]
    unit = design_payload["scale_unit"]

    if wavelength is not None:
        scale_bar_length_um = np.round(2 * wavelength * 1e6)
        scale_bar_length = scale_bar_length_um * 1e-6
        label_text = f"{int(scale_bar_length_um)} µm"
    else:
        min_dim = min(width, height)
        scale_bar_length_physical = min_dim * 0.18
        if scale_bar_length_physical > 0:
            order = 10 ** np.floor(np.log10(scale_bar_length_physical))
            normalized = scale_bar_length_physical / order
            if normalized <= 1.25:
                nice_value = 1 * order
            elif normalized <= 2.5:
                nice_value = 2 * order
            elif normalized <= 6:
                nice_value = 5 * order
            else:
                nice_value = 10 * order
            scale_bar_length = nice_value
        else:
            scale_bar_length = min_dim * 0.15

        display_value = scale_bar_length * scale_factor
        if display_value >= 1:
            label_text = f"{display_value:.0f} {unit}"
        elif display_value >= 0.1:
            label_text = f"{display_value:.1f} {unit}"
        else:
            label_text = f"{display_value:.2f} {unit}"

    margin_x = width * 0.1
    margin_y = height * 0.1
    x_start = width - scale_bar_length - margin_x
    x_end = width - margin_x
    y_pos = margin_y
    ax.plot([x_start, x_end], [y_pos, y_pos], "w", linewidth=3, solid_capstyle="butt")
    ax.text(
        (x_start + x_end) / 2,
        y_pos - height * 0.02,
        label_text,
        ha="center",
        va="top",
        color="white",
        fontsize=fontsize,
    )


def _snapshot_color_limits(snapshots):
    if not snapshots:
        return None, None

    data_min = np.inf
    data_max = -np.inf
    has_pos = False
    has_neg = False
    for snapshot in snapshots:
        field = np.asarray(snapshot["field"], dtype=np.float64)
        finite = field[np.isfinite(field)]
        if finite.size == 0:
            continue
        data_min = min(data_min, float(np.min(finite)))
        data_max = max(data_max, float(np.max(finite)))
        has_pos = has_pos or bool(np.any(finite > 0.0))
        has_neg = has_neg or bool(np.any(finite < 0.0))

    if not np.isfinite(data_min) or not np.isfinite(data_max):
        return None, None
    if has_pos and has_neg:
        vmax = max(abs(data_min), abs(data_max), 1e-12)
        return -vmax, vmax
    if data_min == data_max:
        pad = max(abs(data_max) * 1e-12, 1e-12)
        return data_min - pad, data_max + pad
    return data_min, data_max


def _snapshot_figsize(snapshot, *, clean_visualization, base_long_edge=10.0):
    if not clean_visualization:
        return (10.0, 8.0)
    extent = snapshot["extent"]
    width = max(float(extent[1]) - float(extent[0]), 1e-12)
    height = max(float(extent[3]) - float(extent[2]), 1e-12)
    if width >= height:
        return (base_long_edge, base_long_edge * (height / width))
    return (base_long_edge * (width / height), base_long_edge)


def snapshot_figure(
    snapshot,
    *,
    cmap="twilight_zero",
    clean_visualization=False,
    interpolation="bicubic",
    figure=None,
    axes=None,
    vmin=None,
    vmax=None,
):
    """Render a simulation snapshot payload."""
    plt = _pyplot()
    layout = snapshot["layout"]
    design_payload = layout["design"]
    field = snapshot["field"]
    extent = snapshot["extent"]
    figsize = _snapshot_figsize(snapshot, clean_visualization=clean_visualization)

    if figure is None or axes is None:
        if clean_visualization:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0, 0, 1, 1])
        else:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = figure
        fig.clear()
        fig.set_size_inches(*figsize, forward=True)
        ax = fig.add_axes([0, 0, 1, 1]) if clean_visualization else fig.add_subplot(111)

    im = ax.imshow(
        field,
        origin="lower",
        cmap=resolve_cmap(cmap),
        extent=extent,
        interpolation=interpolation,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlim(float(extent[0]), float(extent[1]))
    ax.set_ylim(float(extent[2]), float(extent[3]))
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.0)

    for structure in design_payload["structures"]:
        style = dict(structure["style"])
        if not structure.get("is_pml"):
            style["facecolor"] = "none"
            style["edgecolor"] = "gray"
            style["alpha"] = 0.5
        overlay = dict(structure)
        overlay["style"] = style
        _draw_polygon(ax, overlay)
    for source in design_payload["sources"]:
        _draw_source(ax, source)
    for monitor in design_payload["monitors"]:
        overlay = dict(monitor)
        style = dict(overlay["style"])
        style["edgecolor"] = "gray"
        style["alpha"] = 0.5
        overlay["style"] = style
        _draw_monitor(ax, overlay)
    _draw_boundaries(ax, layout)

    if clean_visualization:
        ax.set_axis_off()
        _draw_scale_bar(ax, design_payload)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    else:
        fig.colorbar(
            im,
            ax=ax,
            orientation="vertical",
            label=f"{snapshot['field_name']} ({snapshot['units']})",
        )
        ax.set_title(
            f"{snapshot['field_name']} at t = {snapshot['time']:.2e} s "
            f"(step {snapshot['step']}/{snapshot['num_steps']})"
        )
        _configure_axes(ax, design_payload)
        fig.tight_layout()
    return fig, ax


def show_snapshots(
    snapshots,
    *,
    cmap="twilight_zero",
    clean_visualization=False,
    interpolation="bicubic",
    pause=0.001,
    show=True,
):
    """Show a sequence of stored snapshot payloads."""
    if not snapshots:
        return None, None
    plt = _pyplot()
    context = {"fig": None, "ax": None}
    for snapshot in snapshots:
        fig, ax = snapshot_figure(
            snapshot,
            cmap=cmap,
            clean_visualization=clean_visualization,
            interpolation=interpolation,
            figure=context["fig"],
            axes=context["ax"],
        )
        context["fig"], context["ax"] = fig, ax
        if show:
            plt.show(block=False)
            plt.pause(pause)
    return context["fig"], context["ax"]


def save_snapshot_video(
    snapshots,
    *,
    filename,
    fps=30,
    dpi=150,
    cmap="twilight_zero",
    clean_visualization=False,
    interpolation="bicubic",
):
    """Save stored simulation snapshots to a video file."""
    if not snapshots:
        return None

    plt = _pyplot()
    FFMpegWriter = _mpl_types()["FFMpegWriter"]
    output = Path(filename)
    fig, ax = plt.subplots(
        figsize=_snapshot_figsize(snapshots[0], clean_visualization=clean_visualization)
    )
    vmin, vmax = _snapshot_color_limits(snapshots)
    writer = FFMpegWriter(fps=fps, bitrate=5000)
    with writer.saving(fig, str(output), dpi=dpi):
        for snapshot in snapshots:
            snapshot_figure(
                snapshot,
                cmap=cmap,
                clean_visualization=clean_visualization,
                interpolation=interpolation,
                figure=fig,
                axes=ax,
                vmin=vmin,
                vmax=vmax,
            )
            writer.grab_frame()
    plt.close(fig)
    return output


__all__ = [
    "animate_monitor_fields",
    "get_twilight_zero_cmap",
    "plot_design",
    "plot_grid",
    "plot_mode_profile",
    "plot_monitor_field",
    "plot_monitor_power",
    "plot_signal",
    "plot_simulation",
    "resolve_cmap",
    "save_snapshot_video",
    "show_snapshots",
    "snapshot_figure",
]
