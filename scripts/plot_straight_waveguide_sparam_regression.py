from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HELPER_PATH = ROOT / "tests" / "modal_sparam_physical_case.py"
HELPER_SPEC = importlib.util.spec_from_file_location(
    "_modal_sparam_physical_case", HELPER_PATH
)
HELPER = importlib.util.module_from_spec(HELPER_SPEC)
sys.modules["_modal_sparam_physical_case"] = HELPER
assert HELPER_SPEC.loader is not None
HELPER_SPEC.loader.exec_module(HELPER)

StraightWaveguideSParamConfig = HELPER.StraightWaveguideSParamConfig
StraightWaveguideSParamResult = HELPER.StraightWaveguideSParamResult
run_straight_waveguide_sparam_case = HELPER.run_straight_waveguide_sparam_case
summarize_center_metrics = HELPER.summarize_center_metrics


def _db(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(values), 1e-12))


def _result_summary(result: StraightWaveguideSParamResult) -> dict:
    center = summarize_center_metrics(result)
    return {
        "resolution_ppw": result.resolution_ppw,
        "dx_nm": result.dx_nm,
        "dt_fs": result.dt_fs,
        "steps_to_decay": result.steps_to_decay,
        "runtime_s": result.runtime_s,
        "monitor_x_um": result.monitor_x_um,
        "center_metrics": center,
        "phase_residual_rad_by_monitor": result.phase_residual_rad_by_monitor,
        "phase_slope_s_by_monitor": result.phase_slope_s_by_monitor,
        "max_condition_number": {
            name: float(np.nanmax(values))
            for name, values in result.condition_numbers.items()
        },
    }


def plot_results(
    results: list[StraightWaveguideSParamResult],
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), dpi=180)
    ax_mag, ax_power, ax_phase, ax_cond = axes.ravel()

    colors = {10: "#1f77b4", 12: "#d62728", 14: "#2ca02c"}
    for result in results:
        wl = np.asarray(result.wavelengths_um, dtype=float)
        color = colors.get(result.resolution_ppw, None)
        ax_mag.plot(
            wl,
            _db(result.s11),
            marker="o",
            color=color,
            linestyle="--",
            label=f"S11 ppw{result.resolution_ppw}",
        )
        for monitor_name, linestyle in (("mid", "-"), ("far", ":")):
            ax_mag.plot(
                wl,
                _db(result.s21_by_monitor[monitor_name]),
                marker="o",
                color=color,
                linestyle=linestyle,
                label=f"S21 {monitor_name} ppw{result.resolution_ppw}",
            )
            ax_power.plot(
                wl,
                result.power_sum_by_monitor[monitor_name],
                marker="o",
                color=color,
                linestyle=linestyle,
                label=f"{monitor_name} ppw{result.resolution_ppw}",
            )

        denom = np.where(
            np.abs(result.s21_by_monitor["mid"]) > 1e-30,
            result.s21_by_monitor["mid"],
            1e-30 + 0.0j,
        )
        ratio = result.s21_by_monitor["far"] / denom
        phase = np.unwrap(np.angle(ratio))
        f = np.asarray(result.frequencies_hz, dtype=float)
        slope, intercept = np.polyfit(f - float(np.mean(f)), phase, 1)
        fit = slope * (f - float(np.mean(f))) + intercept
        ax_phase.plot(
            wl,
            phase,
            marker="o",
            color=color,
            label=f"far/mid ppw{result.resolution_ppw}",
        )
        ax_phase.plot(wl, fit, color=color, alpha=0.45, linewidth=1.0)

        for monitor_name, linestyle in (("o1", "--"), ("mid", "-"), ("far", ":")):
            ax_cond.plot(
                wl,
                result.condition_numbers[monitor_name],
                color=color,
                linestyle=linestyle,
                alpha=0.85,
                label=f"{monitor_name} ppw{result.resolution_ppw}",
            )

    ax_mag.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax_mag.axhline(-30.0, color="black", linewidth=0.8, alpha=0.25)
    ax_mag.set_title("Modal S-parameters")
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.set_xlabel("Wavelength (um)")
    ax_mag.grid(alpha=0.25)
    ax_mag.legend(fontsize=7, ncol=2)

    ax_power.axhline(1.0, color="black", linewidth=0.8, alpha=0.35)
    ax_power.axhspan(0.97, 1.03, color="green", alpha=0.08)
    ax_power.set_title("Per-output guided power closure")
    ax_power.set_ylabel("|S11|^2 + |S21|^2")
    ax_power.set_xlabel("Wavelength (um)")
    ax_power.grid(alpha=0.25)
    ax_power.legend(fontsize=7, ncol=2)

    ax_phase.set_title("Downstream phase consistency")
    ax_phase.set_ylabel("Unwrapped phase of S21_far / S21_mid (rad)")
    ax_phase.set_xlabel("Wavelength (um)")
    ax_phase.grid(alpha=0.25)
    ax_phase.legend(fontsize=7)

    ax_cond.set_title("Projection conditioning")
    ax_cond.set_ylabel("Condition number")
    ax_cond.set_xlabel("Wavelength (um)")
    ax_cond.set_yscale("log")
    ax_cond.grid(alpha=0.25, which="both")
    ax_cond.legend(fontsize=6, ncol=2)

    fig.suptitle("Straight Waveguide Physical Modal S-parameter Regression", y=0.99)
    fig.tight_layout()
    png_path = output_dir / "straight_waveguide_sparam_regression.png"
    fig.savefig(png_path)
    plt.close(fig)

    payload = {
        "config": asdict(StraightWaveguideSParamConfig()),
        "results": [_result_summary(result) for result in results],
    }
    json_path = output_dir / "straight_waveguide_sparam_regression.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return png_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the straight-waveguide physical modal S-parameter regression."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/straight_waveguide_sparam_regression"),
    )
    parser.add_argument(
        "--ppw",
        type=int,
        nargs="+",
        default=[10, 12],
        help="Resolution points per wavelength values to run.",
    )
    args = parser.parse_args()

    cfg = StraightWaveguideSParamConfig()
    results = [
        run_straight_waveguide_sparam_case(resolution_ppw=ppw, cfg=cfg)
        for ppw in args.ppw
    ]
    png_path, json_path = plot_results(results, output_dir=args.output_dir)
    print(f"plot={png_path}")
    print(f"summary={json_path}")
    for result in results:
        metrics = summarize_center_metrics(result)
        print(
            "ppw={ppw} dx={dx:.2f}nm S11={s11:.2f}dB "
            "S21_mid={s21_mid:.4f}dB S21_far={s21_far:.4f}dB "
            "P_mid={p_mid:.5f} P_far={p_far:.5f}".format(
                ppw=result.resolution_ppw,
                dx=result.dx_nm,
                s11=metrics["s11_db"],
                s21_mid=metrics["s21_mid_db"],
                s21_far=metrics["s21_far_db"],
                p_mid=metrics["power_sum_mid"],
                p_far=metrics["power_sum_far"],
            )
        )


if __name__ == "__main__":
    main()
