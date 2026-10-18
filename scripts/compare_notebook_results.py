"""Compare exports from validate_modal_notebook.py, preserving complex fields.

Reports every difference without classifying intentional changes as regressions.
Use --require-close to fail when any array exceeds the specified tolerance.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def compare(reference: Path, candidate: Path, *, rtol: float, atol_scale: float):
    metadata = [
        json.loads((directory / "results.json").read_text())
        for directory in (reference, candidate)
    ]
    for key in ("grid_shape", "steps", "nfreqs", "metric_kind"):
        if metadata[0][key] != metadata[1][key]:
            raise ValueError(f"Mismatched {key}: {candidate}")
    errors = {}
    with (
        np.load(reference / "arrays.npz") as expected,
        np.load(candidate / "arrays.npz") as actual,
    ):
        if set(expected.files) != set(actual.files):
            raise ValueError(f"Mismatched exported arrays: {candidate}")
        for name in expected.files:
            dtype = (
                np.complex128
                if np.iscomplexobj(expected[name]) or np.iscomplexobj(actual[name])
                else np.float64
            )
            x, y = expected[name].astype(dtype), actual[name].astype(dtype)
            if x.shape != y.shape:
                raise ValueError(f"Mismatched shape for {name}: {candidate}")
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f"Non-finite {name}: {candidate}")
            scale = float(np.max(np.abs(x), initial=0))
            atol = max(1e-12, atol_scale * scale)
            errors[name] = {
                "shape": list(x.shape),
                "array_equal": bool(np.array_equal(x, y)),
                "relative_l2": float(
                    np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-300)
                ),
                "max_abs": float(np.max(np.abs(x - y), initial=0)),
                "max_reference": scale,
                "close": bool(np.allclose(x, y, rtol=rtol, atol=atol)),
            }
    return {
        "reference": reference.name,
        "candidate": candidate.name,
        "arrays": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=3e-5)
    parser.add_argument("--atol-scale", type=float, default=2e-6)
    parser.add_argument("--require-close", action="store_true")
    args = parser.parse_args()
    if args.rtol < 0 or args.atol_scale < 0:
        parser.error("Tolerances must be nonnegative")
    comparisons = [
        compare(
            args.reference,
            candidate,
            rtol=args.rtol,
            atol_scale=args.atol_scale,
        )
        for candidate in args.candidates
    ]
    result = {
        "rtol": args.rtol,
        "atol_scale": args.atol_scale,
        "absolute_floor": 1e-12,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    passed = all(
        value["close"] for row in comparisons for value in row["arrays"].values()
    )
    print("All arrays within tolerance" if passed else "Differences found; see output")
    if args.require_close and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
