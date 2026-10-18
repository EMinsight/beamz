"""Reproduce the intrinsic-tensor guided-index refinement measurements."""

import argparse
import json
import time
from pathlib import Path

from tests.validation.analytical.test_tensor_port_maxwell import mode_for

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--rectilinear", action="store_true")
args = parser.parse_args()
if args.rectilinear:
    from tests.validation.analytical.test_rectilinear_tensor_modes import (
        rectilinear_mode,
    )

    mode_for = rectilinear_mode

rows = []
for cells in (20, 40, 80) if args.rectilinear else (20, 40, 60, 80):
    start = time.perf_counter()
    mode = mode_for(cells=cells)
    rows.append(
        {
            "cells": cells,
            "spacing_nm": 2000 / cells,
            "neff": mode.neff.real,
            "ampere_residual": mode.diagnostics["tensor_ampere_residual"],
            "seconds": time.perf_counter() - start,
        }
    )
path = Path("benchmarks/results/issue361") / (
    "rectilinear/refinement.json" if args.rectilinear else "beamz-refinement.json"
)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
