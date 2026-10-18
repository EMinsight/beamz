"""Independent MPB reference; run with a Python that provides meep.mpb.

Units are micrometres. The transverse periodic supercell is widened separately
from resolution refinement to check isolated-guide convergence.
"""

import argparse
import json

import meep as mp
from meep import mpb

parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, default=30)
parser.add_argument("--width", type=float, default=4.0)
parser.add_argument("--output", required=True)
args = parser.parse_args()
mp.verbosity(0)
solver = mpb.ModeSolver(
    geometry_lattice=mp.Lattice(size=mp.Vector3(0, args.width, args.width)),
    geometry=[
        mp.Block(
            size=mp.Vector3(mp.inf, 0.6, 0.6),
            material=mp.Medium(
                epsilon_diag=mp.Vector3(6.0, 5.0, 4.0),
                epsilon_offdiag=mp.Vector3(0.2, 0.1, 0.3),
            ),
        )
    ],
    resolution=args.resolution,
    num_bands=3,
    tolerance=1e-10,
)
frequency = 1 / 1.55
k = solver.find_k(
    mp.NO_PARITY,
    frequency,
    1,
    3,
    mp.Vector3(1, 0, 0),
    1e-8,
    2.0 * frequency,
    1.01 * frequency,
    2.5 * frequency,
)
result = {
    "meep_version": mp.__version__,
    "resolution_per_um": args.resolution,
    "width_um": args.width,
    "frequency_per_um": frequency,
    "neff": [float(value / frequency) for value in k],
}
with open(args.output, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result))
