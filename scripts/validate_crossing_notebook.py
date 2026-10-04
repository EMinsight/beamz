"""Execute the full cosine crossing notebook and export regression evidence.

Original notebook cells are unchanged; a final cell saves raw fields and spectra.
Run separately against baseline and candidate checkouts using the same environment.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.pop("BEAMZ_DOCS_TEST", None)
    os.environ.update(
        PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        PYTHONPATH=str(root),
        MPLBACKEND="module://matplotlib_inline.backend_inline",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    path = root / "examples/notebooks/cosine_waveguide_crossing.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            """
import hashlib
import json
import sys
import jax
import beamz._cuda as extension
assert not test_mode
assert len(freqs) == 101
assert Path(bz.__file__).resolve().parent.parent == Path(ROOT)
assert Path(extension.__file__).resolve().parent.parent == Path(ROOT)
assert any(device.platform == "gpu" for device in jax.devices())
raw = sim_data.renormalize(None)
arrays = {
    name: np.asarray(globals()[name])
    for name in (
        "freqs", "ldas", "flux_through", "flux_cross", "source_power",
        "T_through", "T_cross", "transmission_db", "crosstalk_db",
    )
}
arrays["neffs"] = np.asarray(modes.neffs)
for name in ("flux_through", "flux_cross"):
    arrays["raw_" + name] = np.asarray(raw[name].flux)
for component in ("Ex", "Ey", "Ez"):
    arrays["raw_" + component] = np.asarray(raw["field"].dft_fields[component])
for name, value in arrays.items():
    assert np.isfinite(value).all(), name
np.savez_compressed(Path(OUTPUT) / "arrays.npz", **arrays)
metadata = {
    "backend": backend, "grid_shape": list(sim.grid.shape),
    "steps": sim.num_steps, "nfreqs": len(freqs),
    "jax_version": jax.__version__, "python": sys.executable,
    "devices": [device.device_kind for device in jax.devices()],
    "extension_version": extension.__version__,
    "extension_sha256": hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
    "arrays": {name: list(value.shape) for name, value in arrays.items()},
}
(Path(OUTPUT) / "results.json").write_text(json.dumps(metadata, indent=2))
print(metadata)
""".replace("ROOT", repr(str(root))).replace("OUTPUT", repr(str(output)))
        )
    )
    start = time.monotonic()
    client = NotebookClient(
        notebook,
        timeout=3600,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
        on_cell_start=lambda cell, cell_index: print(
            f"cell {cell_index}: {time.monotonic() - start:.1f}s", flush=True
        ),
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output / "cosine_waveguide_crossing.ipynb")
        (output / "provenance.json").write_text(
            json.dumps(
                {
                    "root": str(root),
                    "commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=root, text=True
                    ).strip(),
                    "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "elapsed_s": time.monotonic() - start,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
