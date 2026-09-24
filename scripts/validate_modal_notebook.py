"""Execute the full modal tutorial and export unnormalized numerical results.

Run once per checkout in a fresh process with that checkout on PYTHONPATH.
The notebook cells are unchanged; a final cell exports arrays for comparison.
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
        BEAMZ_EXECUTION_BACKEND="cuda_streamed",
    )
    path = root / "examples/notebooks/modal_sources_monitors.ipynb"
    notebook = nbformat.read(path, as_version=4)
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            """
import json
import beamz._cuda as extension
assert not test_mode
assert Path(bz.__file__).resolve().parent.parent == Path(ROOT)
assert Path(extension.__file__).resolve().parent.parent == Path(ROOT)
arrays = {}
for label, data in (
    ("single", sim_data_single), ("broadband", sim_data_bb),
    ("junction", sim_data_jct_bb),
):
    raw = data.renormalize(None)
    arrays[label + "_flux"] = np.asarray(raw["flux"].flux)
    arrays[label + "_amps"] = np.asarray(raw.mode("mode").amps)
    arrays[label + "_mode_flux"] = np.asarray(raw.mode("mode").flux)
    arrays[label + "_ey"] = np.asarray(raw["field"].dft_fields["Ey"])
arrays["neffs"] = np.asarray(modes.neffs)
arrays["profile_freqs"] = np.asarray(profile_freqs)
arrays["freqs"] = np.asarray(freqs)
for name, value in arrays.items():
    assert np.isfinite(value).all(), name
np.savez_compressed(Path(OUTPUT) / "arrays.npz", **arrays)
metadata = {
    "backend": sim_single.compile().config.backend,
    "grid_shape": list(sim0.grid.shape), "steps": sim0.num_steps,
    "nfreqs": nfreqs, "broadband_profiles": broadband_profile_count,
    "extension_version": extension.__version__,
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
        nbformat.write(notebook, output / "modal_sources_monitors.ipynb")
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
