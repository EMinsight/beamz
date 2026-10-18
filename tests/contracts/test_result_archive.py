"""Result archives preserve scientific data and reject unknown formats/types."""

import json

import numpy as np
import pytest

from beamz.simulation.results import SimulationResults


@pytest.mark.parametrize(
    "manifest, message",
    [
        ({"format": "unknown", "result": None}, "version"),
        (
            {
                "format": "beamz.simulation.results/1",
                "result": {"class": "UnknownClass", "fields": {}},
            },
            "class",
        ),
        ({"format": "beamz.simulation.results/1", "result": None}, "SimulationResults"),
    ],
)
def test_result_archive_rejects_unknown_records(tmp_path, manifest, message):
    path = tmp_path / "bad.npz"
    np.savez(path, manifest=np.asarray(json.dumps(manifest)))
    with pytest.raises(ValueError, match=message):
        SimulationResults.load(path)
