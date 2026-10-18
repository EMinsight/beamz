"""Modal reference powers must not inherit sampled-flux attenuation."""

import numpy as np
import pytest
import xarray as xr

from beamz.analysis.adapters import ModeMonitorData


def modal_data(amps, flux, reconstructed, *, freqs=(1.0, 2.0)):
    return ModeMonitorData(
        monitor=None,
        amps=xr.DataArray(
            np.asarray(amps, dtype=complex).reshape(2, 2, 1),
            dims=("f", "direction", "mode_index"),
            coords={"f": list(freqs), "direction": ["+", "-"], "mode_index": [0]},
        ),
        flux=xr.DataArray(flux, dims="f", coords={"f": list(freqs)}),
        modal_flux=xr.DataArray(reconstructed, dims="f", coords={"f": list(freqs)}),
        projection_residual=np.array([0.01, 0.02]),
        condition_number=np.array([1.1, 1.2]),
    )


@pytest.mark.parametrize("input_power", (0.25, 1.0, 4.0))
def test_shared_modal_reference_preserves_phases_and_sampled_accounting(input_power):
    incident = np.array([input_power, 2 * input_power])
    phases = np.exp(1j * np.array([0.3, -0.7]))
    reference = modal_data(
        np.column_stack((np.sqrt(incident) * phases, np.zeros(2))),
        0.95 * incident,
        0.94 * incident,
    )
    target = modal_data(
        np.column_stack((np.sqrt(0.6 * incident) * phases, np.sqrt(0.1 * incident))),
        0.48 * incident,
        0.47 * incident,
    )
    normalized = target.normalized_to(reference)
    np.testing.assert_allclose(normalized.power.sel(direction="+"), 0.6)
    np.testing.assert_allclose(normalized.power.sel(direction="-"), 0.1)
    np.testing.assert_allclose(normalized.flux, 0.48)
    np.testing.assert_allclose(normalized.modal_flux, 0.47)
    np.testing.assert_allclose(normalized.unresolved_flux, 0.01)
    np.testing.assert_allclose(
        normalized.amps.sel(direction="+").values[:, 0], np.sqrt(0.6) * phases
    )
    np.testing.assert_array_equal(
        normalized.projection_residual, target.projection_residual
    )
    np.testing.assert_allclose(target.flux, 0.48 * incident)
    assert normalized.power.attrs["units"] == "W"
    assert not normalized.amps.values.flags.writeable


def test_reference_normalization_scales_all_powers_without_clipping():
    reference = modal_data([1, 0, 2, 0], [0.9, 3.6], [0.9, 3.6])
    result = modal_data([2, 0, 4, 0], [3.6, 14.4], [3.6, 14.4])
    np.testing.assert_allclose(
        result.normalized_to(reference, power=2).power.sel(direction="+"), 8
    )
    np.testing.assert_allclose(
        reference.normalized_to(reference).power.sel(direction="+"), 1
    )


def test_reference_rejects_wrong_frequency_order_and_zero_incident_power():
    data = modal_data([1, 0, 1, 0], [0.9, 0.9], [0.9, 0.9])
    shifted = modal_data([1, 0, 1, 0], [0.9, 0.9], [0.9, 0.9], freqs=(2.0, 1.0))
    with pytest.raises(ValueError, match="frequencies"):
        data.normalized_to(shifted)
    with pytest.raises(ValueError, match="Incident"):
        data.normalized_to(data, direction="-")
    with pytest.raises(TypeError, match="ModeMonitorData"):
        data.normalized_to(data.flux)
