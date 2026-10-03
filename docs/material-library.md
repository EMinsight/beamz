# Material library

BeamZ bundles named dispersive materials that work offline and can be used
directly in structures and simulations.

```python
import beamz as bz
from beamz import material_library

silica = material_library["SiO2"]["Malitson1965"]
aluminum = material_library["Al"].medium  # default: Rakic1995
structure = bz.Box(center=(0, 0, 0), size=(1e-6, 1e-6, 40e-9), material=aluminum)
```

Lookups return immutable `PoleResidue` materials. `frequency_range` is in Hz;
`eps_model(frequency)` evaluates complex relative permittivity. Every bundled
model currently declares **400–700 nm**, even when its source covers a wider band.

| Material | Variant | Construction |
| --- | --- | --- |
| `SiO2` (fused silica) | `Malitson1965` | Exact Sellmeier conversion |
| `SiN` (silicon nitride) | `Philipp1973` | Exact Sellmeier conversion |
| `aSi` (amorphous silicon, 60 nm film) | `Pierce1972` | Three-oscillator passive fit |
| `Al` (aluminum) | `Rakic1995` | Four-oscillator passive fit |
| `CMOS_RGB` | `red`, `green`, `blue` | Passive fits to analytic filter targets |

The first four materials use CC0 data from the
[refractiveindex.info database](https://github.com/polyanskiy/refractiveindex.info-database).
The three filters are original hypothetical materials, not measured products.

## Inspect data and fit quality

```python
variant = material_library["aSi"].variants["Pierce1972"]
print(variant.description, variant.source, variant.license)
print(variant.conditions, variant.references)
print(variant.fit)
wavelength_m, n, k = variant.nk_data.T
```

`nk_data` returns a fresh array with wavelength [m], n, k columns. For tabulated
physical sources, it retains original samples inside the use band plus
interpolated endpoints. For formulas, it samples the analytic definition. For synthetic filters,
it contains the intended n,k targets; evaluate the medium to obtain the fitted
response. Changing the array does not change library data.
Library and variant mappings are read-only.

Silica and nitride require no numerical fit. For silicon and aluminum the
catalog reports RMS n,k errors on a 151-point interpolation grid and maximum
absolute errors at retained source samples/endpoints. Fit error and FDTD
resolution error are separate. Check the band, sample conditions, and error
report before choosing a model. In particular, different forms and deposition
conditions of silicon or nitride are not interchangeable.

## Synthetic filter targets

The filter targets have n = 1.45, passband k = 0.01, and stopband k = 0.46.
Raised-cosine transitions span 470–500 nm and/or 600–630 nm. Red passes above
630 nm, green from 500–600 nm, and blue below 470 nm, within the 400–700 nm band.

Twelve stable pole pairs are fitted per filter, prioritizing extinction and
absorption-only transmission through 1 µm. Each fit must pass an independent
dense passivity audit and keep the maximum full-spectrum transmission error below
0.02 (two percentage points). The catalog records the actual errors.

The constant-index target is not met within 0.01: the fitted media retain index
dispersion, particularly near band edges. `index_target_met` and
`transmission_target_met` report these separate outcomes explicitly. The fits
are not equivalent to a hypothetical constant-index filter. Fresnel reflections
and interference also affect the transmission of a complete film or device.

## Rebuild or extend the catalog

Install the development dependencies and run
`python scripts/build_material_library.py`. This runs on the CPU and uses only
bundled, pinned source snapshots. The catalog retains source URLs, SHA-256
checksums, citations, conditions, fit settings, and diagnostics. Raw YAML files,
CC0 terms, and a data README ship in `beamz/material_library/data`.

To add a variant, select and pin a suitable source, normalize its units, choose
a use band, and convert its analytic formula or fit passive causal oscillators.
Validate optical constants against the source and exercise the model in an
analytical slab test before adding it to the curated catalog. The generator
currently supports Sellmeier formula 1 and tabulated n,k; it rejects other
formats. It does not automatically import the entire upstream database.

These are scalar bulk models for single-device JAX execution. A larger source
catalog alone does not add dispersive anisotropy, surface-conductivity models,
or temperature-dependent constitutive equations. Those require separate solver
and API work. Changing any model requires fresh simulation results; notebook
cache keys include the material coefficients.
