# Bundled optical data

The four `.yml` files are unmodified snapshots of the public-domain
[refractiveindex.info database](https://github.com/polyanskiy/refractiveindex.info-database)
at revision `c5c2f188e848453def5970e347399d653df2ffc2`, distributed under
CC0 1.0 (see `LICENSE-CC0-1.0.txt`). The catalog records each source URL,
SHA-256 checksum, original references, and available sample conditions.
Please cite the original experimental papers and M. N. Polyanskiy,
*Scientific Data* **11**, 94 (2024), https://doi.org/10.1038/s41597-023-02898-2.

`python scripts/build_material_library.py` rebuilds the catalog offline using
the development dependencies. All models declare a 400–700 nm use band.
Silica and silicon nitride use exact conversions of the source Sellmeier
formulas. Amorphous silicon (a 60 nm film) and aluminum use independently
computed positive Lorentz fits, with three and four oscillators respectively.
The fitting grid has 151 linearly interpolated n,k samples; these are not new
measurements. CSVs retain the source samples within the use band plus
interpolated endpoints. Formula CSVs are sampled analytic values. All CSV
wavelengths are in **metres**. Source YAML wavelengths are in micrometres.

The catalog reports RMS n,k errors on the interpolation grid and maximum
absolute errors at the retained source samples/endpoints. These metrics
measure optical-constant fitting, not simulation convergence. The underlying
measurements cover wider bands; that does not extend the declared fit band.
Material preparation and measurement conditions matter when choosing a variant.

The RGB filters are synthetic design targets, **not measured data**. Targets
are generated analytically with n = 1.45 and k = 0.01 in the passband, k = 0.46
in the stopband, and raised-cosine transitions over 30 nm. The red transition
is 600–630 nm, the blue transition is 470–500 nm, and green passes 500–600 nm
with those two transitions on either side. CSVs contain these targets, not the
fitted response. No copied filter tables or fitted coefficients are used.

The builder fits 12 stable conjugate pole pairs per filter using constrained
nonlinear least squares (SciPy SLSQP). It first fits residues with fixed poles,
then refines the poles. The objective weights index, extinction, and full-spectrum
absorption-only transmission by 0.05, 1, and 5 respectively in the final stage.
Transmission is evaluated through a 1 µm thickness. Sampled passivity constraints
cover frequencies beyond the use band; a separate dense audit adds missed
negative-loss minima to the constraints before accepting a model. This is a
numerical passivity check, not a global mathematical certificate.

Acceptance requires convergence, the passivity audit, and at most 0.02 absolute
full-spectrum transmission error on a 3001-point validation grid. The constant-index
target of 0.01 error is **not met**: index dispersion is retained and reported
rather than removed. Both target statuses and maximum errors appear in the
catalog. No device-efficiency data enter the fitting objective.

The generator, synthetic targets, and synthetic models use the project's
Apache-2.0 license. Physical source data retain CC0.
