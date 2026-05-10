# Modal S-parameter DFT and Yee Phase Convention

BeamZ monitor DFTs use the phasor convention

```text
f(t) = Re{F exp(-i omega t)}
F ~= 2 sum_t w(t) f(t) exp(+i omega t) / sum_t w(t)
```

For a real signal `A cos(omega t + phi)`, the stored complex DFT amplitude is
`A exp(-i phi)`. This is the convention used by modal projection and
S-parameter extraction.

## Temporal Yee Stagger

One BeamZ time step is ordered as:

```text
H update -> source M injection -> E update -> source J injection -> monitor sample
```

When monitors are sampled, the timestamp is `T = t + dt`. The E fields have just
been updated to `T`; the leapfrog H fields are the H half-step at `T - dt/2`.

If a component is physically sampled at `T + tau`, but the DFT accumulator uses
`exp(+i omega T)`, the accumulated phasor is

```text
F_measured = F exp(-i omega tau)
```

so the projection-time phasor is recovered with

```text
F = F_measured exp(+i omega tau)
```

Therefore:

```text
E components: tau = 0       -> multiply by 1
H components: tau = -dt / 2 -> multiply by exp(-i omega dt / 2)
```

This is the only phase applied by the raw monitor component samplers
(`_sample_monitor_component_dft`, FFT spectra, and CW demodulation).

## Modal Plane Reference

ModeSource and modal S-parameter extraction use an H-referenced modal gauge: the
mode profile phase is aligned to the dominant H component. After the temporal
correction above, the sampled E components must still be propagated from their
Yee E plane to that H reference plane before solving for the forward/backward
mode coefficients.

For the dominant tangential E/H pair on a native Yee plane, the separation is one
half grid cell along the port axis. For a port with direction sign `s = +/-1`,
grid spacing `d`, and solved effective index `n_eff`, the 2D delay is

```text
tau_EH = s * (d / 2) * Re(n_eff) / c
```

and E components are multiplied by

```text
exp(+i omega tau_EH)
```

H components are already the projection reference after the temporal half-step,
so they receive no additional spatial modal-plane phase. In 3D, BeamZ uses the
same half-cell separation but converts it through the 1D Yee numerical
dispersion relation, matching the 3D ModeSource launch timing.

The total relative E/H projection phase is therefore not a fixed number of time
steps. It is the sum of the temporal H half-step and the spatial E-to-H modal
plane propagation delay:

```text
relative delay = dt / 2 + tau_EH
```

For example, in the straight-waveguide regression at 10 points per wavelength,
`tau_EH` is about `1.39 dt`; the derived relative correction is therefore about
`1.89 dt`. This explains why the old empirical full-step E/H rotation happened
to look close for that case, without making that rotation a general convention.

## What This Replaces

The older modal extraction path used an empirical full-step E/H phase rotation:
E components were advanced by `exp(+i omega dt)` and H components delayed by
`exp(-i omega dt)`. That made the relative E/H correction two full time steps and
was not tied to the leapfrog sample times or to the solved modal phase velocity.
The derived convention above is now used consistently by DFT monitor extraction,
FFT-sampled monitor extraction, and CW demodulation before modal projection.
