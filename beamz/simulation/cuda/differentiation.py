"""Native timestep primals with the exact, existing streamed/JAX transpose.

The linearization substitutes already computed post-source fields. Phase VJPs
need only their inputs; additive sources and field DFTs are linear in fields.
Consequently the unused forward phase launches disappear from the backward HLO.
"""

from dataclasses import dataclass, replace
from typing import cast

import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.typing import DTypeLike

from beamz.devices.sources.compiler import BatchedSlabGroup
from beamz.simulation.cuda.runtime import (
    pack_dft_monitors,
    run_program_steps,
    run_source_group_steps,
)
from beamz.simulation.execute import compiled_source_batches, forward_step


@jax.tree_util.register_static
@dataclass(frozen=True)
class _Columns:
    flags: tuple[bool, ...]


def build_native_step(program, context, update_kernel, split_dft, pack_dft):
    """Return a differentiable native step, or None for unsupported recorders."""
    if context.config.backend != "cuda_streamed" or bool(
        jax.config.read("jax_enable_x64")
    ):
        return None
    batches = compiled_source_batches(program.sources)
    if any(rest for _group, rest in batches.values()):
        return None
    if any(
        mon.recorder_index >= 0
        or mon.accumulate_power
        or mon.accumulate_frequency
        or not mon.dft_enabled
        or mon.freq_count <= 0
        or mon.dft_point_count <= 0
        for mon in program.monitors
    ):
        return None
    keys = tuple(batches)
    templates = tuple(batches[key][0] for key in keys)
    packed_monitors = pack_dft_monitors(program.monitors) if program.monitors else None
    base_context = replace(context, source_batches=None)

    def source_values(ctx):
        return tuple(
            (group.waveforms, group.coeffs, group.starts)
            for key in keys
            if (group := ctx.source_batches[key][0]) is not None
        )

    def restore_context(values):
        iterator = iter(values)
        groups = []
        for template in templates:
            if template is None:
                groups.append(None)
            else:
                waveforms, coeffs, starts = next(iterator)
                groups.append(
                    BatchedSlabGroup(
                        waveforms=waveforms,
                        coeffs=coeffs,
                        starts=starts,
                        n=template.n,
                        max_sizes=template.max_sizes,
                        starts_tuple=template.starts_tuple,
                    )
                )
        ctx = replace(
            base_context,
            source_batches=dict(
                zip(keys, ((group, ()) for group in groups), strict=True)
            ),
        )
        return ctx, tuple(groups)

    def native_forward(inputs):
        state, coeffs, sources, clock = inputs
        ctx, groups = restore_context(sources)
        packed = pack_dft(state)
        out = (
            run_program_steps(
                packed,
                ctx,
                coeffs,
                groups,
                packed_monitors,
                1,
                observation_origin=clock[0],
                observation_step_offset=clock[1],
            )
            if packed_monitors is not None
            else run_source_group_steps(packed, ctx, coeffs, groups, 1)
        )
        out = out._replace(
            t=clock[0] + (clock[1] + 1).astype(jnp.float32) * ctx.dt_scalar,
            current_step=state.current_step + jnp.int32(1),
        )
        return split_dft(out)

    @jax.custom_vjp
    def native_step(inputs):
        return native_forward(inputs)

    def step_forward(primal_inputs):
        values = jax.tree.map(lambda primal: primal.value, primal_inputs)
        flags = jax.tree.map(
            lambda primal: (
                bool(primal.perturbed)
                and jnp.issubdtype(primal.value.dtype, jnp.inexact)
            ),
            primal_inputs,
        )
        state_flags, coeff_flags, _, clock_flags = flags
        clock_active = clock_flags[0]
        # CUDA consumes precomputed decay/source coefficients. Other physical
        # planner leaves have zero columns in this timestep's Jacobian.
        used_coefficients = {
            f"{phase}_{kind}_{axis}"
            for phase in ("e", "h")
            for kind in ("decay", "source")
            for axis in "xyz"
        }
        coeff_flags = coeff_flags._replace(
            **{
                name: False
                for name in coeff_flags._fields
                if name not in used_coefficients
            }
        )
        flags = (state_flags, coeff_flags, flags[2], clock_flags)
        out = native_forward(values)
        old, coeffs, sources, clock = values
        h_source = any(getattr(coeff_flags, f"h_source_{axis}") for axis in "xyz")
        e_source = any(getattr(coeff_flags, f"e_source_{axis}") for axis in "xyz")
        e_decay = any(getattr(coeff_flags, f"e_decay_{axis}") for axis in "xyz")
        saved_old = old._replace(
            **{
                f"e{axis}": getattr(old, f"e{axis}")
                if h_source or e_decay
                else jnp.zeros_like(getattr(old, f"e{axis}"))
                for axis in "xyz"
            },
            **{
                f"h{axis}": getattr(old, f"h{axis}")
                if getattr(coeff_flags, f"h_decay_{axis}")
                else jnp.zeros_like(getattr(old, f"h{axis}"))
                for axis in "xyz"
            },
            cpml_psi_h_terms=old.cpml_psi_h_terms
            if h_source
            else jax.tree.map(jnp.zeros_like, old.cpml_psi_h_terms),
            cpml_psi_e_terms=old.cpml_psi_e_terms
            if e_source
            else jax.tree.map(jnp.zeros_like, old.cpml_psi_e_terms),
            dft_vec_re=jax.tree.map(jnp.zeros_like, old.dft_vec_re),
            dft_vec_im=jax.tree.map(jnp.zeros_like, old.dft_vec_im),
            dft_weight_sum=jax.tree.map(jnp.zeros_like, old.dft_weight_sum),
        )
        # Only material-source VJPs and clock derivatives need post-H values;
        # only clock derivatives need post-E values. Field/DFT Jacobians are
        # linear in these fields and do not need their numeric primals.
        known = jax.tree.map(jnp.zeros_like, out)._replace(
            **{
                f"h{axis}": getattr(out, f"h{axis}")
                if e_source or clock_active
                else jnp.zeros_like(getattr(out, f"h{axis}"))
                for axis in "xyz"
            },
            **{
                f"e{axis}": getattr(out, f"e{axis}")
                if clock_active
                else jnp.zeros_like(getattr(out, f"e{axis}"))
                for axis in "xyz"
            },
        )
        return out, (
            (saved_old, coeffs, sources, clock),
            known,
            _Columns(tuple(jax.tree.leaves(flags))),
        )

    def step_backward(residuals, cotangents):
        inputs, known, columns = residuals
        leaves, tree = jax.tree.flatten(inputs)
        active = tuple(
            value for value, flag in zip(leaves, columns.flags, strict=True) if flag
        )

        def linearize(variables):
            iterator = iter(variables)
            state, coeffs, sources, clock = tree.unflatten(
                [
                    next(iterator) if flag else jax.lax.stop_gradient(value)
                    for value, flag in zip(leaves, columns.flags, strict=True)
                ]
            )
            ctx, _groups = restore_context(sources)
            return forward_step(
                state,
                ctx=ctx,
                coeffs=coeffs,
                program=program,
                update_kernel=update_kernel,
                known_fields=known,
                observation_time=clock[0]
                + (clock[1] + 1).astype(jnp.float32) * ctx.dt_scalar,
            )

        _, pullback = jax.vjp(linearize, active)
        cotangents = jax.tree.map(
            lambda gradient, _value: (
                jnp.zeros(
                    cast(ShapedArray, gradient.aval).shape,
                    cast(DTypeLike, cast(ShapedArray, gradient.aval).dtype),
                )
                if isinstance(gradient, jax.custom_derivatives.SymbolicZero)
                else gradient
            ),
            cotangents,
            known,
        )
        gradients = iter(pullback(cotangents)[0])
        return (
            tree.unflatten(
                [next(gradients) if flag else None for flag in columns.flags]
            ),
        )

    native_step.defvjp(step_forward, step_backward, symbolic_zeros=True)

    def step(state, coefficients, ctx, observation_origin, observation_step_offset):
        return native_step(
            (
                state,
                coefficients,
                source_values(ctx),
                (observation_origin, observation_step_offset),
            )
        )

    return step
