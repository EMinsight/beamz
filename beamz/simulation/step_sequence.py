"""Shared orchestration for one electromagnetic step."""

from __future__ import annotations

from typing import TypeVar

StateT = TypeVar("StateT")
PayloadT = TypeVar("PayloadT")


def run_step_sequence(
    state: StateT,
    *,
    pre_e,
    prepare,
    update_h,
    post_h,
    update_e,
    post_e,
    finalize,
) -> StateT:
    """Execute the shared FDTD phase order for one step."""
    state = pre_e(state)
    state, payload = prepare(state)
    state = update_h(state, payload)
    state = post_h(state)
    state = update_e(state, payload)
    state = post_e(state)
    return finalize(state)
