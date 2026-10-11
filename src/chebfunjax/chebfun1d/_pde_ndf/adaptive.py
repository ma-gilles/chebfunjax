"""Polynomial PDE spatial happiness and private callback controls.

Provenance
----------
Chebfun @chebfun/pdeSolve.m adaptiveEvent and spatial setup.
Native commit: 7574c77680d7e82b79626300bf255498271a72df.
Original source: Copyright 2017 The University of Oxford and Chebfun Developers.

scalar_happiness and system_happiness serve the public NDF spatial restart driver.
The state dataclasses and adaptive_event helper support private source controls;
the public driver constructs accepted Chebfuns and manages restart state.
Numerical work uses JAX; shape-changing adaptivity uses Python control flow.
"""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from chebfunjax.tech.chebtech import Chebtech2


@dataclass(frozen=True)
class AcceptedSlice:
    time: float
    values: jax.Array


@dataclass(frozen=True)
class AdaptiveState:
    length: int
    time: float
    accepted: tuple[AcceptedSlice, ...] = ()


@dataclass(frozen=True)
class CallbackResult:
    state: AdaptiveState
    stop: bool
    decisions: tuple[tuple[bool, int], ...]


def initial_state(simplified_length: int, start_time: float) -> AdaptiveState:
    """Input length belongs to the already-simplified source initial Chebfun."""
    if simplified_length < 1:
        raise ValueError("An initial nonempty polynomial is required")
    return AdaptiveState(max(simplified_length, 9), float(start_time))


def scalar_happiness(values: jax.Array, tolerance: float) -> tuple[bool, int]:
    """Source numeric-OP check, including scalar multiply/divide and roundtrip."""
    values = jnp.asarray(values)
    if values.ndim != 1:
        raise NotImplementedError("Only one scalar PDE is qualified")
    if not 0 < tolerance < 1:
        raise ValueError("Tolerance must be between zero and one")
    return system_happiness(values[:, None], tolerance)


def system_happiness(values: jax.Array, tolerance: float) -> tuple[bool, int]:
    """Check the native weighted component combination.

    Provenance
    ----------
    Chebfun @chebfun/pdeSolve.m adaptiveEvent, native commit 7574c77.
    """
    values = jnp.asarray(values)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("Expected spatial rows and component columns")
    if not 0 < tolerance < 1:
        raise ValueError("Tolerance must be between zero and one")
    c = (1 + jnp.sin(jnp.arange(1, values.shape[1] + 1, dtype=jnp.float64)))[:, None]
    weighted = (values @ c / jnp.sum(c))[:, 0]
    coeffs = Chebtech2.vals2coeffs(weighted)
    reconstructed = Chebtech2.coeffs2vals(coeffs)
    happy, cutoff = Chebtech2.happiness_check(
        coeffs,
        reconstructed,
        op=None,
        tol=tolerance,
        vscale=float(jnp.max(jnp.abs(reconstructed))),
        hscale=1.0,
        check="standard",
        sample_test=False,
    )
    return bool(happy), int(cutoff)


def adaptive_event(
    state: AdaptiveState,
    times: jax.Array,
    values: jax.Array,
    designated_times: jax.Array,
    *,
    flag: str = "",
    tolerance: float = 1e-6,
) -> CallbackResult:
    """Replay source stop/store decisions, preserving last accepted state."""
    if flag in ("init", "done"):
        return CallbackResult(state, False, ())
    if flag:
        raise ValueError("Unsupported output callback flag")
    times = jnp.atleast_1d(times)
    values = jnp.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape != (state.length, times.size):
        raise ValueError("Expected scalar columns at the current spatial length")
    designated_times = jnp.atleast_1d(designated_times)
    decisions = []
    for k in range(times.size):
        column = values[:, k]
        happy, cutoff = scalar_happiness(column, tolerance)
        decisions.append((happy, cutoff))
        if not happy:
            return CallbackResult(
                replace(state, length=2 * state.length - 1), True, tuple(decisions)
            )
        time = float(times[k])
        if not bool(jnp.any(jnp.abs(designated_times - time) < 1e-6)):
            continue
        accepted = AcceptedSlice(time, column)
        state = replace(state, time=time, accepted=state.accepted + (accepted,))
    return CallbackResult(state, False, tuple(decisions))
