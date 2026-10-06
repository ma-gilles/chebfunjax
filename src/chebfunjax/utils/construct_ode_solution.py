"""Source translation of Chebfun's public constructODEsol orchestrator.

This candidate routes a caller-supplied solver; it does not implement MATLAB's
ODE45/113/15s/78/89 integrators or their native option defaults.

Provenance
----------
MATLAB source : @chebfun/constructODEsol.m, @chebfun/odesol.m,
                @chebfun/join.m
Chebfun commit: 7574c77
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import jax.numpy as jnp


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _options(args: tuple[Any, ...]) -> Any:
    """Return source's first ODESET-like argument, when supplied."""
    return args[0] if args else None


def _option(options: Any, name: str, default: Any = None) -> Any:
    return _field(options, name, default)


def _event_present(solution: Any) -> bool:
    events = _field(solution, "ie", ())
    if events is None:
        return False
    if isinstance(events, (str, bytes, Mapping, tuple, list)):
        return len(events) != 0
    return jnp.asarray(events).size != 0


def _events_enabled(options: Any) -> bool:
    """MATLAB isempty adapter that does not coerce callback handles to arrays."""
    events = _option(options, "Events", None)
    if events is None:
        return False
    if isinstance(events, (str, bytes, Mapping, tuple, list)):
        return len(events) != 0
    if callable(events):
        return True
    return jnp.asarray(events).size != 0


def _event_time(solution: Any) -> float:
    values = jnp.asarray(_field(solution, "xe")).reshape(-1)
    if values.size != 1:
        raise ValueError("constructODEsol Python adapter requires one scalar event time")
    return float(values[0])


def _source_nan_tail(start: float, end: float, n_components: int):
    """Build the source constant-NaN FUN over the literal supplied interval.

    Domain validates ascending finite endpoints. In particular, this does not
    reverse a backward-event tail to make it constructible: MATLAB's source
    sends that decreasing interval to the bounded constructor too.
    """
    from chebfunjax.chebfun1d.chebfun import Chebfun
    from chebfunjax.domain import Domain

    domain = Domain((start, end))
    return Chebfun.from_values(
        jnp.full((1, n_components), jnp.nan, dtype=jnp.float64),
        domain=domain,
    )


def _source_time_tail(value_start: float, value_end: float,
                      domain_start: float, domain_end: float):
    """Build the literal two-sample time tail used by constructODEsol."""
    from chebfunjax.chebfun1d.chebfun import Chebfun
    from chebfunjax.domain import Domain

    return Chebfun.from_values(
        jnp.asarray([value_start, value_end], dtype=jnp.float64),
        domain=Domain((domain_start, domain_end)),
    )


def _constructODEsol(
    solver: Callable[..., Any],
    odefun: Callable[..., Any],
    tspan: Any,
    y0: Any,
    *solver_args: Any,
    return_time: bool = False,
):
    """Run the source restart/event orchestration around a supplied solver.

    The solver is called as ``solver(odefun, (ta, tb), y0, *solver_args)`` and
    must return the same adapter record accepted by public ``Chebfun.odesol``:
    component-by-time ``y``, callable dense interpolant ``sol(x)`` returning
    components-by-samples, optional ``extdata.options``, and optional event
    fields ``ie``/``xe``. ``return_time`` models MATLAB's ``nargout > 1``.

    Python adapters: ``tspan`` is converted to a tuple of static host floats;
    ``solver_args[0]`` is treated as the ODESET-like options object; event
    presence is tested from the returned record. Native MATLAB solver defaults,
    ODESET parsing, Events callbacks, and native integrator behavior are not
    reproduced here.

    Provenance
    ----------
    MATLAB source : @chebfun/constructODEsol.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.chebfun1d.chebfun import Chebfun

    span = tuple(float(v) for v in tspan)
    if len(span) < 2:
        raise ValueError("constructODEsol requires at least two tspan values")
    initial = jnp.asarray(y0)
    options = _options(solver_args)
    restart = bool(_option(options, "restartSolver", True))

    def call(span_arg: tuple[float, ...], state: Any):
        return solver(odefun, span_arg, state, *solver_args)

    def as_outputs(solution: Any, domain: tuple[float, ...]):
        return Chebfun.odesol(
            solution, domain, options=options, return_time=return_time
        )

    # Source branch: one call for a two-point span or restartSolver=false.
    if len(span) == 2 or not restart:
        # The no-restart source branch forwards every supplied tspan knot.
        sol = call(span, initial)
        if not _event_present(sol):
            return as_outputs(sol, span)

        old_end = span[1]
        event = _event_time(sol)
        # Literal MATLAB indexing: overwrite tspan(2), even when a longer
        # no-restart span was supplied.
        valid_span = (span[0], event) + span[2:]
        valid = Chebfun.odesol(sol, valid_span, options=options)
        states = jnp.asarray(_field(sol, "y"))
        nan_tail = _source_nan_tail(event, old_end, states.shape[0])
        joined = valid.join(nan_tail)
        if not return_time:
            return joined
        time_tail = _source_time_tail(event, old_end, event, old_end)
        time_head = _source_time_tail(span[0], event, span[0], event)
        return time_head.join(time_tail), joined

    # Source branch: restart once per consecutive tspan pair.
    num_pieces = len(span) - 1
    event_detection = _events_enabled(options)
    solutions = []
    state = initial
    stopped = False
    for k in range(num_pieces):
        sol_piece = call((span[k], span[k + 1]), state)
        solutions.append(sol_piece)
        if event_detection and _event_present(sol_piece):
            stopped = True
            break
        state = jnp.asarray(_field(sol_piece, "y"))[:, -1]

    if not stopped:
        return as_outputs(solutions, span)

    old_end = span[-1]
    event = _event_time(solutions[-1])
    valid_span = span[:len(solutions) + 1]
    valid_span = valid_span[:-1] + (event,)
    valid_time, valid = Chebfun.odesol(
        solutions, valid_span, options=options, return_time=True
    )
    states0 = jnp.asarray(_field(solutions[0], "y"))
    nan_tail = _source_nan_tail(event, old_end, states0.shape[0])
    joined = valid.join(nan_tail)
    if not return_time:
        return joined

    # Literal source quirk: this tail starts at tspan(2), not the detected
    # event, when an event follows multiple restarted intervals. Chebfun.join
    # remaps its interval by length and keeps these supplied values.
    time_tail_start = valid_span[1]
    time_tail = _source_time_tail(
        time_tail_start, old_end, time_tail_start, old_end
    )
    return valid_time.join(time_tail), joined


def constructODEsol(solver, odefun, tspan, y0, *solver_args, return_time=False):
    """Run the complete source restart/event construction around a solver.

    See the module's solver-record adapter. Python scalar Chebfuns store a
    single column as one-dimensional coefficients, preserving scalar calls;
    systems retain the source coupled array representation and tolerance.
    return_time=True adapts the two-output form to (t, y).

    Provenance
    ----------
    MATLAB source : @chebfun/constructODEsol.m
    Chebfun commit: 7574c77
    """
    result = _constructODEsol(solver, odefun, tspan, y0, *solver_args,
                             return_time=return_time)
    if return_time:
        t, y = result
    else:
        y = result
    if y.n_columns == 1:
        y = y.extract_columns(0)
    return (t, y) if return_time else y
