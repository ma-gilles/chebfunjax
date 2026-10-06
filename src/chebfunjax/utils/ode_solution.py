"""Construction scales and tolerances for marched ODE trajectories.

Provenance
----------
MATLAB source : @chebfun/odesol.m (vscale and RelTol/AbsTol conversion)
Chebfun commit: 7574c77
"""

import jax
import jax.numpy as jnp


def odesol(sol, domain, options=None, *, return_time=False):
    """Convert a dense ODE solution or restarted solutions to a Chebfun.

    The Python solver adapter expects objects with component-by-time ``y``
    and a callable ``sol(x)`` returning dense component-by-sample values.
    ``options`` accepts MATLAB's RelTol, AbsTol and happinessCheck names;
    omitted options are read from the first object's extdata.options if set.
    Set return_time=True for the source two-output form ``(t, y)``.
    This constructs a representation; it does not replace the integrator.

    Provenance
    ----------
    MATLAB source : @chebfun/odesol.m
    Chebfun commit: 7574c77
    """
    solutions = list(sol) if isinstance(sol, (list, tuple)) else [sol]
    if not solutions:
        raise ValueError("odesol requires at least one dense solution")

    def field(obj, name, default=None):
        return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)

    options_empty = (options is None or
                     (isinstance(options, (dict, list, tuple)) and not options))
    if options_empty:
        options = field(field(solutions[0], "extdata", {}), "options", {})
    rel = field(options, "RelTol", 1e-6)
    abs_ = field(options, "AbsTol", 1e-6)
    rel = 1e-6 if rel is None or jnp.asarray(rel).size == 0 else rel
    abs_ = 1e-6 if abs_ is None or jnp.asarray(abs_).size == 0 else abs_
    from chebfunjax.chebpref import ChebopPref
    check = field(options, "happinessCheck", ChebopPref().happinessCheck)
    if isinstance(check, str):
        check = check.lower().lstrip("@").removesuffix("check")
    states = jnp.concatenate([jnp.asarray(field(piece, "y")) for piece in solutions], axis=1)
    interpolants = []
    for piece in solutions:
        dense = field(piece, "sol")
        if not callable(dense):
            raise ValueError("odesol requires callable dense output on every solution")

        def values(x, dense=dense):
            return jnp.asarray(dense(x)).T

        interpolants.append(values)
    solution = _odesol_from_dense(interpolants, domain, states, rel, abs_, check=check)
    if return_time:
        from chebfunjax.chebfun1d.chebfun import Chebfun
        return Chebfun.identity(solution.domain), solution
    return solution


def _odesol_from_dense(interpolants, domain, states, relative_tolerance=1e-6,
                       absolute_tolerance=1e-6, *, check="standard"):
    """Construct the source array-valued solution from dense interpolants.

    Each callable returns samples-by-components values. One callable can
    cover a piecewise domain (restartSolver=False); otherwise there must be
    one per domain interval, ordered in the direction of the input domain.
    ``states`` contains all solver states, with components along axis0.

    This adapter ports representation construction, not an ODE integrator.
    It is adaptive Python control flow and is not a JIT entry point.

    Provenance
    ----------
    MATLAB source : @chebfun/odesol.m
    Chebfun commit: 7574c77
    """
    import warnings

    from chebfunjax.chebfun1d.chebfun import (
        Chebfun,
        _construct_with_splitting,
        _Piece,
    )
    from chebfunjax.domain import Domain

    states = jnp.asarray(states)
    if states.ndim != 2 or not states.shape[0] or not states.shape[1]:
        raise ValueError("odesol states must be a nonempty component-by-time matrix")
    input_bounds = tuple(float(value) for value in domain)
    if len(input_bounds) < 2 or not all(jnp.isfinite(jnp.asarray(input_bounds))):
        raise ValueError("odesol requires a finite domain with at least two endpoints")
    # MATLAB [dom,idx] = sort(dom); retain its literal endpoint-index rule
    # for reversing a cell of restarted solver handles.
    sort_indices = tuple(sorted(range(len(input_bounds)),
                                key=lambda i: input_bounds[i]))
    bounds = tuple(input_bounds[i] for i in sort_indices)
    if any(left == right for left, right in zip(bounds[:-1], bounds[1:])):
        raise ValueError("odesol domain breakpoints must be distinct")
    operators = [interpolants] if callable(interpolants) else list(interpolants)
    if len(operators) not in (1, len(bounds)-1):
        raise ValueError("odesol needs one interpolant or one per domain interval")
    single_interpolant = len(operators) == 1
    if len(sort_indices) > 1 and sort_indices[0] > sort_indices[1]:
        operators.reverse()
    if len(operators) == 1:
        operators = operators * (len(bounds)-1)

    tolerance, _ = _ode_fit_parameters(
        states, relative_tolerance, absolute_tolerance)
    tolerance = float(tolerance)
    hscale = max(abs(value) for value in bounds)

    def construct(splitting):
        pieces = []
        # constructor.m/getFun reduces the scales of a happy FUN to a
        # scalar before carrying them to the next problem-domain interval.
        running_scales = jnp.asarray(0.0, dtype=jnp.float64)
        for operator, left, right in zip(operators, bounds[:-1], bounds[1:]):
            if splitting:
                fitted = _construct_with_splitting(
                    operator, left, right, maxpow2=16, tol=tolerance,
                    split_max_length=20000, sample_test=False, check=check,
                    vscale=running_scales, hscale=hscale,
                )
                pieces.extend(fitted.funs)
            else:
                piece = _Piece.from_function(
                    operator, left, right, tol=tolerance, vscale=running_scales,
                    sample_test=False, check=check, hscale=hscale/(right-left),
                )
                pieces.append(piece)
            for piece in (fitted.funs if splitting else [piece]):
                if not piece.ishappy:
                    continue
                sampled_values = piece.tech.coeffs2vals(piece.tech.coeffs)
                running_scales = jnp.maximum(
                    running_scales, jnp.max(jnp.abs(sampled_values)))
        output_domain = (pieces[0].interval[0],) + tuple(
            piece.interval[1] for piece in pieces)
        solution = Chebfun(funs=pieces, domain=Domain(output_domain))
        if single_interpolant:
            # chebfun.m/getValuesAtBreakpoints evaluates a single function
            # handle at the breaks, including endpoints after chopping.
            # A cell of restarted handles instead keeps averaged FUN limits.
            values = jnp.asarray(operators[0](jnp.asarray(output_domain)))
            values = jnp.where(jnp.isnan(values), solution.point_values, values)
            solution = solution.set_point_values(values)
        return solution

    # The source suppresses constructor-notResolved messages, then retries
    # only if its first FUN is unhappy. Other errors must still propagate.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Chebtech[12]\.from_function: function did not converge.*")
        solution = construct(False)
        if not solution.funs[0].ishappy:
            solution = construct(True)
            if not solution.ishappy:
                warnings.warn(
                    "odesol: Function not resolved with splitting on, using 20000 pts.",
                    stacklevel=2,
                )
    return solution


@jax.jit
def _ode_fit_parameters(states, relative_tolerance, absolute_tolerance):
    """Return the source's shared fitting tolerance and component scales.

    ``states`` is a component-by-time array containing every marched segment.
    The source drops identically zero components before computing
    max(RelTol, AbsTol/vscale), using eps if all components are zero.

    Provenance
    ----------
    MATLAB source : @chebfun/odesol.m
    Chebfun commit: 7574c77
    """
    states = jnp.asarray(states)
    states = states.astype(jnp.complex128 if jnp.iscomplexobj(states)
                            else jnp.float64)
    scales = jnp.max(jnp.abs(states), axis=1)
    active = scales != 0.0
    rel = jnp.broadcast_to(jnp.asarray(relative_tolerance), scales.shape)
    abs_ = jnp.broadcast_to(jnp.asarray(absolute_tolerance), scales.shape)
    bounds = jnp.maximum(rel, abs_/jnp.where(active, scales, 1.0))
    tolerance = jnp.where(jnp.any(active),
                          jnp.max(jnp.where(active, bounds, 0.0)),
                          jnp.finfo(jnp.float64).eps)
    return tolerance, scales
