"""Compare exponential time-stepping schemes for Spin PDE operators.

Provenance
----------
MATLAB source : spincomp.m, spin.m, spin2.m, spin3.m,
    @spinoperator/solvepde.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and
    The Chebfun Developers.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from chebfunjax.operators.spinop import Spinop, spin
from chebfunjax.operators.spinop2 import Spinop2, spin2
from chebfunjax.operators.spinop3 import Spinop3, spin3
from chebfunjax.operators.spinpref import SpinPref, SpinPref2, SpinPref3
from chebfunjax.utils.quadrature import trigpts

__all__ = ["spincomp"]


def _family(S, pref):
    if isinstance(S, Spinop) and isinstance(pref, SpinPref):
        return 1
    if isinstance(S, Spinop2) and isinstance(pref, SpinPref2):
        return 2
    if isinstance(S, Spinop3) and isinstance(pref, SpinPref3):
        return 3
    raise TypeError(
        "spincomp requires a matching Spinop/SpinPref, Spinop2/SpinPref2, "
        "or Spinop3/SpinPref3 pair.")


def _components(solution):
    components = getattr(solution, "components", None)
    if components is not None:
        return list(components)
    return [solution]


def _contains_nan(component):
    """Inspect the numeric leaves of an Equinox Chebfun representation."""
    for leaf in jax.tree_util.tree_leaves(component):
        try:
            values = jnp.asarray(leaf)
            if jnp.issubdtype(values.dtype, jnp.number) and bool(
                    jnp.any(jnp.isnan(values))):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _initial_domain(S):
    initial = S.init
    if isinstance(initial, (list, tuple)):
        initial = initial[0]
    domain = getattr(initial, "domain", None)
    if domain is None:
        domain = S.domain
    if hasattr(domain, "breakpoints"):
        domain = domain.breakpoints
    if hasattr(domain, "a") and hasattr(domain, "b"):
        return (float(domain.a), float(domain.b))
    values = tuple(float(value) for value in domain)
    return values[:2]


def _sample(solution, grid, dim):
    components = _components(solution)
    if dim == 1:
        return [jnp.asarray(component(grid)) for component in components]
    if dim == 2:
        xx, yy = jnp.meshgrid(grid, grid)
        return [jnp.asarray(component(xx, yy)) for component in components]
    xx, yy, zz = jnp.meshgrid(grid, grid, grid)
    return [jnp.asarray(component(xx, yy, zz)) for component in components]


def _solver(S, dim, N, dt, scheme, pref, *, timing=True):
    if dim == 1:
        result = spin(S, N, dt, scheme=scheme, M=pref.M, dealias=False,
                      return_computing_time=timing)
        return result if timing else (result, None)
    local = type(pref)(M=pref.M)
    local.scheme = scheme
    # Native spincomp creates fresh defaults, copies only M, and suppresses
    # movie output for every reference and comparison solve.
    local.plot = "off"
    if dim == 2:
        result = spin2(S, N, dt, local, return_computing_time=timing)
    else:
        result = spin3(S, N, dt, local, return_computing_time=timing)
    return result if timing else (result, None)


def _plot_results(timesteps, elapsed, errors, schemes, final_time):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2)
    ax_error, ax_time = axes
    for index, label in enumerate(schemes):
        if index < 7:
            fmt, marker_size = ".-", 30
        elif index < 14:
            fmt, marker_size = "x-", 10
        else:
            fmt, marker_size = "o-", 10
        ax_error.loglog(timesteps, errors[:, index], fmt, linewidth=2,
                        markersize=marker_size)
        ax_time.loglog(elapsed[:, index], errors[:, index], fmt, linewidth=2,
                       markersize=marker_size)

    left = float(10.0 ** jnp.floor(jnp.log10(jnp.min(timesteps))))
    right = float(10.0 ** jnp.ceil(jnp.log10(jnp.max(timesteps))))
    # Keep MATLAB's reductions and IEEE log/power behavior on the JAX side:
    # an exact zero produces -inf then 0, so the outer max selects 1e-12.
    # Native min/max omit NaNs; the outer scalar pairs do so as well.
    down = jnp.fmax(
        jnp.fmin(1e-10, 10.0 ** jnp.floor(jnp.log10(jnp.nanmin(errors)))),
        1e-12,
    )
    up = jnp.fmin(
        jnp.fmax(1.0, 10.0 ** jnp.ceil(jnp.log10(jnp.nanmax(errors)))),
        1e2,
    )
    down, up = float(down), float(up)

    ax_error.tick_params(labelsize=16)
    ax_time.tick_params(labelsize=16)
    ax_error.set_xlim(left, right)
    ax_error.set_ylim(down, up)
    ax_error.set_xlabel("Relative time-step")
    ax_error.set_ylabel(f"Relative error at t = {final_time:.3f}")
    ax_error.legend(schemes, loc="upper left")

    min_time = float(jnp.min(elapsed))
    max_time = float(jnp.max(elapsed))
    ax_time.set_xlim(
        float(10.0 ** jnp.floor(jnp.log10(min_time))),
        float(10.0 ** jnp.ceil(jnp.log10(max_time))),
    )
    ax_time.set_ylim(down, up)
    ax_time.set_xlabel("Computer time (s)")
    ax_time.set_ylabel(f"Relative error at t = {final_time:.3f}")
    ax_time.legend(schemes, loc="upper right")
    return figure


def spincomp(S, N, dt, pref, *, plot=True):
    """Compare scheme accuracy and runtime at one or more time steps.

    This follows MATLAB ``spincomp`` for 1D, 2D and 3D diagonal Spin
    operators. It computes a reference solution using ``pecec736`` at half
    the smallest requested step, then reports the maximum sampled absolute
    error over 32 trigonometric points per spatial axis, normalized by the
    maximum magnitude of the reference solution. ``time`` contains each
    solver's internal time-stepping duration, excluding setup and output
    conversion. Both returned arrays have shape ``(len(dt), len(pref.scheme))``.

    The MATLAB convergence figure is created by default. Set ``plot=False``
    to suppress it, which is a Python convenience for headless callers. MATLAB
    spincomp has no sphere branch; sphere operators are therefore unsupported.

    Parameters
    ----------
    S : Spinop, Spinop2 or Spinop3
        PDE operator with an initial condition and time interval.
    N : int
        Number of Fourier grid points per spatial axis.
    dt : scalar or sequence of float
        Time steps to compare.
    pref : SpinPref, SpinPref2 or SpinPref3
        Matching preference object. Its ``scheme`` can be a string or a
        sequence of scheme names.
    plot : bool, optional
        Create the two MATLAB-style log-log comparison plots (default True).

    Returns
    -------
    err, time : jax.Array
        Relative sampled errors and internal solver times, each shaped by time
        step and scheme.

    Provenance
    ----------
    MATLAB source : spincomp.m, spin.m, spin2.m, spin3.m,
        @spinoperator/solvepde.m
    Chebfun commit: 7574c77
    """
    dim = _family(S, pref)
    step_array = jnp.atleast_1d(jnp.asarray(dt, dtype=jnp.float64)).reshape(-1)
    if step_array.size == 0:
        raise ValueError("dt must contain at least one positive time step.")
    steps = [float(value) for value in step_array]
    if any(not math.isfinite(value) or value <= 0 for value in steps):
        raise ValueError("dt values must be finite and positive.")
    scheme_value = pref.scheme
    schemes = [scheme_value] if isinstance(scheme_value, str) else list(scheme_value)
    schemes = [str(scheme) for scheme in schemes]
    if not schemes:
        raise ValueError("pref.scheme must contain at least one scheme.")

    grid = trigpts(32, _initial_domain(S))[0]
    reference, _ = _solver(
        S, dim, N, min(steps) / 2.0, "pecec736", pref, timing=False)
    reference_values = _sample(reference, grid, dim)
    scale = 0.0
    for values in reference_values:
        scale = max(scale, float(jnp.max(jnp.abs(values))))

    final_time = float(S.tspan[-1])
    errors = []
    elapsed = []
    for step in steps:
        error_row = []
        time_row = []
        for scheme in schemes:
            solution, computing_time = _solver(S, dim, N, step, scheme, pref)
            time_row.append(float(computing_time))
            values = _sample(solution, grid, dim)
            components = _components(solution)
            if len(values) != len(reference_values):
                raise ValueError("Solution and reference variable counts differ.")
            if any(_contains_nan(component) for component in components):
                error_row.append(float("nan"))
                continue
            error = 0.0
            for actual, expected in zip(values, reference_values):
                error = max(error, float(jnp.max(jnp.abs(actual - expected))))
            error_row.append(error)
        errors.append(error_row)
        elapsed.append(time_row)

    errors = jnp.asarray(errors, dtype=jnp.float64) / scale
    elapsed = jnp.asarray(elapsed, dtype=jnp.float64)
    relative_steps = step_array / final_time
    if plot:
        _plot_results(relative_steps, elapsed, errors, schemes, final_time)
        import matplotlib.pyplot as plt

        plt.show(block=False)
    return errors, elapsed
