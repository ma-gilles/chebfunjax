"""Source-shaped headless spin3 output and time collection.

Provenance
----------
MATLAB source : spin3.m, @spinoperator/solvepde.m,
    @spinop3/{discretize,getGrid,getDealiasingIndexes}.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math
import time
from numbers import Integral

import jax
import jax.numpy as jnp

from chebfunjax.operators._spin2_expinteg import (
    Expinteg,
    compute_coeffs,
    one_step,
    start_multistep,
)
from chebfunjax.operators._spin2_public import Spin2SolutionMatrix
from chebfunjax.operators.spinpref import SpinPref3
from chebfunjax.utils.quadrature import trigpts


class Spin3SolutionMatrix(Spin2SolutionMatrix):
    """3D functions indexed by unknown row and saved-time column.

    Integer access and iteration are zero-based column-major. This bounded
    adapter preserves the components view, not full MATLAB Chebmatrix algebra.

    Provenance
    ----------
    MATLAB source: @spinoperator/solvepde.m output construction.
    Chebfun commit: 7574c77.
    """


def _preferences(args, kwargs):
    pref = SpinPref3()
    names = {name.lower(): name for name in pref._defaults}
    if len(args) == 1 and isinstance(args[0], SpinPref3):
        for name in pref._defaults:
            setattr(pref, name, getattr(args[0], name))
    else:
        if len(args) % 2:
            raise TypeError("Expected a SpinPref3 object or name/value pairs.")
        for name, value in zip(args[::2], args[1::2]):
            if not isinstance(name, str) or name.lower() not in names:
                raise ValueError(f"Unrecognized spin3 preference: {name!r}")
            setattr(pref, names[name.lower()], value)
    for name, value in kwargs.items():
        if name.lower() not in names:
            raise ValueError(f"Unrecognized spin3 preference: {name!r}")
        setattr(pref, names[name.lower()], value)
    if not isinstance(pref.M, Integral) or isinstance(pref.M, bool) or pref.M < 1:
        raise ValueError("M must be a positive integer.")
    if isinstance(pref.dealias, bool):
        pref.dealias = "on" if pref.dealias else "off"
    if not isinstance(pref.dealias, str) or pref.dealias.lower() not in ("on", "off"):
        raise ValueError("dealias must be 'on' or 'off'.")
    return pref


def _linear_coefficients(S):
    """Expand the public stored differential-operator coefficients."""
    scales = [S._lin_scales] if S._n_vars == 1 else S._lin_scales
    names = [S._lin_ops] if S._n_vars == 1 else S._lin_ops
    operations = ("lap", "biharm", "triharm", "quadharm", "quintharm")
    rows = []
    for coefficients, operators in zip(scales, names):
        row = [0.] * 5
        for coefficient, operator in zip(coefficients, operators):
            row[operations.index(operator)] += coefficient
        rows.append(row)
    return rows


def _symbol(coefficients, n, domain):
    # Literal source discretize uses the x-domain width in ALL THREE directions.
    k = jnp.fft.fftfreq(n) * n
    d2 = -(k**2) * (2 * jnp.pi / (domain[1] - domain[0])) ** 2
    lap = d2[:, None, None] + d2[None, :, None] + d2[None, None, :]
    a, b, c, d, e = coefficients
    return a * lap + b * lap**2 + c * lap**3 + d * lap**4 + e * lap**5


def _dealias_indexes(n):
    # Source masks a central CUBE, not the union of high-frequency strips.
    start = n // 2 - math.ceil(n / 6)
    stop = n // 2 + math.ceil(n / 6)
    positions = jnp.arange(n)
    middle = (positions >= start) & (positions < stop)
    return middle[:, None, None] & middle[None, :, None] & middle[None, None, :]


def _output_values(coefficients, n, nvars, dealias):
    parts = []
    for k in range(nvars):
        c = coefficients[k * n:(k + 1) * n]
        if dealias:
            c = jnp.where(_dealias_indexes(n), 0, c)
        v = jnp.fft.ifftn(c)
        if bool(jnp.max(jnp.abs(jnp.imag(v))) < jnp.max(jnp.abs(v)) * 1e-10):
            v = jnp.real(v)
        parts.append(v)
    return parts


def _block_until_ready(values):
    for leaf in jax.tree_util.tree_leaves(values):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def _record_states(S, n, dt, pref, *, return_computing_time=False):
    domain = tuple(float(x) for x in S.domain)
    times = tuple(float(t) for t in S.tspan)
    if len(times) < 2 or not all(math.isfinite(t) for t in times):
        raise ValueError("spin3 requires at least two finite requested times.")
    if len(domain) != 6 or not all(math.isfinite(x) for x in domain):
        raise ValueError("spin3 requires a finite rectangular domain.")
    if any(b <= a for a, b in zip(domain[::2], domain[1::2])):
        raise ValueError("Domain endpoints must increase.")
    nvars = S._n_vars
    init = [S.init] if nvars == 1 else list(S.init)
    if len(init) != nvars:
        raise ValueError("Initial-condition count must equal the number of variables.")
    for f in init:
        if hasattr(f, "domain") and tuple(float(x) for x in f.domain) != domain:
            raise ValueError("The initial condition and operator do not live on the same domain.")
    x, _ = trigpts(n, domain[:2])
    y, _ = trigpts(n, domain[2:4])
    z, _ = trigpts(n, domain[4:])
    xx, yy, zz = jnp.meshgrid(x, y, z, indexing="xy")
    initial = [jnp.broadcast_to(jnp.asarray(f(xx, yy, zz)), (n, n, n)) for f in init]
    c0 = jnp.concatenate([jnp.fft.fftn(v) for v in initial], axis=0)
    lin = _linear_coefficients(S)
    L = jnp.concatenate([_symbol(c, n, domain) for c in lin], axis=0)
    real_operator = bool(jnp.all(jnp.imag(L) == 0))
    if real_operator:
        L = jnp.real(L)

    def nv(stacked):
        if nvars == 1:
            return jnp.broadcast_to(jnp.asarray(S._nonlin_vals(stacked)), stacked.shape)
        parts = [stacked[k * n:(k + 1) * n] for k in range(nvars)]
        return jnp.concatenate([
            jnp.broadcast_to(jnp.asarray(fn(*parts)), (n, n, n))
            for fn in S._nonlin_vals
        ], axis=0)

    # Source initial nonlinear evaluation uses vInit itself, not FFT replay.
    nonlin_initial = nv(jnp.concatenate(initial, axis=0))
    nc0 = jnp.concatenate([
        jnp.fft.fftn(nonlin_initial[k * n:(k + 1) * n]) for k in range(nvars)
    ], axis=0)
    scheme = Expinteg(pref.scheme)
    if scheme.steps == 1:
        states, nonlinear = [c0], [nc0]
    else:
        states, nonlinear = start_multistep(
            scheme, dt, L, pref.M, real_operator, 1, nv,
            jnp.fft.ifftn, jnp.fft.fftn, nvars, c0, nc0,
        )
    coefficients = compute_coeffs(scheme, dt, L, pref.M, real_operator)
    saved, actual_times = [initial], [0.0]
    step = scheme.steps - 1
    t = step * dt
    pos = 1
    if return_computing_time:
        # MATLAB solvepde starts tic after setup and multistep initialization.
        _block_until_ready((states, nonlinear, coefficients, saved))
        start = time.perf_counter()
    while t < times[-1]:
        states, nonlinear = one_step(
            scheme, coefficients, 1, nv, jnp.fft.ifftn, jnp.fft.fftn,
            nvars, states, nonlinear,
        )
        if bool(jnp.any(jnp.isnan(states[0]))):
            raise RuntimeError("The solution blew up. Try a smaller time-step.")
        step += 1
        t = step * dt
        # Preserve source value-copy order: cOld is retained BEFORE masking
        # cNew for output, so dealiasing does not alter the next step's state.
        if abs(t - times[pos]) < 1e-10:
            saved.append(_output_values(states[0], n, nvars, pref.dealias.lower() == "on"))
            actual_times.append(t)
            pos += 1
            if pos == len(times):
                break
    if return_computing_time and len(times) == 2 and len(actual_times) < 2:
        raise IndexError("native solvepde has no saved tOut(2) checkpoint")
    if return_computing_time:
        _block_until_ready((states, nonlinear, saved))
        computing_time = time.perf_counter() - start
        return (saved, jnp.asarray(actual_times, dtype=jnp.float64), domain,
                len(times), computing_time)
    return saved, jnp.asarray(actual_times, dtype=jnp.float64), domain, len(times)


def solve_public(S, n, dt, args, kwargs, *, return_times=False,
                 return_computing_time=False):
    """Construct source final or variable-by-time function output.

    ``return_times`` expresses MATLAB public spin3's two-output call.
    ``return_computing_time`` exposes solvepde's internal time-stepping duration
    for source-shaped comparison utilities.

    Provenance
    ----------
    MATLAB source : spin3.m, @spinoperator/solvepde.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.chebfun3d.chebfun3 import chebfun3

    if not isinstance(n, Integral) or isinstance(n, bool) or n < 1:
        raise ValueError("N must be a positive integer.")
    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")
    if S._lin_scales is None or S._nonlin_vals is None or S.init is None:
        raise ValueError("Set linear, nonlinear and initial-condition data before solving.")
    pref = _preferences(args, kwargs)
    recorded = _record_states(
        S, n, dt, pref, return_computing_time=return_computing_time)
    if return_computing_time:
        saved, actual_times, domain, requested_count, computing_time = recorded
    else:
        saved, actual_times, domain, requested_count = recorded
    selected = saved[-1:] if requested_count == 2 else saved
    rows = [[chebfun3(jnp.transpose(column[k], (1, 0, 2)), domain, trig=True)
             for column in selected] for k in range(S._n_vars)]
    result = rows[0][0] if requested_count == 2 and S._n_vars == 1 else Spin3SolutionMatrix(rows)
    if return_computing_time:
        return ((result, actual_times, computing_time) if return_times
                else (result, computing_time))
    return (result, actual_times) if return_times else result
