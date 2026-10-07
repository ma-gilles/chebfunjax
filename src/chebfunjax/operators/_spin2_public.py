"""Source-shaped headless spin2 output and time collection.

Provenance
----------
MATLAB source : spin2.m, @spinoperator/solvepde.m,
    @spinop2/{discretize,getGrid,getDealiasingIndexes}.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math
from numbers import Integral

import jax.numpy as jnp

from chebfunjax.operators._spin2_expinteg import (
    Expinteg,
    compute_coeffs,
    one_step,
    start_multistep,
)
from chebfunjax.operators.spinpref import SpinPref2
from chebfunjax.utils.quadrature import trigpts


class Spin2SolutionMatrix:
    """Function matrix with variables in rows and saved times in columns.

    This bounded adapter implements shape, element access and column-major
    iteration, not the full MATLAB Chebmatrix algebra. Integer indexing is
    zero-based column-major; ``result[row, time]`` selects a Chebfun2.

    Provenance
    ----------
    MATLAB source : @spinoperator/solvepde.m (output construction),
        @chebmatrix/subsref.m
    Chebfun commit: 7574c77
    """

    def __init__(self, rows):
        self._rows = tuple(tuple(row) for row in rows)
        self.shape = (len(self._rows), len(self._rows[0]))
        if not all(len(row) == self.shape[1] for row in self._rows):
            raise ValueError("A function matrix must be rectangular.")

    def __getitem__(self, index):
        if isinstance(index, tuple):
            row, col = index
        else:
            col, row = divmod(index, self.shape[0])
        return self._rows[row][col]

    def __len__(self):
        return self.shape[0] * self.shape[1]

    def __iter__(self):
        return (self[k] for k in range(len(self)))

    @property
    def components(self):
        """Column-major entries, preserving the prior one-column vector API.

        Provenance
        ----------
        MATLAB source : @spinoperator/solvepde.m
        Chebfun commit: 7574c77
        """
        return list(self)


def _preferences(args, kwargs):
    pref = SpinPref2()
    names = {name.lower(): name for name in pref._defaults}
    if len(args) == 1 and isinstance(args[0], SpinPref2):
        for name in pref._defaults:
            setattr(pref, name, getattr(args[0], name))
    else:
        if len(args) % 2:
            raise TypeError("Expected a SpinPref2 object or name/value pairs.")
        for name, value in zip(args[::2], args[1::2]):
            if not isinstance(name, str) or name.lower() not in names:
                raise ValueError(f"Unrecognized spin2 preference: {name!r}")
            setattr(pref, names[name.lower()], value)
    for name, value in kwargs.items():
        if name.lower() not in names:
            raise ValueError(f"Unrecognized spin2 preference: {name!r}")
        setattr(pref, names[name.lower()], value)
    if not isinstance(pref.M, Integral) or isinstance(pref.M, bool) or pref.M < 1:
        raise ValueError("M must be a positive integer.")
    if isinstance(pref.dealias, bool):
        pref.dealias = "on" if pref.dealias else "off"
    if not isinstance(pref.dealias, str) or pref.dealias.lower() not in ("on", "off"):
        raise ValueError("dealias must be 'on' or 'off'.")
    return pref


def _symbol(coefficients, n, domain):
    # Literal source discretize uses the x-domain width in BOTH directions.
    k = jnp.fft.fftfreq(n) * n
    d2 = -(k**2) * (2 * jnp.pi / (domain[1] - domain[0])) ** 2
    lap = d2[:, None] + d2[None, :]
    a, b, c, d, e = coefficients
    return a * lap + b * lap**2 + c * lap**3 + d * lap**4 + e * lap**5


def _dealias_indexes(n):
    # Source masks a central SQUARE, not the union of high-frequency strips.
    start = n // 2 - math.ceil(n / 6)
    stop = n // 2 + math.ceil(n / 6)
    positions = jnp.arange(n)
    middle = (positions >= start) & (positions < stop)
    return middle[:, None] & middle[None, :]


def _output_values(coefficients, n, nvars, dealias):
    parts = []
    for k in range(nvars):
        c = coefficients[k * n:(k + 1) * n]
        if dealias:
            c = jnp.where(_dealias_indexes(n), 0, c)
        v = jnp.fft.ifft2(c)
        if bool(jnp.max(jnp.abs(jnp.imag(v))) < jnp.max(jnp.abs(v)) * 1e-10):
            v = jnp.real(v)
        parts.append(v)
    return parts


def _record_states(S, n, dt, pref):
    domain = tuple(float(x) for x in S.domain)
    times = tuple(float(t) for t in S.tspan)
    if len(times) < 2 or not all(math.isfinite(t) for t in times):
        raise ValueError("spin2 requires at least two finite requested times.")
    if len(domain) != 4 or not all(math.isfinite(x) for x in domain):
        raise ValueError("spin2 requires a finite rectangular domain.")
    if domain[1] <= domain[0] or domain[3] <= domain[2]:
        raise ValueError("Domain endpoints must increase.")
    nvars = S._n_vars
    init = [S.init] if nvars == 1 else list(S.init)
    if len(init) != nvars:
        raise ValueError("Initial-condition count must equal the number of variables.")
    for f in init:
        if hasattr(f, "domain") and tuple(float(x) for x in f.domain) != domain:
            raise ValueError("The initial condition and operator do not live on the same domain.")
    x, _ = trigpts(n, domain[:2])
    y, _ = trigpts(n, domain[2:])
    xx, yy = jnp.meshgrid(x, y, indexing="xy")
    initial = [jnp.broadcast_to(jnp.asarray(f(xx, yy)), (n, n)) for f in init]
    c0 = jnp.concatenate([jnp.fft.fft2(v) for v in initial], axis=0)
    lin = [S._lin_coeffs] if nvars == 1 else S._lin_coeffs
    L = jnp.concatenate([_symbol(c, n, domain) for c in lin], axis=0)
    real_operator = bool(jnp.all(jnp.imag(L) == 0))
    if real_operator:
        L = jnp.real(L)

    def nv(stacked):
        if nvars == 1:
            return jnp.broadcast_to(jnp.asarray(S._nonlin_vals(stacked)), stacked.shape)
        parts = [stacked[k * n:(k + 1) * n] for k in range(nvars)]
        return jnp.concatenate([
            jnp.broadcast_to(jnp.asarray(fn(*parts)), (n, n))
            for fn in S._nonlin_vals
        ], axis=0)

    # Source initial nonlinear evaluation uses vInit itself, not FFT replay.
    nonlin_initial = nv(jnp.concatenate(initial, axis=0))
    nc0 = jnp.concatenate([
        jnp.fft.fft2(nonlin_initial[k * n:(k + 1) * n]) for k in range(nvars)
    ], axis=0)
    scheme = Expinteg(pref.scheme)
    if scheme.steps == 1:
        states, nonlinear = [c0], [nc0]
    else:
        states, nonlinear = start_multistep(
            scheme, dt, L, pref.M, real_operator, 1, nv,
            jnp.fft.ifft2, jnp.fft.fft2, nvars, c0, nc0,
        )
    coefficients = compute_coeffs(scheme, dt, L, pref.M, real_operator)
    saved, actual_times = [initial], [0.0]
    step = scheme.steps - 1
    t = step * dt
    pos = 1
    while t < times[-1]:
        states, nonlinear = one_step(
            scheme, coefficients, 1, nv, jnp.fft.ifft2, jnp.fft.fft2,
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
    return saved, jnp.asarray(actual_times, dtype=jnp.float64), domain, len(times)


def solve_public(S, n, dt, args, kwargs, *, return_times=False):
    """Construct source final or variable-by-time function output.

    ``return_times`` expresses MATLAB public spin2's two-output call; it
    does not expose internal solvepde's third computing-time output.

    Provenance
    ----------
    MATLAB source : spin2.m, @spinoperator/solvepde.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.chebfun2d.chebfun2 import Chebfun2

    if not isinstance(n, Integral) or isinstance(n, bool) or n < 1:
        raise ValueError("N must be a positive integer.")
    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")
    if S._lin_coeffs is None or S._nonlin_vals is None or S.init is None:
        raise ValueError("Set linear, nonlinear and initial-condition data before solving.")
    pref = _preferences(args, kwargs)
    saved, actual_times, domain, requested_count = _record_states(S, n, dt, pref)
    selected = saved[-1:] if requested_count == 2 else saved
    rows = [[Chebfun2.from_values(column[k], domain=domain, trig=True)
             for column in selected] for k in range(S._n_vars)]
    result = rows[0][0] if requested_count == 2 and S._n_vars == 1 else Spin2SolutionMatrix(rows)
    return (result, actual_times) if return_times else result
