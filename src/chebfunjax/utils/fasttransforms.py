"""Discrete cosine/sine/Legendre transforms (MATLAB chebfun.dct family).

Cosine and sine transforms use the source Chebtech/FFT reductions in JAX.

Provenance
----------
MATLAB source : @chebfun/dct.m, idct.m, dst.m, idst.m, dlt.m, idlt.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and the
Chebfun Developers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

__all__ = ["dct", "idct", "dst", "idst", "dlt", "idlt"]


def _transform_columns(x):
    """Interpret vectors as columns; preserve explicit matrix columns."""
    u = jnp.asarray(x)
    shape = u.shape
    if u.ndim > 2:
        raise ValueError("Transform input must be a scalar, vector or column matrix")
    if u.ndim < 2:
        u = u.reshape((-1, 1))
    if not u.shape[0]:
        raise ValueError("Transform input must have at least one row")
    return u.astype(jnp.result_type(u.dtype, jnp.float64)), shape


def dct(x, kind: int = 2):
    """Columnwise discrete cosine transform, source types I through VIII.

    Uses Chebfun's unnormalized convention, not orthonormal toolbox scaling.
    ``kind`` must be static when tracing with JAX.

    Provenance
    ----------
    MATLAB source : @chebfun/dct.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

    u, shape = _transform_columns(x)
    n, m = u.shape
    if kind == 1:
        ends = jnp.asarray([0, n - 1])
        y = Chebtech2.coeffs2vals(u.at[ends].set(.5 * u[ends]))[::-1]
    elif kind == 2:
        y = (n / 2) * Chebtech1.vals2coeffs(u[::-1])
        y = y.at[0].set(2 * y[0])
    elif kind == 3:
        y = Chebtech1.coeffs2vals(u.at[0].set(.5 * u[0]))[::-1]
    elif kind == 4:
        v = jnp.zeros((2 * n, m), dtype=u.dtype).at[1::2].set(u)
        y = dct(v, 3)[:n]
    elif kind == 5:
        v = jnp.concatenate((2 * u[:1], u[1:], jnp.zeros((n, m), u.dtype)))
        y = dct(v, 1)[::2]
    elif kind == 6:
        y = dct(jnp.concatenate((u, jnp.zeros((n - 1, m), u.dtype))), 2)[::2]
    elif kind == 7:
        v = jnp.zeros((2 * n - 1, m), u.dtype).at[::2].set(u)
        y = dct(v.at[0].set(2 * v[0]), 3)[:n]
    elif kind == 8:
        y = dct(jnp.concatenate((u, jnp.zeros((n + 1, m), u.dtype))), 2)[1::2]
    else:
        raise ValueError("CHEBEFUN:CHEBFUN:dct:type: Unknown DCT type.")
    return y.reshape(shape)


def idct(x, kind: int = 2):
    """Inverse source DCT of the same type, I through VIII.

    Provenance
    ----------
    MATLAB source : @chebfun/idct.m
    Chebfun commit: 7574c77
    """
    u, shape = _transform_columns(x)
    n = u.shape[0]
    if kind in (1, 2, 3, 4):
        inverse = {1: 1, 2: 3, 3: 2, 4: 4}[kind]
        denominator = jnp.asarray(n - 1 if kind == 1 else n, jnp.float64)
        y = (2 / denominator) * dct(u, inverse)
    elif kind == 5:
        y = (4 / (2 * n - 1)) * dct(u.at[0].set(u[0] / 2), 5)
        y = y.at[0].set(y[0] / 2)
    elif kind == 6:
        y = (4 / (2 * n - 1)) * dct(u.at[0].set(u[0] / 2), 7)
        y = y.at[-1].set(y[-1] / 2)
    elif kind == 7:
        y = (4 / (2 * n - 1)) * dct(u.at[-1].set(u[-1] / 2), 6)
        y = y.at[0].set(y[0] / 2)
    elif kind == 8:
        y = (4 / (2 * n + 1)) * dct(u, 8)
    else:
        raise ValueError("CHEBEFUN:CHEBFUN:idct:type: Unknown IDCT type.")
    return y.reshape(shape)


def dst(x, kind: int = 1):
    """Columnwise discrete sine transform, source types I through VIII.

    Type I deliberately preserves the source's unusual pure-imaginary branch:
    it returns ``imag(y)`` as a real array. Traced complex inputs keep a complex
    storage dtype, with the same selected numeric values.

    Provenance
    ----------
    MATLAB source : @chebfun/dst.m
    Chebfun commit: 7574c77
    """
    u, shape = _transform_columns(x)
    n, m = u.shape
    if kind == 1:
        z = jnp.zeros((1, m), u.dtype)
        y = jnp.fft.fft(jnp.concatenate((z, u, z, -u[::-1])), axis=0)
        y = -1j * (y[2 * n + 1:n + 1:-1] / 2)
        if not jnp.iscomplexobj(u):
            y = jnp.real(y)
        elif isinstance(u, jax.core.Tracer):
            y = jnp.where(jnp.all(jnp.real(u) == 0), jnp.imag(y), y)
        elif bool(jnp.all(jnp.real(u) == 0)):
            y = jnp.imag(y)
    elif kind in (2, 3):
        v = u.at[::2].set(-u[::2])
        y = dct(v[::-1], kind)
        y = y.at[::2].set(-y[::2])[::-1]
    elif kind == 4:
        y = dct(u[::-1], 4)
        y = y.at[1::2].set(-y[1::2])
    elif kind == 5:
        y = dst(jnp.concatenate((u, jnp.zeros((n, m), u.dtype))), 1)[1::2]
    elif kind == 6:
        v = jnp.zeros((2 * n, m), u.dtype).at[::2].set(u)
        y = dst(v, 1)[:n]
    elif kind == 7:
        y = dst(jnp.concatenate((u, jnp.zeros((n, m), u.dtype))), 1)[::2]
    elif kind == 8:
        y = dst(jnp.concatenate((u, jnp.zeros((n + 1, m), u.dtype))), 2)[:2*n:2]
    else:
        raise ValueError("CHEBEFUN:CHEBFUN:dct:type: Unknown DST type.")
    return y.reshape(shape)


def idst(x, kind: int = 1):
    """Inverse source DST I through VII; type VIII raises the native error.

    Provenance
    ----------
    MATLAB source : @chebfun/idst.m
    Chebfun commit: 7574c77
    """
    u, shape = _transform_columns(x)
    n = u.shape[0]
    if kind in (1, 2, 3, 4):
        inverse = {1: 1, 2: 3, 3: 2, 4: 4}[kind]
        y = (2 / (n + 1 if kind == 1 else n)) * dst(u, inverse)
    elif kind in (5, 6, 7):
        inverse = {5: 5, 6: 7, 7: 6}[kind]
        y = (4 / (2 * n + 1)) * dst(u, inverse)
    elif kind == 8:
        raise ValueError("CHEBFUN:CHEBFUN:IDST:EIGHT: Not implemented")
    else:
        raise ValueError("CHEBFUN:CHEBFUN:idct:type: Unknown IDST type.")
    return y.reshape(shape)


def dlt(c):
    """Evaluate Legendre coefficient columns at the Gauss-Legendre nodes.

    For fewer than 5000 rows, this uses MATLAB's rolling three-term
    recurrence. Larger inputs use the source Legendre-to-Chebyshev transform
    followed by the separate 18-term `ndct_legpts` expansion. Inputs and
    outputs are JAX arrays; vectors and `(n, m)` coefficient matrices are
    supported.

    Provenance
    ----------
    MATLAB source : @chebfun/dlt.m, local dlt_direct and ndct_legpts
    Chebfun commit: 7574c77
    """
    c = jnp.asarray(c)
    if c.ndim not in (1, 2):
        raise ValueError("c must be a vector or a 2-D column matrix")
    n = c.shape[0]
    if n == 0:
        return c
    if n == 1:
        return jnp.ones_like(c) + 0 * c
    if n >= 5000:
        from chebfunjax.utils.transforms import leg2cheb

        return _dlt_ndct_legpts(leg2cheb(c))

    from chebfunjax.utils.quadrature import legpts

    x, _ = legpts(n)
    vector_input = c.ndim == 1
    cm = c[:, None] if vector_input else c
    p0 = jnp.ones_like(x)
    p1 = x
    values = p0[:, None] * cm[0, :][None, :] + p1[:, None] * cm[1, :][None, :]

    def step(k, state):
        pm1, p, out = state
        kf = jnp.asarray(k, dtype=x.dtype)
        pp1 = ((2.0 - 1.0 / (kf + 1.0)) * (p * x)
               - (1.0 - 1.0 / (kf + 1.0)) * pm1)
        out = out + pp1[:, None] * cm[k + 1, :][None, :]
        return p, pp1, out

    _, _, values = lax.fori_loop(1, n - 1, step, (p0, p1, values))
    return values[:, 0] if vector_input else values


def _dct3_scaled(c):
    """MATLAB dct3_scaled: `chebtech1.coeffs2vals(c)[::-1]`."""
    from chebfunjax.tech.chebtech import Chebtech1

    return Chebtech1.coeffs2vals(c)[::-1]


def _dst3_shifted(c):
    r"""MATLAB shifted DST-III from its sine/cosine identity.

    At first-kind angles ``theta_j=(j+1/2)*pi/n``,
    ``sin(m*theta_j)=(-1)^j*cos((n-m)*theta_j)``. Thus the source
    `[c(2:end);0]` shifted DST-III is the scaled DCT-III of
    `[0;c(end:-1:2)]`, with alternating row signs.
    """
    n = c.shape[0]
    shifted = jnp.concatenate((jnp.zeros_like(c[:1]), c[:0:-1]), axis=0)
    rows = jnp.where(jnp.arange(n) % 2 == 0, 1.0, -1.0)
    return _dct3_scaled(shifted) * (rows[:, None] if c.ndim == 2 else rows)


def _dlt_ndct_legpts(c_cheb):
    """Source 18-term `ndct_legpts` expansion at Gauss-Legendre points."""
    from chebfunjax.utils.quadrature import legpts

    c0 = jnp.asarray(c_cheb)
    vector_input = c0.ndim == 1
    c = c0[:, None] if vector_input else c0
    n = c.shape[0]
    _x, _w, _v, theta = legpts(n, newtheta=True)
    t_leg = theta[::-1]
    t_cheb = (jnp.arange(n, dtype=jnp.float64) + 0.5) * jnp.pi / n
    dt = t_leg - t_cheb
    # Literal MATLAB mirrored-angle correction, with its descending LHS
    # assignment represented as a reversed RHS into the contiguous tail.
    dt = dt.at[n // 2:].set(-dt[: (n + 1) // 2][::-1])
    dt = dt[:, None]
    nn = jnp.arange(n, dtype=jnp.float64)[:, None]

    values = _dct3_scaled(c)
    current = c
    dt_power = jnp.ones_like(dt)
    factorial = 1.0
    active = jnp.asarray(True)
    eps = jnp.asarray(jnp.finfo(jnp.float64).eps)
    for ell in range(1, 18):
        current = jnp.where(active, nn * current, current)
        dt_power = dt_power * dt
        factorial *= ell
        transform = _dst3_shifted(current) if ell % 2 else _dct3_scaled(current)
        sign = -1.0 if ((ell + 1) // 2) % 2 else 1.0
        delta = (sign / factorial) * dt_power * transform
        values = values + jnp.where(active, delta, jnp.zeros_like(delta))
        # MATLAB adds the first sub-eps term; only subsequent terms stop.
        row_inf = jnp.max(jnp.sum(jnp.abs(delta), axis=1))
        active = active & (row_inf >= eps)

    values = values[::-1, :]
    all_real = jnp.all(jnp.imag(current) == 0)
    all_pure_imag = jnp.all(jnp.real(current) == 0)
    if jnp.iscomplexobj(current):
        if isinstance(current, jax.core.Tracer):
            values = jnp.where(
                all_real,
                jnp.real(values).astype(values.dtype),
                jnp.where(
                    all_pure_imag,
                    jnp.imag(values).astype(values.dtype),
                    values,
                ),
            )
        elif bool(all_real):
            values = jnp.real(values)
        elif bool(all_pure_imag):
            values = jnp.imag(values)
    return values[:, 0] if vector_input else values


def idlt(v):
    """Inverse discrete Legendre transform (MATLAB ``chebfun.idlt``).

    Uses the source direct recurrence below 5000 rows and the source
    NDCT-transpose/Leg2Cheb pipeline above that threshold.

    Provenance
    ----------
    MATLAB source : @chebfun/idlt.m, local idlt_direct and ndct_transpose
    Chebfun commit: 7574c77
    """
    from chebfunjax.utils.transforms import _legendre_idlt

    return _legendre_idlt(jnp.asarray(v))
