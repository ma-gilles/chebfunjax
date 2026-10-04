# uses-numpy: dct/idct/dst/idst retain their existing SciPy implementations
"""Discrete cosine/sine/Legendre transforms (MATLAB chebfun.dct family).

Added by Claude Fable 5 (MISSING_FEATURES named-utilities sweep).

Provenance
----------
MATLAB source : chebfun.dct / chebfun.dst / chebfun.dlt / chebfun.idlt
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from scipy.fft import dct as _sdct
from scipy.fft import dst as _sdst

__all__ = ["dct", "idct", "dst", "idst", "dlt", "idlt"]


def dct(x, kind: int = 2):
    """Discrete cosine transform, MATLAB chebfun.dct normalization
    (plain unnormalized sums, types I-IV)."""
    x = np.asarray(x, dtype=float)
    return _sdct(x, type=kind, axis=0, norm=None) / 2.0 \
        if kind in (2, 3) else _sdct(x, type=kind, axis=0, norm=None) / 2.0


def idct(x, kind: int = 2):
    """Inverse DCT (the inverse of :func:`dct` of the same kind)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    inv_kind = {1: 1, 2: 3, 3: 2, 4: 4}[kind]
    scale = {1: 2.0 / max(n - 1, 1), 2: 2.0 / n, 3: 2.0 / n,
             4: 2.0 / n}[kind]
    return _sdct(x, type=inv_kind, axis=0, norm=None) * scale / 2.0


def dst(x, kind: int = 1):
    """Discrete sine transform (unnormalized sums)."""
    x = np.asarray(x, dtype=float)
    return _sdst(x, type=kind, axis=0, norm=None) / 2.0


def idst(x, kind: int = 1):
    """Inverse DST of the same kind."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    inv_kind = {1: 1, 2: 3, 3: 2, 4: 4}[kind]
    scale = {1: 2.0 / (n + 1), 2: 2.0 / n, 3: 2.0 / n,
             4: 2.0 / n}[kind]
    return _sdst(x, type=inv_kind, axis=0, norm=None) * scale / 2.0


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
