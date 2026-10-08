"""JAX Jacobi functions, AGM descent and source imaginary transformation."""

import math

import jax
import jax.numpy as jnp


@jax.jit
def _real_ellipj(u, m, tol):
    u, m = jnp.broadcast_arrays(u, m)
    a, b = jnp.ones_like(m), jnp.sqrt(1-m)
    scale = jnp.ones_like(m)
    previous = jnp.sqrt(m)
    levels = jnp.zeros_like(m, dtype=jnp.int32)

    def ascend(state, i):
        a, b, previous, levels, scale = state
        running = jnp.any((jnp.abs(previous) > tol) & (m >= 0) & (m <= 1))
        an = (a+b)/2
        c = (a-b)/2
        crossing = (jnp.abs(previous) > tol) & (jnp.abs(c) <= tol)
        levels = jnp.where(running & crossing, i+1, levels)
        scale = jnp.where(running & crossing, 2.0**(i+1), scale)
        ratio = jnp.where(running, c/an, 0)
        return (jnp.where(running, an, a),
                jnp.where(running, jnp.sqrt(a*b), b),
                jnp.where(running, c, previous), levels, scale), ratio

    (a, _, _, levels, scale), ratios = jax.lax.scan(
        ascend, (a, b, previous, levels, scale), jnp.arange(64, dtype=jnp.int32))

    def descend(phi, item):
        ratio, i = item
        correction = jnp.arcsin(jnp.clip(ratio*jnp.sin(phi), -1, 1))
        return jnp.where(levels > i, (phi+correction)/2, phi), None

    phi, _ = jax.lax.scan(descend, scale*a*u,
                          (ratios, jnp.arange(64, dtype=jnp.int32)), reverse=True)
    sn, cn = jnp.sin(phi), jnp.cos(phi)
    dn = jnp.sqrt((1-m)+m*cn*cn)
    # Stable endpoint limits, including |u| large enough that cosh overflows.
    e = jnp.exp(-jnp.abs(u))
    sech = 2*e/(1+e*e)
    sn = jnp.where(m == 0, jnp.sin(u), jnp.where(m == 1, jnp.tanh(u), sn))
    cn = jnp.where(m == 0, jnp.cos(u), jnp.where(m == 1, sech, cn))
    dn = jnp.where(m == 0, jnp.ones_like(u), jnp.where(m == 1, sech, dn))
    valid = (m >= 0) & (m <= 1)
    return tuple(jnp.where(valid, v, jnp.nan) for v in (sn, cn, dn))


def ellipj(u, m, tol=None):
    """Return sn, cn, dn for real parameter 0 <= m <= 1.

    Uses arithmetic-geometric mean descent for real arguments and the
    source imaginary transformation for complex arguments. The independent
    builtin adapter uses absolute AGM stopping, retains the terminal
    correction and advances the arithmetic mean across the broadcast array.
    Stable dn arithmetic is retained near m=1. Invalid real
    parameters produce NaNs under JIT. Accuracy near complex poles and for
    extremely large arguments is not guaranteed.

    Provenance
    ----------
    MATLAB source: @chebfun/ellipj.m (complex transformation).
    Chebfun commit: 7574c77
    """
    tol = jnp.finfo(jnp.float64).eps if tol is None else tol
    if jnp.iscomplexobj(tol) or jnp.asarray(tol).ndim != 0:
        raise ValueError("ellipj tolerance must be a real nonnegative scalar")
    if not isinstance(tol, jax.core.Tracer) and (not math.isfinite(float(tol)) or float(tol) < 0):
        raise ValueError("ellipj tolerance must be finite and nonnegative")
    if jnp.iscomplexobj(m):
        raise ValueError("ellipj parameter must be real")
    u = jnp.asarray(u, dtype=jnp.result_type(u, jnp.float64))
    m = jnp.asarray(m, dtype=jnp.float64)
    if not jnp.iscomplexobj(u):
        return _real_ellipj(u, m, tol)
    s, c, d = _real_ellipj(jnp.real(u), m, tol)
    s1, c1, d1 = _real_ellipj(jnp.imag(u), 1-m, tol)
    denom = c1*c1+m*(s*s1)**2
    return ((s*d1+1j*c*d*s1*c1)/denom,
            (c*c1-1j*s*d*s1*d1)/denom,
            (d*c1*d1-1j*m*s*c*s1)/denom)


def _compose_ellipj(u, m, tol):
    """Source Chebfun dispatch, including its literal upper-end fudge to zero."""
    from chebfunjax.chebfun1d.chebfun import Chebfun
    from chebfunjax.chebpref import ChebfunPref

    pref = ChebfunPref(tol) if isinstance(tol, (dict, ChebfunPref)) else ChebfunPref()
    tolerance = pref.chebfuneps if tol is None or isinstance(tol, (dict, ChebfunPref)) else tol
    if isinstance(m, Chebfun):
        if not m.isreal():
            raise ValueError("ellipj parameter must be real")
        mtol = max(jnp.finfo(jnp.float64).eps*m.vscale, tolerance)

        def fudge(v):
            return jnp.where(((v < 0) & (v > -mtol)) |
                             ((v > 1) & (v < 1+mtol)), 0, v)

        if isinstance(u, Chebfun):
            return tuple(u.compose(lambda x, v, k=k: ellipj(x, fudge(v), tolerance)[k],
                                   m, pref=pref) for k in range(3))
        return tuple(m.compose(lambda v, k=k: ellipj(u, fudge(v), tolerance)[k],
                               pref=pref) for k in range(3))
    if jnp.iscomplexobj(m) or not bool(jnp.all((jnp.asarray(m) >= 0) & (jnp.asarray(m) <= 1))):
        raise ValueError("ellipj parameter must be real and lie in [0, 1]")
    return tuple(u.compose(lambda x, k=k: ellipj(x, m, tolerance)[k], pref=pref)
                 for k in range(3))
