"""JAX Jacobi functions, AGM descent and source imaginary transformation."""

import jax
import jax.numpy as jnp


@jax.jit
def _real_ellipj(u, m):
    u, m = jnp.broadcast_arrays(u, m)
    a, b = jnp.ones_like(m), jnp.sqrt(1-m)
    scale = jnp.ones_like(m)

    def ascend(state, _):
        a, b, scale = state
        c = (a-b)/2
        active = jnp.abs(c) > jnp.finfo(a.dtype).eps*a
        an = (a+b)/2
        ratio = jnp.where(active, c/an, 0)
        return (jnp.where(active, an, a), jnp.where(active, jnp.sqrt(a*b), b),
                jnp.where(active, 2*scale, scale)), (ratio, active)

    (a, _, scale), (ratios, active) = jax.lax.scan(ascend, (a, b, scale), None, length=16)

    def descend(phi, item):
        ratio, on = item
        phi = jnp.where(on, (phi+jnp.arcsin(jnp.clip(ratio*jnp.sin(phi), -1, 1)))/2, phi)
        return phi, None

    phi, _ = jax.lax.scan(descend, scale*a*u, (ratios, active), reverse=True)
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


def ellipj(u, m):
    """Return sn, cn, dn for real parameter 0 <= m <= 1.

    Uses arithmetic-geometric mean descent for real arguments and the
    source imaginary transformation for complex arguments. Invalid real
    parameters produce NaNs under JIT. Accuracy near complex poles and for
    extremely large arguments is not guaranteed.

    Provenance
    ----------
    MATLAB source: @chebfun/ellipj.m (complex transformation).
    Chebfun commit: 7574c77
    """
    if jnp.iscomplexobj(m):
        raise ValueError("ellipj parameter must be real")
    u = jnp.asarray(u, dtype=jnp.result_type(u, jnp.float64))
    m = jnp.asarray(m, dtype=jnp.float64)
    if not jnp.iscomplexobj(u):
        return _real_ellipj(u, m)
    s, c, d = _real_ellipj(jnp.real(u), m)
    s1, c1, d1 = _real_ellipj(jnp.imag(u), 1-m)
    denom = c1*c1+m*(s*s1)**2
    return ((s*d1+1j*c*d*s1*c1)/denom,
            (c*c1-1j*s*d*s1*d1)/denom,
            (d*c1*d1-1j*m*s*c*s1)/denom)
