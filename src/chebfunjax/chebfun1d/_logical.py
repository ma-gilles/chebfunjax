"""Source logical Chebfun operations.

Provenance
----------
MATLAB @chebfun/and.m, @classicfun/and.m, @singfun/and.m,
@chebtech/and.m, @trigtech/and.m and @chebfun/domainCheck.m; commit7574c77.
"""

import math

import jax.numpy as jnp


def logical_and(f, g):
    """Split at source roots, combine FUNs and stored values, then merge."""
    from chebfunjax.chebfun1d.chebfun import Chebfun

    if f.isempty() or g.isempty():
        return Chebfun.empty()
    if f.size() != g.size():
        raise ValueError("CHEBFUN:CHEBFUN:and:dims: Dimensions of f and g must agree.")
    f_ends = (f.domain.a, f.domain.b)
    g_ends = (g.domain.a, g.domain.b)
    scales = [max(abs(x) for x in ends) for ends in (f_ends, g_ends)]
    scale = max(1.0 if math.isinf(x) else x for x in scales)
    differences = [x - y for x, y in zip(f_ends, g_ends)]
    if not all(math.isnan(x) or abs(x) < 1e-15 * scale for x in differences):
        raise ValueError("CHEBFUN:CHEBFUN:and:doms: Domains of f and g must agree.")
    f, g = Chebfun._overlap(f.addBreaksAtRoots(), g.addBreaksAtRoots())
    pieces = []
    for left, right in zip(f.funs, g.funs):
        a = getattr(left.tech, "smoothPart", left.tech)
        b = getattr(right.tech, "smoothPart", right.tech)
        values = jnp.logical_and(a(jnp.asarray(0.1273881594)), b(jnp.asarray(0.1273881594)))
        coeffs = jnp.asarray(values, dtype=jnp.float64)
        coeffs = coeffs.reshape((1,)) if coeffs.ndim == 0 else coeffs[None, :]
        pieces.append(left.with_tech(type(a).from_coeffs(coeffs)))
    out = Chebfun(funs=pieces, domain=f.domain)
    out = out.set_point_values(jnp.logical_and(f.point_values, g.point_values).astype(jnp.float64))
    return Chebfun._as_transposed(out.merge(), f.is_transposed)
