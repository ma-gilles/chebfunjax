"""Source logical Chebfun operations.

Provenance
----------
MATLAB @chebfun/and.m, or.m, logical.m; corresponding classicfun/singfun/chebtech methods;
@chebtech/and.m, @trigtech/and.m and @chebfun/domainCheck.m.
Chebfun commit: 7574c77
"""

import math

import jax.numpy as jnp


def _binary_logical(f, g, operation, name):
    """Split at source roots, combine FUNs and stored values, then merge."""
    from chebfunjax.chebfun1d.chebfun import Chebfun

    if f.isempty() or g.isempty():
        return Chebfun.empty()
    if f.size() != g.size():
        raise ValueError(f"CHEBFUN:CHEBFUN:{name}:dims: Dimensions of f and g must agree.")
    f_ends = (f.domain.a, f.domain.b)
    g_ends = (g.domain.a, g.domain.b)
    scales = [max(abs(x) for x in ends) for ends in (f_ends, g_ends)]
    scale = max(1.0 if math.isinf(x) else x for x in scales)
    differences = [x - y for x, y in zip(f_ends, g_ends)]
    if not all(math.isnan(x) or abs(x) < 1e-15 * scale for x in differences):
        raise ValueError(f"CHEBFUN:CHEBFUN:{name}:doms: Domains of f and g must agree.")
    f, g = Chebfun._overlap(f.addBreaksAtRoots(), g.addBreaksAtRoots())
    pieces = []
    for left, right in zip(f.funs, g.funs):
        a = getattr(left.tech, "smoothPart", left.tech)
        b = getattr(right.tech, "smoothPart", right.tech)
        values = operation(a(jnp.asarray(0.1273881594)), b(jnp.asarray(0.1273881594)))
        coeffs = jnp.asarray(values, dtype=jnp.float64)
        coeffs = coeffs.reshape((1,)) if coeffs.ndim == 0 else coeffs[None, :]
        pieces.append(left.with_tech(type(a).from_coeffs(coeffs)))
    out = Chebfun(funs=pieces, domain=f.domain)
    out = out.set_point_values(operation(f.point_values, g.point_values).astype(jnp.float64))
    return Chebfun._as_transposed(out.merge(), f.is_transposed)


def logical_and(f, g):
    """Source logical AND.

    Provenance
    ----------
    MATLAB source : @chebfun/and.m
    Chebfun commit: 7574c77
    """
    return _binary_logical(f, g, jnp.logical_and, "and")


def logical_or(f, g):
    """Source logical OR.

    Provenance
    ----------
    MATLAB source : @chebfun/or.m
    Chebfun commit: 7574c77
    """
    return _binary_logical(f, g, jnp.logical_or, "or")


def logical(f):
    """Convert FUNs and stored breakpoint values to logical values.

    Provenance
    ----------
    MATLAB source : @chebfun/logical.m, @chebtech/logical.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.chebfun1d.chebfun import Chebfun

    if f.isempty():
        return f
    f = f.addBreaksAtRoots()
    pieces = []
    for piece in f.funs:
        tech = getattr(piece.tech, "smoothPart", piece.tech)
        values = tech.coeffs2vals(tech.coeffs)
        coefficients = jnp.asarray(jnp.any(values != 0, axis=0), dtype=jnp.float64)
        coefficients = (
            coefficients.reshape((1,)) if coefficients.ndim == 0 else coefficients[None, :]
        )
        pieces.append(piece.with_tech(type(tech).from_coeffs(coefficients)))
    out = Chebfun(funs=pieces, domain=f.domain).set_point_values(f.point_values)
    tolerance = out.vscale * jnp.finfo(jnp.float64).eps
    out = out.set_point_values((jnp.abs(f.point_values) > tolerance).astype(jnp.float64))
    return Chebfun._as_transposed(out, f.is_transposed)
