"""Bounded scalar FUN division with native boundary-root cancellation.

Provenance
----------
MATLAB source: @chebfun/rdivide.m, @classicfun/rdivide.m,
    @fun/extractBoundaryRoots.m, @singfun/rdivide.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
"""
import jax.numpy as jnp


def bounded_scalar_quotient(numerator, denominator):
    """Return a source quotient, or None outside bounded scalar Chebyshev FUNs.

    Source root insertion precedes overlap. Boundary-root extraction retains
    each smooth numerator for the singular quotient's cancellation stage;
    no denominator root multiplicity is imposed on the final quotient.

    Provenance
    ----------
    MATLAB source: @chebfun/rdivide.m, @classicfun/rdivide.m,
        @fun/extractBoundaryRoots.m, @singfun/rdivide.m.
    Chebfun commit: 7574c77.
    """
    from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
    from chebfunjax.fun.singfun import Singfun
    from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
    from chebfunjax.utils._rdivide_point_values import divide_point_values

    objects = [denominator]
    if isinstance(numerator, Chebfun):
        objects.append(numerator)
    elif jnp.ndim(numerator) != 0:
        return None
    for f in objects:
        if f.deltas:
            return None
        if not bool(jnp.all(jnp.isfinite(jnp.asarray(f.domain.breakpoints)))):
            return None
        if f.n_columns != 1 or not all(
                type(p) is _Piece and isinstance(p.tech, (Chebtech1, Chebtech2))
                for p in f.funs):
            return None
    if len({type(p.tech) for f in objects for p in f.funs}) != 1:
        return None
    g = denominator.addBreaksAtRoots()
    if isinstance(numerator, Chebfun):
        f, g = Chebfun._overlap(numerator, g)
    else:
        f = numerator
    pieces = []
    eps = jnp.finfo(jnp.float64).eps
    for k, pg in enumerate(g.funs):
        gt = pg.tech
        ends = gt(jnp.asarray([-1., 1.]))
        if bool(jnp.any(jnp.abs(ends) < 20*eps*gt.vscale)):
            gt = Singfun(gt, (0., 0.)).extractBoundaryRoots()
        if isinstance(f, Chebfun):
            pf = f.funs[k]
            ft = pf.tech
            if isinstance(gt, Singfun):
                ft = Singfun(ft, (0., 0.))
            pieces.append(pf.with_tech(ft/gt))
        else:
            pieces.append(pg.with_tech(f/gt))
    out = Chebfun(funs=pieces, domain=g.domain)
    fv = f.point_values if isinstance(f, Chebfun) else jnp.asarray(f)
    gv = g.point_values
    pv = divide_point_values(fv, gv)
    if isinstance(f, Chebfun):
        fallback = ((jnp.isnan(pv) & ~(jnp.isnan(fv) | jnp.isnan(gv)))
                    | (jnp.isinf(pv) & (jnp.abs(fv) < 100*eps*f.vscale)))
        if bool(jnp.any(fallback)):
            nodes = jnp.asarray(out.domain.breakpoints)
            average = (out._feval_side(nodes, "left")
                       + out._feval_side(nodes, "right"))/2
            pv = jnp.where(fallback, average, pv)
    return out.set_point_values(pv)
