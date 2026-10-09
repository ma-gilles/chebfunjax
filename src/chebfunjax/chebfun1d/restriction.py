"""Source Chebfun subdomain restriction.

Provenance
----------
MATLAB source: @chebfun/restrict.m and @chebfun/overlap.m; commit 7574c77.
"""
from __future__ import annotations

import math

import jax.numpy as jnp


def restrict(f, domain):
    """Apply columnRestrict, retaining old breaks and explicit point values."""
    from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece, tweak_domain
    from chebfunjax.domain import Domain
    from chebfunjax.fun.bndfun import Bndfun
    from chebfunjax.fun.unbndfun import Unbndfun
    from chebfunjax.tech.trigtech import Trigtech

    raw = domain.breakpoints if isinstance(domain, Domain) else domain
    new = [float(x) for x in jnp.asarray(raw).reshape(-1)]
    f, new, _, _ = tweak_domain(f, new)
    new = [float(x) for x in new]
    new = [x for k, x in enumerate(new) if k == 0 or x != new[k-1]]
    if f.isempty():
        return f
    if len(new) < 2:
        return Chebfun.empty()
    if isinstance(f.funs[0].tech, Trigtech):
        transposed = f.is_transposed
        f = f.change_tech('chebtech2')
        if transposed:
            object.__setattr__(f, '_is_transposed', True)
    old = list(f.domain.breakpoints)
    hs = max(abs(old[0]), abs(old[-1]))
    if not math.isfinite(hs):
        hs = 1.0
    # Source domainCheck with a numeric endpoint vector uses the strict norm
    # comparison (Inf-Inf is NaN, so unbounded domains take the general path).
    if len(new) == 2 and max(abs(old[0]-new[0]), abs(old[-1]-new[-1])) < 1e-15*hs:
        return f
    if new[0] < old[0] or new[-1] > old[-1] or any(a > b for a, b in zip(new, new[1:])):
        raise ValueError('CHEBFUN:CHEBFUN:restrict:subdom: Not a valid subdomain.')
    breaks = sorted(set(new + [x for x in old[1:-1] if new[0] < x < new[-1]]))
    pieces = []
    for a, b in zip(breaks, breaks[1:]):
        piece = next(p for p in f.funs if p.interval[0] <= a and b <= p.interval[1])
        if isinstance(piece, Unbndfun):
            subs = piece.restrict((a, b))
            if not isinstance(subs, (list, tuple)):
                subs = [subs]
            for sub in subs:
                # Preserve the accepted finite Unbndfun -> bounded _Piece
                # protocol conversion, without rebuilding its onefun.
                if isinstance(sub, Bndfun):
                    sub = _Piece(tech=sub.onefun, interval=(sub.domain.a, sub.domain.b))
                pieces.append(sub)
        else:
            pieces.append(piece.restrict(a, b))
    out = Chebfun(funs=pieces, domain=Domain(breaks))
    values = out.point_values
    previous = f.point_values
    for k, x in enumerate(breaks):
        if x in old:
            values = values.at[k].set(previous[old.index(x)])
    out = out.set_point_values(values)
    if f.is_transposed:
        object.__setattr__(out, '_is_transposed', True)
    return out


def overlap(f, g):
    """MATLAB @chebfun/overlap.m, domainCheck.m and hscale.m (7574c77).

    Infinities denote domain endpoints, not a merge tolerance. Source uses
    hscale=1 for unbounded domains, then tweakDomain and an exact union.
    """
    from chebfunjax.chebfun1d.chebfun import tweak_domain

    if f.isempty() and g.isempty():
        return f, g
    if f.isempty() or g.isempty():
        raise ValueError('CHEBFUN:CHEBFUN:overlap:domains: Inconsistent domains; intervals do not match.')
    a = jnp.asarray([f.domain.a, f.domain.b])
    b = jnp.asarray([g.domain.a, g.domain.b])
    hf, hg = jnp.max(jnp.abs(a)), jnp.max(jnp.abs(b))
    hf, hg = jnp.where(jnp.isinf(hf), 1., hf), jnp.where(jnp.isinf(hg), 1., hg)
    delta = a-b
    if not bool(jnp.all((jnp.abs(delta) < 1e-15*jnp.maximum(hf, hg)) | jnp.isnan(delta))):
        raise ValueError('CHEBFUN:CHEBFUN:overlap:domains: Inconsistent domains; intervals do not match.')
    if f.domain.breakpoints == g.domain.breakpoints:
        return f, g
    f, g, _, _ = tweak_domain(f, g)
    domain = jnp.unique(jnp.concatenate([jnp.asarray(f.domain.breakpoints),
                                         jnp.asarray(g.domain.breakpoints)]))
    return restrict(f, domain), restrict(g, domain)
