"""Public Jacobi and ultraspherical polynomial constructors.

Provenance
----------
MATLAB source: jacpoly.m, ultrapoly.m
Chebfun commit: 7574c77
Copyright 2017 by The University of Oxford and The Chebfun Developers.
"""
import jax.numpy as jnp
from jax.scipy.special import gamma, gammaln

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.utils.transforms import jac2cheb


def _arguments(n, domain, name):
    degrees = jnp.asarray(n)
    if (degrees.size == 0 or jnp.iscomplexobj(degrees)
            or bool(jnp.any(~jnp.isfinite(degrees)))
            or bool(jnp.any(degrees < 0))
            or bool(jnp.any(degrees != jnp.floor(degrees)))):
        raise ValueError(f"CHEBFUN:{name}:integern: Nonnegative integer degrees required.")
    transpose = degrees.ndim == 2 and degrees.shape[0] > 1
    ns = degrees.reshape(-1).astype(jnp.int32)
    dom = tuple(float(v) for v in domain)
    if bool(jnp.any(jnp.isinf(jnp.asarray(dom)))):
        raise ValueError(f"CHEBFUN:{name}:infdomain: Unbounded domains are not supported.")
    return ns, dom, transpose


def _construct(coefficients, dom, transpose, pref=None):
    from chebfunjax.chebfun1d.chebfun import chebfun

    f = chebfun(coefficients, domain=(dom[0], dom[-1]), coeffs=True, pref=pref)
    if len(dom) > 2:
        f = f._restrict_breaks(dom)
    return f.transpose() if transpose else f


def jacpoly(n, alpha, beta, domain=(-1.0, 1.0)):
    """Return Jacobi polynomials as a Chebfun, one column per degree.

    Provenance: jacpoly.m, Chebfun commit 7574c77.
    The coefficient-vector helper remains in utils.polynomials.
    """
    ns, dom, transpose = _arguments(n, domain, 'jacpoly')
    count = int(jnp.max(ns)) + 1
    coefficients = jac2cheb(jnp.eye(count)[:, ns], alpha, beta)
    # Source computes a local preference but does not pass it to CHEBFUN.
    return _construct(coefficients, dom, transpose)


def ultrapoly(n, lam, domain=(-1.0, 1.0)):
    """Return ultraspherical polynomials as a Chebfun.

    Provenance: ultrapoly.m, Chebfun commit 7574c77.
    """
    ns, dom, transpose = _arguments(n, domain, 'ultrapoly')
    if lam <= 0:
        raise ValueError('CHEBFUN:ultrapoly:infdomain: LAMBDA must be positive.')
    if lam == .5:
        from chebfunjax.chebfun1d.chebfun import legpoly
        return legpoly(n, dom)
    if lam == 1:
        from chebfunjax.chebfun1d.chebfun import chebpoly
        return chebpoly(n, dom, 2)
    count = int(jnp.max(ns)) + 1
    nn = jnp.arange(count, dtype=jnp.float64)
    scale = gamma(lam+.5)/gamma(2*lam)*jnp.exp(gammaln(2*lam+nn)-gammaln(lam+nn+.5))
    coefficients = jac2cheb(jnp.diag(scale)[:, ns], lam-.5, lam-.5)
    pref = ChebfunPref()
    technology = pref.tech
    if isinstance(technology, str):
        pref.tech = Chebtech1 if technology.lower().lstrip('@') == 'chebtech1' else Chebtech2
    elif technology not in (Chebtech1, Chebtech2):
        pref.tech = Chebtech2
    return _construct(coefficients, dom, transpose, pref)
