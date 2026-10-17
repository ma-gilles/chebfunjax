"""Polynomial residue and integral-operator utilities (MATLAB residue,
fred, volt).

Added by Claude Fable 5 (MISSING_FEATURES named-utilities sweep).

Provenance
----------
MATLAB source : @chebfun/residue.m, @chebfun/fred.m, @chebfun/volt.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["poly", "residue", "fred", "volt"]


def _poly_affine_weights(n, alpha, beta):
    """Batch elementwise BNDFUN/POLY weights, retaining source operations."""
    from jax.scipy.special import gammaln

    indices = jnp.arange(n)
    powers = jnp.maximum(indices[None, :] - indices[:, None], 0)
    log_factorials = gammaln(indices + 1.)
    binom = jnp.round(jnp.exp(log_factorials[None, :]
                             - log_factorials[powers]
                             - log_factorials[:, None]))
    binom = jnp.where(indices[None, :] >= indices[:, None], binom, 0.)
    beta_powers = beta**indices
    # Preserve the source loop's scalar exponentiation, including Python
    # float arithmetic when the domain endpoints are Python floats.
    alpha_powers = jnp.asarray([alpha**j for j in range(n)])
    return binom * beta_powers[powers] * alpha_powers[:, None]


def poly(f, n=None):
    """Descending polynomial coefficients of the selected FUN (one-based).

    Provenance
    ----------
    MATLAB source: @chebfun/poly.m, @bndfun/poly.m, @chebtech/poly.m.
    Chebfun commit: 7574c77.
    Scalar output uses a Python vector; array-valued output retains rows.
    """
    import warnings

    from chebfunjax.fun.singfun import Singfun
    from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

    if n is None:
        if len(f.funs) > 1:
            warnings.warn('F has more than one FUN. Only the polynomial '
                          'coefficients of the first FUN are returned.',
                          UserWarning, stacklevel=2)
        n = 1
    index = jnp.asarray(n)
    if index.ndim != 0 or not bool(jnp.isfinite(index)) or float(index) != int(index) or int(index) < 1:
        raise ValueError('N should be a positive scalar integer.')
    if int(index) > len(f.funs):
        raise ValueError(f'Chebfun only has {len(f.funs)} FUNS')
    piece = f.funs[int(index)-1]
    a, b = piece.interval
    if not bool(jnp.isfinite(a) & jnp.isfinite(b)):
        raise ValueError('POLY does not support UNBNDFUN objects.')
    tech = piece.tech
    if isinstance(tech, Singfun):
        raise ValueError('POLY does not support SINGFUN objects.')
    out = tech.poly()
    # Native constant array-valued CHEBTECH returns its coefficient row.
    if isinstance(tech, (Chebtech1, Chebtech2)) and tech.coeffs.ndim == 2 and tech.n == 1:
        out = tech.coeffs
    scalar = out.ndim == 1
    out = jnp.atleast_2d(out)
    if a != -1.0 or b != 1.0:
        alpha, beta = 2/(b-a), -(b+a)/(b-a)
        out = out[:, ::-1]
        weights = _poly_affine_weights(out.shape[1], alpha, beta)
        for j in range(out.shape[1]):
            # Keep the source tail length and reduction order. Padding a
            # reduction to n could change its floating-point sum tree.
            bba = weights[j, j:]
            out = out.at[:, j].set(jnp.sum(out[:, j:]*bba, axis=1))
        out = out[:, ::-1]
    return out[0] if scalar else out


def _poly_chebfun(coeffs_desc, domain=(-1.0, 1.0)):
    from chebfunjax.chebfun1d.chebfun import chebfun

    coefficients = jnp.atleast_1d(jnp.asarray(coeffs_desc))
    if coefficients.size == 0:
        coefficients = jnp.zeros(1, dtype=coefficients.dtype)
    return chebfun(lambda x: jnp.polyval(coefficients, x), domain=domain,
                   n=max(coefficients.size, 1))


def residue(u, v, k=None):
    """Source partial fractions with ascending repeated-pole powers.

    Forward Chebfun inputs return residues, poles and a quotient Chebfun.
    Numeric residues/poles and optional quotient return numerator/denominator.
    Denominator/quotient domains and numeric/None inverse convenience are
    retained Python extensions; native reconstructs on its preferred domain.

    Provenance
    ----------
    MIT Chebfun source: @chebfun/residue.m, commit 7574c77.
    Numeric algorithm references: licensed MATLAB R2025b residue.m,
    mpoles.m, resi2.m, poly.m, polyder.m and polydiv.m.
    Numerical operations use JAX; grouping and output lengths are eager.
    """
    from chebfunjax.utils._residue_source import forward, inverse

    def check_function(function, check_columns=True):
        if check_columns and function.n_columns > 1:
            raise ValueError("CHEBFUN:CHEBFUN:residue:quasi: Residue does not support CHEBFUN objects with multiple columns.")
        if len(function.funs) > 1:
            raise ValueError("CHEBFUN:CHEBFUN:residue:multipleFuns: Residue does not support CHEBFUNs consisting of multiple FUNs.")
        if not function.isfinite():
            raise ValueError("CHEBFUN:CHEBFUN:residue:inf: Residue does not support unbounded functions.")

    for argument in (u, v):
        if not hasattr(argument, "funs"):
            shape = jnp.asarray(argument).shape
            if len(shape) > 1 and min(shape) > 1:
                raise ValueError("CHEBFUN:CHEBFUN:residue:quasi: Residue does not support objects with multiple columns.")

    if hasattr(u, "funs"):
        check_function(u)
        check_function(v)
        residues, poles, quotient = forward(poly(u), poly(v))
        domain = (float(v.domain.a), float(v.domain.b))
        return residues, poles, _poly_chebfun(quotient, domain)
    if k is None or (hasattr(k, "funs") and k.isempty()):
        quotient = jnp.empty(0)
        domain = (-1.0, 1.0)
    elif hasattr(k, "funs"):
        check_function(k, check_columns=False)
        quotient = poly(k)
        domain = (float(k.domain.a), float(k.domain.b))
    else:
        quotient = jnp.atleast_1d(jnp.asarray(k))
        domain = (-1.0, 1.0)
    numerator, denominator = inverse(jnp.asarray(u), jnp.asarray(v), quotient)
    return _poly_chebfun(numerator, domain), _poly_chebfun(denominator, domain)


def fred(kernel, f, onevar=None):
    """Apply the shared JAX Fredholm integral action or its AD counterpart.

    Provenance
    ----------
    MATLAB source: @chebfun/fred.m, @adchebfun/adchebfun.m (fred).
    Chebfun commit: 7574c77
    """
    from chebfunjax.operators.integral import fred as _fred
    return _fred(kernel, f, onevar)


def volt(kernel, f, onevar=None):
    """Apply the shared JAX Volterra integral action or its AD counterpart.

    Provenance
    ----------
    MATLAB source: @chebfun/volt.m, @adchebfun/adchebfun.m (volt).
    Chebfun commit: 7574c77
    """
    from chebfunjax.operators.integral import volt as _volt
    return _volt(kernel, f, onevar)
