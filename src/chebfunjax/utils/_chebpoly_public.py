"""Public Chebfun construction for the native CHEBPOLY entry point."""

import jax.numpy as jnp

from chebfunjax.chebpref import ChebfunPref


def chebpoly_public(n, domain=None, kind=1):
    """Construct native first/second-kind Chebfuns with degree orientation.

    Provenance
    ----------
    MATLAB source: chebpoly.m, @ultraS/convertmat.m.
    Chebfun commit: 7574c77.
    Python one-dimensional degrees denote MATLAB row vectors.
    Coefficient helper utils.polynomials.chebpoly is a separate API.
    """
    from chebfunjax.chebfun1d.chebfun import chebfun

    degrees = jnp.asarray(n)
    if degrees.ndim > 2 or (degrees.ndim == 2 and min(degrees.shape) > 1):
        raise ValueError("CHEBFUN:chebpoly:integern: The first argument must be a vector of nonnegative integers.")
    if not jnp.issubdtype(degrees.dtype, jnp.number) or jnp.iscomplexobj(degrees):
        raise ValueError("CHEBFUN:chebpoly:integern: The first argument must be a vector of nonnegative integers.")
    if bool(jnp.any((degrees < 0) | (degrees % 1 != 0))):
        raise ValueError("CHEBFUN:chebpoly:integern: The first argument must be a vector of nonnegative integers.")
    transpose = degrees.ndim == 2 and degrees.shape[0] > 1
    degrees = degrees.reshape(-1).astype(jnp.int64)
    if domain is None:
        domain = ChebfunPref().domain
    elif jnp.asarray(domain).size == 1:
        kind = float(jnp.asarray(domain).reshape(()))
        domain = ChebfunPref().domain
    if kind not in (1, 2):
        raise ValueError("CHEBFUN:chebpoly:kind: CHEBPOLY(N, KIND) only supports KIND = 1 or KIND = 2.")
    domain = jnp.asarray(domain)
    if bool(jnp.any(jnp.isinf(domain))):
        raise ValueError("CHEBFUN:chebpoly:infdomain: Chebyshev polynomials are not defined over an unbounded domain.")
    if degrees.size == 0:
        return chebfun([])
    count = int(jnp.max(degrees)) + 1
    coefficients = jnp.zeros((count, degrees.size), dtype=jnp.float64)
    # Literal c(n+1,:)=eye(M), including last-write duplicate row ownership.
    for column in range(degrees.size):
        coefficients = coefficients.at[degrees[column], :].set(jnp.eye(degrees.size)[column])
    if kind == 2:
        # Inverse ultraS.convertmat(N,0,0): parity-preserving cumulative sums.
        for parity in (0, 1):
            values = coefficients[parity::2]
            coefficients = coefficients.at[parity::2].set(
                2 * jnp.cumsum(values[::-1], axis=0)[::-1]
            )
        coefficients = coefficients.at[0].multiply(.5)
    result = chebfun(coefficients[:, 0] if degrees.size == 1 else coefficients, domain=(float(domain[0]), float(domain[-1])), coeffs=True)
    if domain.size > 2:
        result = result.restrict(domain)
    return result.transpose() if transpose else result
