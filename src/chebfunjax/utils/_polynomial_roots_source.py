"""Eager polynomial-root preprocessing from MATLAB R2017a roots.m.

Algorithm adapted from MATLAB R2017a toolbox/matlab/polyfun/roots.m,
Copyright 1984–2008 The MathWorks, Inc. Original Python/JAX implementation.

The CPU eigenvalue backend remains JAX/LAPACK; this is not a native rounding
or eigenvalue-order guarantee. Dynamic trimming and overflow tests are eager.
"""
import jax.numpy as jnp


def polynomial_roots(coefficients):
    """Return roots from descending floating-point vector coefficients.

    Scalar, 1D, row and column inputs are accepted; genuine matrices are not.
    Input floating/complex dtype is retained through companion construction.
    Empty/zero-root allocations use real class(c), as in the native source;
    concatenation with nonempty eigenvalues promotes normally.
    Rejecting rank>2 is an explicit adapter restriction:
    the native validation checks only the first two dimension sizes.

    Source: R2017a toolbox/matlab/polyfun/roots.m, lines22–61: reject
    nonfinite input, strip exact leading/trailing zeros, discard relative
    leading zeros causing division overflow, then prepend zero roots to eig.
    No tolerance-based trimming, polishing or real-dtype coercion is used.
    """
    c = jnp.asarray(coefficients)
    if c.ndim > 2 or (c.ndim == 2 and c.shape[0] > 1 and c.shape[1] > 1):
        raise ValueError('Polynomial coefficients must be a vector')
    if c.dtype not in (jnp.dtype('float32'), jnp.dtype('float64'),
                       jnp.dtype('complex64'), jnp.dtype('complex128')):
        raise TypeError('Polynomial coefficients must be single or double floating/complex')
    if not bool(jnp.all(jnp.isfinite(c))):
        raise ValueError('Polynomial coefficients must be finite')
    c = c.reshape(-1)
    zero_dtype = jnp.real(c).dtype
    nonzero = jnp.flatnonzero(c != 0)
    if nonzero.size == 0:
        return jnp.empty((0,), dtype=zero_dtype)
    last = int(nonzero[-1])
    trailing = c.size-last-1
    c = c[int(nonzero[0]):last+1]
    zero_roots = jnp.zeros((trailing,), dtype=zero_dtype)
    d = c[1:]/c[0]
    while bool(jnp.any(jnp.isinf(d))):
        c = c[1:]
        d = c[1:]/c[0]
    if c.size == 1:
        return zero_roots
    companion = jnp.diag(jnp.ones((c.size-2,), dtype=c.dtype), -1)
    companion = companion.at[0, :].set(-d)
    return jnp.concatenate((zero_roots, jnp.linalg.eigvals(companion)))
