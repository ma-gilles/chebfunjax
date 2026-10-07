"""Shared source CDR reciprocal convention for Spherefun evaluation.

Provenance
----------
MATLAB source : @separableApprox/cdr.m, @separableApprox/feval.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
import jax.numpy as jnp


def inverse_pivots(pivots):
    """Compute source 1/p and replace only infinite reciprocal magnitudes.

    Provenance
    ----------
    MATLAB source : @separableApprox/cdr.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df

    No imaginary tolerance, rank change, finite-value cutoff or NaN cleanup.
    Differentiation with respect to singular pivots is not promised.
    """
    inverse = 1.0 / jnp.asarray(pivots)
    return jnp.where(jnp.isinf(jnp.abs(inverse)), 0, inverse)
