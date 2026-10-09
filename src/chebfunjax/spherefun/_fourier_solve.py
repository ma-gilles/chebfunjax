"""JAX banded Fourier solves for sphere Poisson and Helmholtz equations.

Provenance
----------
Chebfun 7574c77680d7e82b79626300bf255498271a72df:
@spherefun/poisson.m and @spherefun/helmholtz.m. Preserve the source equations,
actual Fourier bands and integral constraint; factorization is JAX band LU
plus compact bordered QR, rather than MATLAB sparse backslash.
"""
import warnings

import jax.numpy as jnp

from ._bordered_qr import solve_bordered
from ._fourier_bands import build_source_operators
from ._nonzero_modes import solve_nonzero_modes


def _solve_modes(operators, matrix, shifts, rhs, integral):
    indices, columns, singular, nonfinite = solve_nonzero_modes(
        matrix.ab, shifts, rhs, lower=matrix.lower, upper=matrix.upper,
        zero_longitude=operators.zero_longitude)
    if bool(jnp.any(singular)):
        raise ValueError('Singular sphere Fourier system in a nonzero longitude mode')
    if bool(jnp.any(nonfinite)):
        raise ValueError('Nonfinite sphere Fourier solve in a nonzero longitude mode')
    zero, status = solve_bordered(
        matrix.ab, rhs[:, operators.zero_longitude], operators.weights, integral,
        lower=matrix.lower, upper=matrix.upper, removed_row=operators.zero_latitude)
    if bool(status.rank_deficient | status.constraint_singular):
        raise ValueError('Singular sphere Fourier system with integral constraint')
    if bool(status.nonfinite):
        raise ValueError('Nonfinite sphere Fourier solve with integral constraint')
    coefficients = jnp.zeros(rhs.shape, dtype=jnp.result_type(columns, zero))
    return coefficients.at[:, indices].set(columns).at[:, operators.zero_longitude].set(zero)


def poisson_coefficients(coefficients, *, mean_tolerance=None):
    """Return source mean-zero solution coefficients and removed RHS mean."""
    coefficients = jnp.asarray(coefficients)
    if coefficients.ndim != 2:
        raise ValueError('Sphere coefficient matrix must have two dimensions')
    operators = build_source_operators(*coefficients.shape)
    rhs, mean, integral = operators.poisson_rhs(coefficients)
    if mean_tolerance is not None and bool(jnp.abs(mean) > mean_tolerance):
        warnings.warn(
            'CHEBFUN:SPHEREFUN:POISSON:meanRHS: The integral of the right '
            'hand side may not be zero, which is required for there to '
            'exist a solution to the Poisson equation. Subtracting the '
            'mean off the right hand side now.', stacklevel=2)
    return (_solve_modes(operators, operators.laplace, operators.dlambda2, rhs, integral),
            mean)


def helmholtz_coefficients(coefficients, k):
    """Return source solution coefficients; caller owns K=0/eigenvalue checks."""
    coefficients = jnp.asarray(coefficients)
    if coefficients.ndim != 2:
        raise ValueError('Sphere coefficient matrix must have two dimensions')
    operators = build_source_operators(*coefficients.shape)
    matrix, shifts, rhs, integral = operators.helmholtz(coefficients, k)
    return _solve_modes(operators, matrix, shifts, rhs, integral)
