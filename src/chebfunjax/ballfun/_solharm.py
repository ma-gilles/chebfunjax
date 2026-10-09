"""JAX solid harmonics and source radial shell contraction.

Provenance
----------
MATLAB source : @ballfun/solharm.m, @ballfun/spherefun.m, @ballfun/constructor.m
Chebfun commit: 7574c77
Copyright 2019 by The University of Oxford and The Chebfun Developers.
"""
import jax.numpy as jnp

from chebfunjax.tech.trigtech import trig_vals2coeffs
from chebfunjax.utils.quadrature import chebpts, trigpts
from chebfunjax.utils.transforms import vals2coeffs


def _complex_coefficients(l, m):
    am = abs(m)
    if l < am:
        raise ValueError("CHEBFUN:BALLFUN:solHarm: Degree must be >= order for solid harmonic ")
    points, _weights = trigpts(2 * l + 1)
    theta = jnp.pi * points
    cosine = jnp.cos(theta)
    previous = jnp.ones_like(theta)
    for degree in range(1, am + 1):
        previous = jnp.sqrt(jnp.asarray((2 * degree + 1) / (2 * degree - (degree == 1)))) * previous
    previous_previous = jnp.zeros_like(theta)
    for degree in range(am + 1, l + 1):
        a = jnp.sqrt(jnp.asarray((4 * degree**2 - 1) / ((degree - am) * (degree + am))))
        b = jnp.sqrt(jnp.asarray((2 * degree + 1) * (degree + am - 1) * (degree - am - 1)
                                / ((degree - am) * (degree + am) * (2 * degree - 3))))
        current = a * cosine * previous - b * previous_previous
        previous_previous, previous = previous, current
    values = (-1)**am * jnp.sin(theta)**am * previous / jnp.sqrt(4 * jnp.pi)
    legendre = trig_vals2coeffs(values).ravel()
    if m < 0:
        legendre = (-1.)**m * legendre
    legendre = legendre * jnp.sqrt(jnp.asarray(2 * l + 3))
    radial = vals2coeffs(chebpts(l + 1, kind=2)**l).ravel()
    coefficients = jnp.zeros((l + 1, 2 * am + 1, 2 * l + 1), dtype=jnp.complex128)
    return coefficients.at[:, am + m, :].set(radial[:, None] * legendre[None, :])


def solid_coefficients(l, m, options=()):
    """Implement native option branching and coefficient realness predicate."""
    l, m = int(l), int(m)
    if not options:
        coefficients = _complex_coefficients(l, abs(m))
        pos = int(m >= 0)
        coefficients = coefficients.at[:, 0, :].set((-1)**(1-pos) * coefficients[:, 2*abs(m), :])
        coefficients = (-1j)**(1-pos) * coefficients / (1 + (m != 0))
    elif isinstance(options[0], str) and options[0] == "complex":
        coefficients = _complex_coefficients(l, m) / jnp.sqrt(jnp.asarray(1 + (abs(m) > 0)))
    else:
        raise ValueError("CHEBFUN:BALLFUN:solharm:input: Unrecognized inputs.")
    # Both angular dimensions are odd; literal constructor conjugacy submatrix.
    _, n, p = coefficients.shape
    check = (coefficients[:, :n//2+1, :p//2+1]
             - jnp.conj(coefficients[:, n//2:, p//2:][:, ::-1, ::-1]))
    real = bool(jnp.max(jnp.abs(check)) < 1e7 * jnp.finfo(jnp.float64).eps)
    return coefficients, real


def radial_clenshaw(coefficients, radius):
    """Literal paired clenshaw_vec, vectorized over lambda/theta columns."""
    coefficients = jnp.asarray(coefficients)
    x = 2 * jnp.asarray(radius, dtype=jnp.float64)
    bk1 = jnp.zeros(coefficients.shape[1:], dtype=coefficients.dtype)
    bk2 = jnp.zeros_like(bk1)
    n = coefficients.shape[0] - 1
    for k in range(n, 1, -2):
        bk2 = coefficients[k] + x * bk1 - bk2
        bk1 = coefficients[k-1] + x * bk2 - bk1
    if n % 2:
        bk1, bk2 = coefficients[1] + x * bk1 - bk2, bk1
    return coefficients[0] + .5 * x * bk1 - bk2
