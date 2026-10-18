"""Native minimax rootsdiff panel sampling and Chebyshev-U colleague roots.

Provenance
----------
MATLAB source: minimax.m, findExtrema and rootsdiff, lines 1898–1952.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
All coefficient transforms, resampling and eigenvalue arithmetic use JAX.
"""

import jax
import jax.numpy as jnp


@jax.jit
def _panel_derivative(values):
    """Transform one native panel without compiling each cropped vector size."""
    n = values.size - 1
    coefficients = jnp.fft.fft(jnp.concatenate((values[::-1], values[1:-1]))) / n
    coefficients = coefficients.at[0].divide(2)
    c = jnp.real(coefficients[: n + 1])
    derivative = c[1:] * jnp.arange(1, c.size)
    significant = jnp.abs(derivative) / jnp.linalg.norm(derivative) > 1e-14
    length = jnp.max(jnp.where(significant, jnp.arange(1, c.size), 0))
    return derivative, length, c[-1] / c[0]


@jax.jit
def _colleague_eigenvalues(cU):
    """Build and solve the source Chebyshev-U colleague for one degree."""
    off = jnp.ones(cU.size - 2) / 2
    matrix = jnp.diag(off, 1) + jnp.diag(off, -1)
    row = (-cU[1:] / cU[0] / 2).at[1].add(0.5)
    matrix = matrix.at[0, :].set(row)
    return jnp.linalg.eigvals(matrix)


def rootsdiff(values, domain, error_handle):
    """Return native derivative-root candidates, including near endpoints."""
    values = jnp.asarray(values)
    n = values.size - 1
    ratio = jnp.asarray(1.0)
    cU = jnp.empty((0,), dtype=jnp.float64)
    while float(jnp.abs(ratio)) > 1e-3 and n <= 64:
        derivative, length, next_ratio = _panel_derivative(values)
        if int(length) == 0:
            return jnp.empty((0,), dtype=jnp.float64)
        cU = derivative[: int(length)][::-1]
        if cU.size <= 1:
            return jnp.empty((0,), dtype=jnp.float64)
        ratio = next_ratio
        if float(jnp.abs(ratio)) > 1e-3:
            n *= 2
            a, b = domain
            points = (a + b) / 2 + jnp.cos(jnp.pi * jnp.arange(n, -1, -1) / n) * (b - a) / 2
            values = jnp.asarray(error_handle(points))
    if cU.size <= 1:
        return jnp.empty((0,), dtype=jnp.float64)
    if cU.size == 2:
        candidates = jnp.asarray([-cU[1] / (2 * cU[0])])
        candidates = candidates[jnp.abs(candidates) <= 1 + 1e-7]
    else:
        candidates = _colleague_eigenvalues(cU)
        keep = (jnp.abs(jnp.imag(candidates)) < 1e-5) & (jnp.abs(candidates) <= 1 + 1e-7)
        candidates = jnp.real(candidates[keep])
    a, b = domain
    return (a + b) / 2 + candidates * (b - a) / 2
