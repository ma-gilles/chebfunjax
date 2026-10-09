"""Analytical controls for the Chebfun3 stored-factor interpolation correction.

Polynomial reference tensors are exact. Exponential reference coefficients use
I_n(a)'s convergent power series; for |a|<=.5 the omitted tails are below1e-30.
The residual is integrated with the exact Chebyshev Gram matrix, not sampled.
The original compose source28 keeps its independent strict10eps predicate.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import _stored_factor_core, chebfun3
from chebfunjax.tech.chebtech import Chebtech2


def expcoeff(a, n):
    degree = jnp.arange(n)
    fact = jnp.concatenate((jnp.ones(1), jnp.cumprod(jnp.arange(1, n, dtype=jnp.float64))))
    term = (a / 2) ** degree / fact
    total = term
    for k in range(30):
        term = term * (a * a / 4) / ((k + 1) * (k + degree + 1))
        total = total + term
    return total.at[1:].multiply(2)


def continuous_error(g, exact):
    C = g.coeffs3()
    shape = tuple(max(a, b) for a, b in zip(C.shape, exact.shape))
    E = jnp.pad(C, tuple((0, n - k) for n, k in zip(shape, C.shape))) - jnp.pad(
        exact, tuple((0, n - k) for n, k in zip(shape, exact.shape))
    )
    grams = []
    for n in shape:
        i = jnp.arange(n)[:, None]
        j = jnp.arange(n)[None, :]

        def integral(k):
            return jnp.where(k % 2 == 0, 2 / jnp.where(k * k != 1, 1 - k * k, 1), 0.0)

        grams.append((integral(i + j) + integral(i - j)) / 2)
    square = jnp.einsum("ijk,abc,ia,jb,kc->", E, E, *grams)
    return {
        "l2": float(jnp.sqrt(jnp.maximum(square, 0))),
        "coefficient_max": float(jnp.max(jnp.abs(E))),
    }


def cases():
    zero = jnp.zeros((4, 4, 4))
    affine = zero.at[0, 0, 0].set(0.75).at[1, 0, 0].set(1).at[0, 1, 0].set(2).at[0, 0, 1].set(-3)
    mixed = (
        zero.at[0, 0, 0]
        .set(2)
        .at[1, 0, 0]
        .set(0.25)
        .at[0, 2, 0]
        .set(1)
        .at[0, 0, 1]
        .set(-0.375)
        .at[0, 0, 3]
        .set(-0.125)
        .at[1, 1, 1]
        .set(1)
    )
    product = jnp.einsum(
        "i,j,k->ijk",
        jnp.array([1.5, 1, 0.5]),
        jnp.array([2.0, -1]),
        jnp.array([1.0, 0.75, 0, 0.25]),
    )
    expo = jnp.einsum("i,j,k->ijk", expcoeff(0.5, 20), expcoeff(0.3, 20), expcoeff(-0.2, 20))
    sphere = (
        zero.at[0, 0, 0].set(1.5).at[2, 0, 0].set(0.5).at[0, 2, 0].set(0.5).at[0, 0, 2].set(0.5)
    )
    return [
        ("sphere", lambda x, y, z: x * x + y * y + z * z, sphere),
        ("rank1_polynomial", lambda x, y, z: (1 + x + x * x) * (2 - y) * (1 + z**3), product),
        ("affine", lambda x, y, z: 0.75 + x + 2 * y - 3 * z, affine),
        ("mixed", lambda x, y, z: 1 + 0.25 * x + 2 * y * y - 0.5 * z**3 + x * y * z, mixed),
        ("rank1_exponential", lambda x, y, z: jnp.exp(0.5 * x + 0.3 * y - 0.2 * z), expo),
        (
            "rank2_exponential",
            lambda x, y, z: jnp.exp(0.5 * x + 0.3 * y - 0.2 * z) + 0.125 * x * y,
            expo.at[1, 1, 0].add(0.125),
        ),
        (
            "small_component",
            lambda x, y, z: jnp.exp(0.5 * x + 0.3 * y - 0.2 * z) + 1e-12 * x * y,
            expo.at[1, 1, 0].add(1e-12),
        ),
    ]


@pytest.mark.parametrize("name,fn,exact", cases(), ids=lambda v: v if isinstance(v, str) else None)
def test_analytic_constructor(name, fn, exact):
    result = chebfun3(fn)
    error = continuous_error(result, exact)
    scale = max(1.0, float(jnp.max(jnp.abs(exact))))
    assert error["l2"] < 100 * jnp.finfo(jnp.float64).eps * scale


def test_singular_stored_basis_retains_source_core():
    zero = Chebtech2.from_coeffs(jnp.array([0.0]))
    fallback = jnp.array([[[2.0]]])
    core = _stored_factor_core(
        jnp.ones((1, 1, 1)),
        ([zero], [zero], [zero]),
        (jnp.zeros(1),) * 3,
        (jnp.ones(1),) * 3,
        fallback,
    )
    assert jnp.array_equal(core, fallback)
