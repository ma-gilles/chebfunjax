"""Complex polynomial interpolation control for the source interp1 dependency.

This follow-up is separate from the immutable20-test focused gate. It checks
complex data preservation against a known polynomial over the full domain.

Provenance
----------
MATLAB source : @chebfun/interp1.m (interp1Poly)
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""

import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.utils.quadrature import chebpts_ab


def test_complex_degree_seven_interior_data():
    domain = (-2.0, 3.0)
    x = chebpts_ab(10, *domain)[1:-1]

    def expected(t):
        u = (2 * t - domain[0] - domain[1]) / (domain[1] - domain[0])
        return 64 * u**7 - 112 * u**5 + 56 * u**3 - 7 * u + 1j * (u**5 + 0.2)

    f = cj.Chebfun.interp1(x, expected(x), domain)
    sample = jnp.linspace(*domain, 101)
    actual = f(sample)
    assert len(f) == 8
    assert jnp.issubdtype(f.funs[0].tech.coeffs.dtype, jnp.complexfloating)
    assert float(jnp.max(jnp.abs(jnp.imag(actual)))) > 0.5
    assert float(jnp.max(jnp.abs(actual - expected(sample)))) < 1e4 * float(
        jnp.finfo(jnp.float64).eps
    )
