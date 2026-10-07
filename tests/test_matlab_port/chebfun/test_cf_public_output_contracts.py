"""Additional public-output contracts, distinct from eight literal source clauses.

Provenance
----------
MATLAB source : @chebfun/cf.m (polynomialCF/rationalCF and newDomain)
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
These analytic API controls do not replace coefficient golden comparisons.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax import cf
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.domain import Domain


@pytest.mark.parametrize("domain", [(-1., 1.), (2., 6.)], ids=["canonical", "mapped"])
def test_rational_public_normalization_and_handle(domain):
    a, b = domain
    f = Chebfun.from_function(lambda x: jnp.exp((2*x-a-b)/(b-a)),
                             domain=Domain(domain))
    p, q, r, error = cf(f, 4, 3)
    eps = jnp.finfo(jnp.float64).eps
    # Source product(x-z)/product(-z), then newDomain: midpoint value1.
    assert abs(float(q((a+b)/2))-1.) < 64*eps
    assert tuple(p.domain.breakpoints) == domain
    assert tuple(q.domain.breakpoints) == domain
    assert len(p) <= 5 and len(q) <= 4
    assert not p.is_transposed and not q.is_transposed
    qc = jnp.asarray(q.funs[0].coeffs)
    # Independent triangle inequality excludes real denominator zeros here.
    assert float(jnp.abs(qc[0])-jnp.sum(jnp.abs(qc[1:]))) > 0
    x = jnp.linspace(a, b, 41)
    expected = p(x)/q(x)
    np.testing.assert_allclose(r(x), expected, rtol=64*float(eps), atol=64*float(eps))
    assert bool(jnp.all(jnp.isfinite(r(x)))) and bool(jnp.isfinite(error))


def test_mapped_polynomial_denominator_and_error():
    f = Chebfun.from_coeffs(jnp.array([1., 2., 3.]), Domain((2., 6.)))
    p, q, r, error = cf(f, 1, 0)
    np.testing.assert_array_equal(q.funs[0].coeffs, [1.])
    np.testing.assert_array_equal(p.funs[0].coeffs, [1., 2.])
    assert error == 3.
    np.testing.assert_array_equal(r(jnp.array([2., 4., 6.])), [-1., 1., 3.])
