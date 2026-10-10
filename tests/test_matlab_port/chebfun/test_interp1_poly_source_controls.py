"""Fixed-degree polynomial interpolation controls for Watson's interior nodes.

These are focused regression controls, not additional literal native cases.

Provenance
----------
MATLAB source : @chebfun/interp1.m (interp1Poly), @chebfun/polyfitL1.m
Related native tests : tests/chebfun/test_interp1.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.utils.quadrature import chebpts_ab

EPS = float(jnp.finfo(jnp.float64).eps)


def _t7(x):
    return 64 * x**7 - 112 * x**5 + 56 * x**3 - 7 * x


@pytest.mark.parametrize("domain", [(-1.0, 1.0), (-2.0, 3.0)])
def test_interior_sites_recover_degree_seven_on_full_domain(domain):
    a, b = domain
    x = chebpts_ab(10, a, b)[1:-1]
    normalized = lambda t: (2 * t - a - b) / (b - a)
    y = _t7(normalized(x))
    f = cj.Chebfun.interp1(x, y, domain)
    sample = jnp.linspace(a, b, 101)
    assert len(f) == 8
    assert (float(f.domain.a), float(f.domain.b)) == domain
    assert float(jnp.max(jnp.abs(f(sample) - _t7(normalized(sample))))) < 1e4 * EPS


def test_array_valued_interior_data_recover_full_domain():
    x = chebpts_ab(10, -1.0, 1.0)[1:-1]
    y = jnp.stack([_t7(x), x**5], axis=-1)
    f = cj.Chebfun.interp1(x, y, (-1.0, 1.0))
    sample = jnp.linspace(-1.0, 1.0, 101)
    expected = jnp.stack([_t7(sample), sample**5], axis=-1)
    assert f.n_columns == 2 and len(f) == 8
    assert float(jnp.max(jnp.abs(f(sample) - expected))) < 1e4 * EPS


def test_unsorted_interior_sites_recover_cubic():
    x = jnp.asarray([-0.8, 0.2, 0.6, -0.3])
    y = x**3 - 2 * x + 1
    f = cj.Chebfun.interp1(x, y, (-1.0, 1.0))
    sample = jnp.linspace(-1.0, 1.0, 101)
    assert len(f) == 4
    assert float(jnp.max(jnp.abs(f(sample) - (sample**3 - 2 * sample + 1)))) < 1e4 * EPS
