"""Two residual columns distinguish MATLAB's default Frobenius subspace path.

Provenance
----------
MATLAB source : @chebfun/subspace.m, @chebfun/norm.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
These are authored analytic controls, paired to the pinned native probe;
original test_subspace.m predicates remain in their original module.
"""

import jax.numpy as jnp
import pytest

import chebfunjax as cj


@pytest.mark.parametrize('case', [
    'real_equal_ranks', 'real_smaller_left_rank',
    'complex_equal_ranks', 'real_reversed',
])
def test_native_small_angle_uses_frobenius_residual(case):
    a = [cj.chebfun(lambda x: jnp.ones_like(x) / jnp.sqrt(2.0)),
         cj.chebfun(lambda x: jnp.sqrt(1.5) * x)]
    c = [cj.chebfun(lambda x: jnp.sqrt(2.5) * (3*x*x-1)/2),
         cj.chebfun(lambda x: jnp.sqrt(3.5) * (5*x*x*x-3*x)/2)]
    angles = jnp.asarray([0.1, 0.2], dtype=jnp.float64)
    phase = 1j if case == 'complex_equal_ranks' else 1.0
    b = [a[k]*jnp.cos(angles[k]) + c[k]*(phase*jnp.sin(angles[k]))
         for k in range(2)]
    left, right = a, b
    if case == 'real_smaller_left_rank':
        d = cj.chebfun(lambda x: 3/jnp.sqrt(2.0)*(35*x**4-30*x*x+3)/8)
        right = b + [d]
    elif case == 'real_reversed':
        left, right = b, a
    expected = jnp.arcsin(jnp.sqrt(jnp.sum(jnp.sin(angles)**2)))
    actual = cj.subspace(left, right)
    assert abs(float(actual - expected)) < 1e3*jnp.finfo(jnp.float64).eps
    # The spectral formula returns 0.2, which would pass single-column tests.
    assert abs(float(actual) - 0.2) > 1e-3
