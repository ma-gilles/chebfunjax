"""Leading coefficient ownership in native first-order conversion.

Provenance
----------
MATLAB source : @treeVar/toFirstOrder.m, coeffFun(t, coeffArg)
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
These are source-semantic regression controls, not original test predicates.
"""
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.operators.treevar import to_first_order


def test_nonconstant_coefficient_equal_at_old_five_probes():
    x = cj.chebfun(lambda t: t, domain=(-3., 3.))
    coefficient = 2 + .01*x*(x*x - 1)*(x*x - 4)
    converted, _, _, coeffs, _ = to_first_order(
        lambda u: coefficient * u.diff(), 1., (-3., 3.))
    assert isinstance(coeffs[0], cj.Chebfun)
    assert float((coeffs[0] - coefficient).norm()) < 5e-14
    expected = 1 / (2 + .01*2.5*(2.5**2 - 1)*(2.5**2 - 4))
    assert abs(float(converted(2.5, [0])[0, 0]) - expected) < 5e-14


def test_constant_chebfun_coefficient_keeps_source_representation():
    coefficient = cj.chebfun(lambda t: 2 + jnp.zeros_like(t), domain=(-3., 3.))
    _, _, _, coeffs, _ = to_first_order(
        lambda u: coefficient * u.diff(), 1., (-3., 3.))
    assert isinstance(coeffs[0], cj.Chebfun)
    assert float((coeffs[0] - coefficient).norm()) < 5e-14


def test_negative_leading_coefficient_composes_function_leaf():
    coefficient = cj.chebfun(lambda t: 2 + t*t, domain=(-3., 3.))
    converted, _, _, coeffs, _ = to_first_order(
        lambda u: -(coefficient * u.diff()), 1., (-3., 3.))
    assert isinstance(coeffs[0], cj.Chebfun)
    assert float((coeffs[0] + coefficient).norm()) < 5e-14
    assert abs(float(converted(1.5, [0])[0, 0]) + 1 / 4.25) < 5e-14
