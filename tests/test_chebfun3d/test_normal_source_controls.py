"""Supplemental controls for normal and pointwise division.

Provenance
----------
MATLAB source : @chebfun3/normal.m, @chebfun3/rdivide.m, @chebfun3v/rdivide.m
Chebfun commit: 7574c77
No original normal test exists.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

DOMAIN = (0.0, 2.0, -1.0, 2.0, 3.0, 7.0)
POINT = (0.7, 0.2, 4.1)


@pytest.fixture(scope="module")
def linear():
    return Chebfun3.from_function(lambda x, y, z: x + 2 * y + 3 * z,
                                  domain=DOMAIN)


def test_default_and_unrecognized_option_return_analytic_gradient(linear):
    f = linear
    for normal in (f.normal(), f.normal("other")):
        assert normal.domain == DOMAIN
        np.testing.assert_allclose(np.asarray(normal(*POINT)), [1.0, 2.0, 3.0],
                                   rtol=0.0, atol=2e-12)


def test_unit_normal_uses_global_chebfun3v_norm_and_rdivide(linear):
    f = linear
    normal = f.normal("UnIt")
    expected = np.asarray([1.0, 2.0, 3.0]) / np.sqrt(14.0 * 24.0)
    assert normal.domain == DOMAIN
    np.testing.assert_allclose(np.asarray(normal(*POINT)), expected,
                               rtol=0.0, atol=2e-12)

    # The source vector quotient receives Chebfun3v.norm's JAX scalar.
    gradient = f.grad()
    divided = gradient.rdivide(gradient.norm())
    np.testing.assert_allclose(np.asarray(divided(*POINT)), expected,
                               rtol=0.0, atol=2e-12)


def test_empty_and_zero_normal_follow_native_branches():
    empty = Chebfun3.empty()
    default = empty.normal()
    unit = empty.normal("unit")
    assert default.isempty() and default.n_components == 3
    assert unit.isempty() and unit.n_components == 0
    assert isinstance(unit, Chebfun3v)

    constant = Chebfun3.from_function(lambda x, y, z: 5.0,
                                      domain=DOMAIN)
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN3:normal:zeroNormal"):
        constant.normal("unit")


def test_chebfun3_rdivide_accepts_zero_dimensional_jax_scalar(linear):
    f = linear
    result = f.rdivide(jnp.asarray(2.0))
    assert result.domain == DOMAIN
    np.testing.assert_allclose(float(result(*POINT)),
                               (POINT[0] + 2 * POINT[1] + 3 * POINT[2]) / 2,
                               rtol=0.0, atol=2e-12)
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN3:rdivide:divByZero"):
        f.rdivide(jnp.asarray(0.0))

    via_operator = f / jnp.asarray(2.0)
    np.testing.assert_allclose(float(via_operator(*POINT)),
                               (POINT[0] + 2 * POINT[1] + 3 * POINT[2]) / 2,
                               rtol=0.0, atol=2e-12)


def test_scalar_left_rdivide_requires_single_sign_and_no_sampled_zero():
    positive = Chebfun3.from_function(lambda x, y, z: x * x + 2.0,
                                     domain=DOMAIN)
    result = Chebfun3.rdivide(jnp.asarray(2.0), positive)
    expected = 2.0 / (POINT[0] ** 2 + 2.0)
    assert result.domain == DOMAIN
    np.testing.assert_allclose(float(result(*POINT)), expected,
                               rtol=0.0, atol=3e-11)

    sign_change = Chebfun3.from_function(lambda x, y, z: x - 1.0, domain=DOMAIN)
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN3:rdivide:zero"):
        Chebfun3.rdivide(2.0, sign_change)


def test_chebfun3_rdivide_function_branch_and_domain_check(linear):
    denominator = Chebfun3.from_function(lambda x, y, z: x * x + 2.0,
                                         domain=DOMAIN)
    quotient = linear.rdivide(denominator)
    expected = (POINT[0] + 2 * POINT[1] + 3 * POINT[2]) / (POINT[0] ** 2 + 2)
    np.testing.assert_allclose(float(quotient(*POINT)), expected,
                               rtol=0.0, atol=5e-10)
    np.testing.assert_allclose(float((linear / denominator)(*POINT)), expected,
                               rtol=0.0, atol=5e-10)

    other_domain = Chebfun3.from_function(
        lambda x, y, z: x * x + 2.0,
        domain=(0.0, 2.1, -1.0, 2.0, 3.0, 7.0))
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN3:rdivide:domains"):
        linear.rdivide(other_domain)


def test_chebfun3v_rdivide_all_operand_families(linear):
    gradient = linear.grad()
    point_gradient = np.asarray([1.0, 2.0, 3.0])
    point_scalar = POINT[0] + 2 * POINT[1] + 3 * POINT[2]

    # Vector / scalar and vector / Chebfun3 use componentwise source dispatch.
    by_scalar = gradient.rdivide(jnp.asarray(2.0))
    np.testing.assert_allclose(np.asarray(by_scalar(*POINT)),
                               point_gradient / 2.0, rtol=0.0, atol=2e-12)
    denominator = Chebfun3.from_function(lambda x, y, z: x * x + 2.0,
                                         domain=DOMAIN)
    by_function = gradient.rdivide(denominator)
    np.testing.assert_allclose(np.asarray(by_function(*POINT)),
                               point_gradient / (POINT[0] ** 2 + 2),
                               rtol=0.0, atol=5e-10)

    # Vector / vector and numeric / vector exercise both remaining source legs.
    doubled = gradient * 2.0
    by_vector = gradient.rdivide(doubled)
    np.testing.assert_allclose(np.asarray(by_vector(*POINT)),
                               np.full(3, 0.5), rtol=0.0, atol=2e-12)
    numeric_left = Chebfun3v.rdivide(jnp.asarray(2.0), gradient)
    np.testing.assert_allclose(np.asarray(numeric_left(*POINT)),
                               2.0 / point_gradient, rtol=0.0, atol=2e-12)
    reflected = 2.0 / gradient
    np.testing.assert_allclose(np.asarray(reflected(*POINT)),
                               2.0 / point_gradient, rtol=0.0, atol=2e-12)
    via_operator = gradient / jnp.asarray(2.0)
    np.testing.assert_allclose(np.asarray(via_operator(*POINT)),
                               point_gradient / 2.0, rtol=0.0, atol=2e-12)

    # The pinned source accidentally sets nG = F.nComponents, making its
    # unequal-size check unreachable; this fixture records that literal branch.
    short_left = Chebfun3v(linear)
    longer_right = gradient
    source_count = Chebfun3v.rdivide(short_left, longer_right)
    assert source_count.n_components == 1
    np.testing.assert_allclose(float(source_count.components[0](*POINT)),
                               point_scalar, rtol=0.0, atol=2e-12)


def test_scalar_chebfun3_left_and_numeric_empty_rdivide_branches(linear):
    gradient = linear.grad()
    numerator = Chebfun3.from_function(lambda x, y, z: 2.0 + 0 * x,
                                       domain=DOMAIN)
    quotient = Chebfun3v.rdivide(numerator, gradient)
    np.testing.assert_allclose(np.asarray(quotient(*POINT)),
                               [2.0, 1.0, 2.0 / 3.0],
                               rtol=0.0, atol=2e-12)

    empty_numeric = jnp.empty((0,), dtype=jnp.float64)
    assert Chebfun3.rdivide(empty_numeric, linear).isempty()
    assert Chebfun3.rdivide(linear, empty_numeric).isempty()
    assert Chebfun3v.rdivide(empty_numeric, gradient).n_components == 0
    assert Chebfun3v.rdivide(gradient, empty_numeric).n_components == 0
    assert (gradient / empty_numeric).n_components == 0
    assert (linear / empty_numeric).isempty()
    assert gradient.rdivide([]).n_components == 0
    assert linear.rdivide([]).isempty()

    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN3:rdivide:badInputs"):
        linear.rdivide(object())
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN3V:rdivide:inputs"):
        gradient.rdivide(object())
