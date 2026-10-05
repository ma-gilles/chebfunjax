"""Scratch tests for source Singfun power operations and endpoint-root handling.

Source groups 22 and 31, with independent bounded controls. Source-case
samples use NumPy RandomState/MT19937 seed 6178 as a reproducible Python
adapter; equivalence to MATLAB ``seedRNG(6178)`` is not established.

Provenance
----------
MATLAB tests : tests/chebfun/test_power.m, source groups 22, 31
MATLAB APIs  : @chebfun/power.m, @singfun/power.m,
               @singfun/extractBoundaryRoots.m,
               @singfun/simplifyExponents.m
Chebfun commit: 7574c77
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)
DOMAIN = (-2.0, 7.0)


def _source_random_points(seed=6178):
    """MATLAB local-helper formula with a Python MT19937 stream adapter."""
    dom_check = (DOMAIN[0] + 0.1, DOMAIN[1] - 0.1)
    rng = np.random.RandomState(seed)
    return (dom_check[1] - dom_check[0]) * rng.rand(100) + dom_check[0]


def _relative_sample_error(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    err = float(np.max(np.abs(actual - expected), initial=0.0))
    scale = float(np.max(np.abs(expected), initial=0.0))
    return err, scale


def _source_singular_chebfun(op, exponents, *, splitting=False):
    """Construct the source example on [-2,7] with supplied endpoint orders."""
    return cj.chebfun(
        op,
        domain=DOMAIN,
        exps=exponents,
        splitting=splitting,
    )


class TestSingfunPowerSourceCases:
    def test_source_pass22_singular_square(self):
        power = -1.5
        base = _source_singular_chebfun(
            lambda x: (x - DOMAIN[0]) ** power,
            (power, 0.0),
        )
        result = base ** 2
        x = jnp.asarray(_source_random_points())
        exact = (x - DOMAIN[0]) ** (2 * power)
        err, scale = _relative_sample_error(result(x), exact)
        assert err < 1e2 * EPS * scale


    def test_source_pass31_positive_fractional_singular_power(self):
        endpoint_power = -1.2
        exponent = 0.77
        base = _source_singular_chebfun(
            lambda x: (x - DOMAIN[0]) ** endpoint_power,
            (endpoint_power, 0.0),
            splitting=True,
        )
        result = base ** exponent
        x = jnp.asarray(_source_random_points())
        exact = (x - DOMAIN[0]) ** (endpoint_power * exponent)
        err, scale = _relative_sample_error(result(x), exact)
        assert err < 1e1 * EPS * scale


def _endpoint_root_singfun(side):
    """Low-degree independent smooth factor with one exact endpoint root."""
    if side == "left":
        smooth = Chebtech2.from_coeffs(jnp.asarray([1.0, 1.0]))
    elif side == "right":
        smooth = Chebtech2.from_coeffs(jnp.asarray([1.0, -1.0]))
    else:
        raise ValueError("side must be 'left' or 'right'")
    return Singfun(smooth, (0.0, 0.0))


def _endpoint_polynomial(side, x, power):
    factor = 1.0 + x if side == "left" else 1.0 - x
    return factor ** power


class TestSingfunPowerBoundaryRootControls:
    """Independent representation controls for the source operation order."""

    def test_power_extracts_left_or_right_root_before_scaling(self):
        x = jnp.asarray([-0.8, -0.25, 0.3, 0.85])
        for side in ("left", "right"):
            original = _endpoint_root_singfun(side)
            for power in (2.0, 0.5):
                result = original ** power
                expected = _endpoint_polynomial(side, x, power)
                expected_exponents = (
                    (2.0, 0.0) if side == "left" else (0.0, 2.0)
                ) if power == 2.0 else (
                    (0.5, 0.0) if side == "left" else (0.0, 0.5)
                )
                if power == 2.0:
                    # simplifyExponents absorbs the integer endpoint power.
                    expected_exponents = (0.0, 0.0)
                assert result.exponents == expected_exponents
                np.testing.assert_allclose(
                    np.asarray(result(x)),
                    np.asarray(expected),
                    rtol=0.0,
                    atol=(16 if power == 2.0 else 20) * EPS,
                )


    def test_power_preserves_input_singfun(self):
        original = _endpoint_root_singfun("left")
        before_exponents = original.exponents
        before_values = np.asarray(original.smoothPart(jnp.asarray([-0.7, 0.2, 0.9])))

        _ = original ** 2.0

        assert original.exponents == before_exponents
        np.testing.assert_array_equal(
            np.asarray(original.smoothPart(jnp.asarray([-0.7, 0.2, 0.9]))),
            before_values,
        )

SCALE = 1.0 + 1.0j
X = jnp.asarray([-0.85, -0.4, 0.1, 0.55, 0.9])


def _complex_endpoint_singfun(tech_cls, side):
    # Chebyshev coefficients [a, +a] encode a*(1+x); [a, -a] encode
    # a*(1-x). The root at the selected endpoint is exact in coefficient data.
    sign = 1.0 if side == "left" else -1.0
    smooth = tech_cls.from_coeffs(jnp.asarray([SCALE, sign * SCALE]))
    return Singfun(smooth, (0.0, 0.0))


@pytest.mark.parametrize("tech_cls", [Chebtech1, Chebtech2])
@pytest.mark.parametrize("side", ["left", "right"])
def test_complex_endpoint_root_fractional_power_retains_kind(tech_cls, side):
    f = _complex_endpoint_singfun(tech_cls, side)
    out = f ** 0.5
    factor = 1.0 + X if side == "left" else 1.0 - X
    expected = jnp.sqrt(SCALE * factor)

    expected_exponents = (
        (0.5, 0.0) if side == "left" else (0.0, 0.5)
    )
    assert out.exponents == expected_exponents
    assert isinstance(out.smoothPart, tech_cls)
    np.testing.assert_allclose(
        np.asarray(out(X)), np.asarray(expected), rtol=0.0, atol=20 * EPS
    )


@pytest.mark.parametrize("power", [0.5, 2.0])
def test_empty_singfun_power_preserves_empty(power):
    # Python empty-object adapter; MATLAB empty power has no fresh fixture.
    f = Singfun.empty()
    out = f ** power
    assert out.isempty()



@pytest.mark.parametrize("tech_cls", [Chebtech1, Chebtech2])
@pytest.mark.parametrize("side", ["left", "right"])
def test_complex_boundary_extraction_preserves_coefficients_and_kind(tech_cls, side):
    original = _complex_endpoint_singfun(tech_cls, side)
    before = np.asarray(original.smoothPart.coeffs).copy()
    result = original.extractBoundaryRoots()
    assert type(result.smoothPart) is tech_cls
    assert result.exponents == ((1., 0.) if side == "left" else (0., 1.))
    np.testing.assert_allclose(np.asarray(result.smoothPart.coeffs), [SCALE],
                               rtol=0., atol=20*EPS)
    np.testing.assert_array_equal(np.asarray(original.smoothPart.coeffs), before)


@pytest.mark.parametrize("tech_cls", [Chebtech1, Chebtech2])
def test_scaled_fractional_exponent_absorbs_integer_part(tech_cls):
    original = Singfun(tech_cls.from_coeffs(jnp.asarray([2.])), (.6, 0.))
    result = original ** 2
    assert result.exponents == pytest.approx((.2, 0.), rel=0., abs=EPS)
    x = jnp.asarray([-.8, -.25, .3, .85])
    expected = 4.*(1.+x)**1.2
    np.testing.assert_allclose(np.asarray(result(x)), np.asarray(expected),
                               rtol=0., atol=20*EPS)
    assert original.exponents == (.6, 0.)
    # Source times swaps a constant left tech onto the default-Tech2
    # multiplier constructed by simplifyExponents, so both inputs yield 2.
    assert type(result.smoothPart) is Chebtech2
