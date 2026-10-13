"""Small arithmetic controls from native ``@chebfun3t`` source contracts.

These controls exercise dense coefficient storage and source dispatch without
running adaptive high-frequency composition or the full constructor battery.

Provenance
----------
MATLAB source : ``@chebfun3t/{plus,minus,uminus,mtimes,times,rdivide,
    power,domainCheck,vertscale}.m``
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d.chebfun3t import Chebfun3T


def _f(coeffs, domain=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0), vscale=1.0):
    return Chebfun3T.from_coeffs(jnp.asarray(coeffs), domain=domain,
                                 vscale=vscale)


def test_plus_pads_literal_coefficient_arrays_and_scalar_promotes_complex():
    f = _f(np.arange(1.0, 5.0).reshape((2, 2, 1)))
    g = _f(np.array([[[10.0], [20.0], [30.0]]]))
    summed = f + g
    expected = np.zeros((2, 3, 1))
    expected[:, :2, :] += np.asarray(f.coeffs)
    expected[:1, :, :] += np.asarray(g.coeffs)
    np.testing.assert_array_equal(np.asarray(summed.coeffs), expected)

    shifted = f + (2.0 + 3.0j)
    assert jnp.issubdtype(shifted.coeffs.dtype, jnp.complexfloating)
    expected_shift = np.asarray(f.coeffs, dtype=np.complex128).copy()
    expected_shift[0, 0, 0] += 2.0 + 3.0j
    np.testing.assert_array_equal(np.asarray(shifted.coeffs), expected_shift)
    assert float(shifted.vscale) == pytest.approx(abs(1.0 + 2.0 + 3.0j))


def test_unary_minus_and_scalar_multiply_preserve_source_scale_rules():
    f = _f([[[2.0]], [[-0.25]]], vscale=3.0)
    negated = -f
    np.testing.assert_array_equal(np.asarray(negated.coeffs), -np.asarray(f.coeffs))
    assert float(negated.vscale) == 3.0

    scaled = -2.0 * f
    np.testing.assert_array_equal(np.asarray(scaled.coeffs), -2.0 * np.asarray(f.coeffs))
    assert float(scaled.vscale) == 6.0


def test_reverse_power_and_empty_array_dispatch():
    constant = _f([[[3.0]]], vscale=3.0)
    result = 2.0 ** constant
    np.testing.assert_array_equal(np.asarray(result.coeffs), np.array([[[8.0]]]))

    empty = constant + []
    assert isinstance(empty, Chebfun3T) and empty.isempty()
    assert isinstance(constant ** [], Chebfun3T)
    assert (constant ** []).isempty()
    with pytest.raises(ValueError, match="Sizes are inconsistent") as exc:
        _ = constant * []
    assert getattr(exc.value, "identifier", None) == "CHEBFUN:CHEBFUN3T:mtimes:size"


def test_source_domain_rules_and_zero_divisor_identifier():
    f = _f([[[1.0]]])
    shifted_domain = _f([[[2.0]]], domain=(-1.0, 1.0 + 1e-10,
                                             -1.0, 1.0, -1.0, 1.0))
    with pytest.raises(ValueError, match="not defined on the same domain"):
        _ = f + shifted_domain
    with pytest.raises(ValueError, match="Inconsistent domains"):
        _ = f * shifted_domain
    with pytest.raises(ValueError, match="Division by zero") as exc:
        _ = f / 0.0
    assert getattr(exc.value, "identifier", None) == "CHEBFUN:CHEBFUN3T:rdivide:divByZero"
