"""Analytic exact-repeat controls for the Chebfun residue wrapper.

Provenance
----------
MATLAB source : ratinterp.m, trigratinterp.m, @chebfun/residue.m
Chebfun commit: 7574c77
Exact-equal computed roots only; builtin near-root clustering is unqualified.
"""
import jax.numpy as jnp
import numpy as np  # uses-numpy: independent polynomial operands/assertions
import pytest

from chebfunjax.utils.ratapprox import (
    _chebyshev_to_descending_polynomial,
    _source_residues_at_exact_roots,
    _trigrat_source_poly_residues,
    _trigrat_source_x_roots,
)


@pytest.mark.parametrize('point', [0., .5+.25j])
@pytest.mark.parametrize('multiplicity', [2, 3])
def test_general_taylor_first_residue(point, multiplicity):
    # p/q = A/(s-z) + B/(s-z)^2 [+ C/(s-z)^3].
    # Source wrapper emits A in every slot, not [A,B,C].
    A, B, C = 1.+2j, -3.+.5j, 2.-1j
    q = np.poly([point]*multiplicity)
    if multiplicity == 2:
        p = np.array([A, B-A*point])
    else:
        p = np.array([A, B-2*A*point, A*point**2-B*point+C])
    poles = jnp.full((multiplicity,), point, dtype=jnp.complex128)
    actual = _source_residues_at_exact_roots(p, q, poles)
    np.testing.assert_allclose(actual, A, rtol=0, atol=32*np.finfo(float).eps*abs(A))


def test_mixed_repeated_and_simple_groups():
    # 1/[s^2(s-2)] = -.25/s -.5/s^2 + .25/(s-2).
    poles = jnp.array([2., 0., 0.], dtype=jnp.complex128)
    actual = _source_residues_at_exact_roots([1.], [1., -2., 0., 0.], poles)
    np.testing.assert_array_equal(actual, [.25, -.25, -.25])


@pytest.mark.parametrize('p,expected', [([1.], 0.), ([1., 2.], 1.)])
def test_source_wrapper_exact_zero_double_pole(p, expected):
    poles, residues = _trigrat_source_poly_residues(jnp.array(p), jnp.array([1., 0., 0.]))
    np.testing.assert_array_equal(poles, [0., 0.])
    np.testing.assert_array_equal(residues, [expected, expected])


def test_six_and_seven_output_coefficient_interpretation_stays_distinct():
    # Fourier q=2*exp(-i*pi*x) has no finite x poles (six outputs).
    # @trigtech/poly hands [2,0,0] unchanged to residue: polynomial 2*s^2.
    bc = jnp.array([2., 0., 0.])
    six = _trigrat_source_x_roots(bc, -1., 1.)
    seven, residues = _trigrat_source_poly_residues(jnp.array([1.]), bc)
    assert six.size == 0
    np.testing.assert_array_equal(seven, [0., 0.])
    np.testing.assert_array_equal(residues, [0., 0.])


def test_chebyshev_basis_conversion_for_residue():
    # 2*T3 + (1+i)*T2 - 3*T1 + 4 = 8*x^3+(2+2i)*x^2-9*x+3-i.
    actual = _chebyshev_to_descending_polynomial(jnp.array([4., -3., 1.+1j, 2.]))
    np.testing.assert_array_equal(actual, [8., 2.+2j, -9., 3.-1j])
