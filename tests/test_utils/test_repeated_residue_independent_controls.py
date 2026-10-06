"""Analytic exact-repeat controls for the Chebfun residue wrapper.

Provenance
----------
MATLAB source : ratinterp.m, trigratinterp.m, @chebfun/residue.m
Chebfun commit: 7574c77
Exact-equal computed roots only; builtin near-root clustering is unqualified.
"""

import jax.numpy as jnp
import numpy as np  # uses-numpy: independent analytic assertions
import pytest

from chebfunjax.utils.ratapprox import (
    _chebyshev_to_descending_polynomial,
    _has_exact_duplicate_roots,
    _source_residues_at_exact_roots,
)


@pytest.mark.parametrize("p", [[1.], [1., 1., -6., 0., 1.]])
@pytest.mark.parametrize("scale", [1., 2.-3j])
def test_improper_direct_term_and_common_scaling_do_not_change_residue(p, scale):
    # q=s^2(s-2); second numerator=q*(s+3)+1. The polynomial direct term
    # contributes no Laurent coefficient. Common nonzero scaling cancels.
    q = jnp.array([1., -2., 0., 0.])
    result = _source_residues_at_exact_roots(
        scale*jnp.asarray(p), scale*q, jnp.array([0., 2., 0.]))
    np.testing.assert_allclose(result, [-.25, .25, -.25], rtol=0,
                               atol=128*np.finfo(float).eps)


def test_nonadjacent_triple_group_retains_simple_pole():
    # 1/(s^3(s-2)) has first Laurent coefficients-1/8 and+1/8.
    result = _source_residues_at_exact_roots(
        [1.], [1., -2., 0., 0., 0.], jnp.array([0., 2., 0., 0.]))
    np.testing.assert_array_equal(result, [-.125, .125, -.125, -.125])


def test_physical_affine_map_scales_first_laurent_coefficient_only_once():
    # t=(s-5)/3: 1/(t^2(t-2))=27/((s-5)^2(s-11)).
    result = _source_residues_at_exact_roots(
        [27.], [1., -21., 135., -275.], jnp.array([5., 11., 5.]))
    np.testing.assert_allclose(result, [-.75, .75, -.75], rtol=0,
                               atol=128*np.finfo(float).eps)


def test_chebyshev_denominator_represents_same_repeated_rational_function():
    # q=.25*T3-T2+.75*T1-T0=s^2(s-2), independently derived.
    q = _chebyshev_to_descending_polynomial(jnp.array([-1., .75, -1., .25]))
    result = _source_residues_at_exact_roots([1.], q, jnp.array([0., 0., 2.]))
    np.testing.assert_array_equal(result, [-.25, -.25, .25])


@pytest.mark.parametrize("poles,expected", [
    ([0., 1e-12], False),
    ([1.+2j, 1.+(2.+1e-12)*1j], False),
    ([0., 1e-12, 0.], True),
    ([1.+2j, -3j, 1.+2j], True),
    ([], False),
    ([1.], False),
])
def test_exact_classifier_does_not_move_or_cluster_nearby_roots(poles, expected):
    original = jnp.asarray(poles, dtype=jnp.complex128)
    snapshot = np.asarray(original).copy()
    assert _has_exact_duplicate_roots(original) is expected
    np.testing.assert_array_equal(original, snapshot)
