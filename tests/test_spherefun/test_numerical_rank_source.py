"""Literal source rank predicates and independent public analytic controls.

Provenance
----------
MATLAB source : @separableApprox/rank.m, @spherefun/rank.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Synthetic spectra isolate the literal predicate, never production solver data.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.spherefun._rank import rank_decision
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.tech.trigtech import Trigtech


@pytest.mark.parametrize('disable', [False, True])
@pytest.mark.parametrize('s,tol,expected,present', [
    ([], 0.0, 0, False),
    ([0.0, 0.0], 0.0, 0, True),
    ([0.0, 0.0], -1.0, 0, True),
    ([0.0, 0.0], float('nan'), 0, True),
    ([2.0, 1.0, 0.0], 0.0, 2, True),
    ([2.0, 1.0, 0.0], 0.5, 1, True),
    ([2.0, 1.0, 0.0], 1.0, 0, False),
    ([2.0, 1.0, 0.0], 2.0, 0, False),
    ([2.0, 1.0, 0.0], -0.1, 3, True),
    ([2.0, 1.0, 0.0], float('inf'), 0, False),
    ([2.0, 1.0, 0.0], float('nan'), 0, False),
    ([2.0, 1e-30, 0.0], 0.0, 2, True),
], ids=['empty', 'zero', 'zero-negative', 'zero-nan', 'positive',
        'strict-boundary', 'unit', 'above-unit', 'negative', 'infinite',
        'nan', 'tiny-positive'])
def test_literal_source_predicate(disable, s, tol, expected, present):
    with jax.disable_jit(disable):
        actual, has_result = jax.jit(rank_decision)(jnp.asarray(s), jnp.asarray(tol))
    assert int(actual) == expected
    assert bool(has_result) is present


def _field(coefficients):
    cols = [Trigtech.from_coeffs(jnp.asarray(c), is_real=True) for c in coefficients]
    row = Trigtech.from_coeffs(jnp.asarray([1.0]), is_real=True)
    return Spherefun(cols=cols, rows=[row] * len(cols), pivots=jnp.ones(len(cols)),
                     idx_plus=tuple(range(len(cols))), idx_minus=())


@pytest.mark.parametrize('disable', [False, True])
def test_zero_storage_and_spectral_rank_are_distinct(disable):
    f = _field([[0.0]])
    with jax.disable_jit(disable):
        assert int(f.numerical_rank()) == 0
    assert f.rank == len(f) == 1  # Preserve legacy storage-count property.


@pytest.mark.parametrize('disable', [False, True])
def test_redundant_factors_with_explicit_relative_tolerance(disable):
    f = _field([[1.0], [2.0]])
    with jax.disable_jit(disable):
        assert int(f.numerical_rank(1e-12)) == 1
    assert f.rank == len(f) == 2
    # Do NOT assert default rank==1: source tol=0 counts roundoff-positive
    # singular values of a redundant finite-precision QR/SVD as nonzero.


@pytest.mark.parametrize('disable', [False, True])
def test_public_empty_and_no_selected_spectrum(disable):
    with jax.disable_jit(disable):
        np.testing.assert_array_equal(Spherefun.empty().numerical_rank(), [])
        np.testing.assert_array_equal(_field([[1.0]]).numerical_rank(1.0), [])


@pytest.mark.parametrize('disable', [False, True])
@pytest.mark.parametrize('state', ['empty', 'zero'])
@pytest.mark.parametrize('tolerance', [1 + 2j, [0.0, 1.0]], ids=['complex', 'vector'])
def test_source_early_returns_do_not_inspect_tolerance(disable, state, tolerance):
    f = Spherefun.empty() if state == 'empty' else _field([[0.0]])
    with jax.disable_jit(disable):
        actual = f.numerical_rank(tolerance)
    if state == 'empty':
        np.testing.assert_array_equal(actual, [])
    else:
        assert int(actual) == 0
