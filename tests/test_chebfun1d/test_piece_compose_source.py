"""Bounded-piece composition contracts and independent exponential values.

Provenance: @chebfun/exp.m, @bndfun/compose.m, @chebtech/compose.m,
Chebfun7574c77. The minimum grid and absence of off-grid sampleTest are
source contracts; exponential values are checked against the defining formula.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import _Piece
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = np.finfo(float).eps


@pytest.mark.parametrize('Tech', [Chebtech1, Chebtech2])
def test_piece_composition_uses_source_grid_policy(Tech):
    tech = Tech.from_coeffs(jnp.zeros(65).at[32].set(.25).at[0].set(.1))
    piece = _Piece(tech=tech, interval=(2., 5.))
    observed = []

    def operation(values):
        observed.append(values.shape[0])
        return values

    result = piece._apply_fun(operation)
    assert observed[0] >= tech.n
    assert 2 not in observed
    assert type(result.tech) is Tech
    assert result.interval == piece.interval
    points = jnp.array([2.01, 2.37, 3.25, 4.71])
    np.testing.assert_allclose(result(points), piece(points), rtol=100*EPS, atol=100*EPS)


@pytest.mark.parametrize('Tech', [Chebtech1, Chebtech2])
def test_piece_exponential_preserves_kind_and_matches_independent_formula(Tech):
    coefficients = jnp.zeros(33).at[32].set(.25).at[0].set(.1)
    piece = _Piece(tech=Tech.from_coeffs(coefficients), interval=(2., 5.))
    result = piece.exp()
    assert type(result.tech) is Tech
    assert result.tech.ishappy
    reference_points = np.linspace(-.999, .999, 513)
    points = 3.5 + 1.5 * reference_points
    expected = np.exp(.1 + .25 * np.cos(32 * np.arccos(reference_points)))
    np.testing.assert_allclose(result(jnp.asarray(points)), expected,
                               rtol=200*EPS, atol=200*EPS)
