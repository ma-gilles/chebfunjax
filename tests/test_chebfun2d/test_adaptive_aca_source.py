"""Callable construction's ACA follows pinned constructor.m selection/arithmetic.

Provenance
----------
MATLAB source: @chebfun2/constructor.m (completeACA)
Chebfun commit: 7574c77
Exact small analytic controls; no new native MATLAB capture is claimed.
"""
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d import Chebfun2
from chebfunjax.chebfun2d.separable_approx import _complete_aca


def test_column_major_tie():
    values = np.asarray([[0., 1., 0.], [1., 0., 0.]])
    _, positions, _, _, _ = _complete_aca(values, 0., 1)
    np.testing.assert_array_equal(positions, [[1, 0], [0, 1]])


def test_diagonal_selection_controls_stopping():
    # Source selects .9 on the diagonal since 1.1-.9 < absTol, then stops.
    values = np.asarray([[.9, 1.1], [0., .8]])
    pivots, _, _, _, failed = _complete_aca(values, 1., 1)
    np.testing.assert_array_equal(pivots, [0.])
    assert not failed


def test_divide_row_before_outer_product():
    # All mathematical intermediates of the source order remain finite.
    values = np.asarray([[1e200, 2e200], [3e200, 4e200]])
    pivots, _, rows, cols, _ = _complete_aca(values, 1e184, 1)
    assert np.all(np.isfinite(pivots))
    assert np.all(np.isfinite(rows)) and np.all(np.isfinite(cols))
    np.testing.assert_allclose((cols / pivots) @ rows / 1e200,
                               values / 1e200, rtol=4*np.finfo(float).eps)


def test_rank_budget_forces_refinement_at_exact_residual_zero():
    _, _, _, _, failed = _complete_aca(np.ones((4, 4)), 0., 4)
    assert failed


def test_public_large_separable_callable():
    f = Chebfun2.from_function(lambda x, y: 1e200*(x+2)*(y+3))
    x = jnp.asarray([-.5, 0., .25])
    y = jnp.asarray([.75, -.2, .3])
    assert f.rank == 1
    np.testing.assert_allclose(f(x, y)/1e200, (x+2)*(y+3),
                               rtol=32*np.finfo(float).eps)
