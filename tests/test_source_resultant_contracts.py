"""Independent source controls for the resultant helpers.

Provenance
----------
MATLAB source: @chebfun2v/roots.m
Chebfun commit: 7574c77
"""
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.resultant import (
    _blockingxy,
    _chebT1rts,
    _chebT1rts_matgep,
    _generalized_eigvals,
    _onevar,
)


def test_coordinate_gap_groups_preserve_source_chain_rule():
    x = jnp.array([-.9, -.7, -.5, .2, .3])
    y = jnp.array([0., 1., 1.1, 0., 0.])
    groups = _blockingxy(x, y, .21)
    assert [g.tolist() for g in groups] == [[0], [1, 2], [3, 4]]
    assert _blockingxy(jnp.empty(0), jnp.empty(0), .1) == []


def test_zero_equation_matrix_reduction_uses_reduced_orientation():
    # g=T2(y)+x. Native onevar selects a representative by setting y=0,
    # yielding x=1. The mistaken unreduced matrix orientation returned(0,1).
    f = jnp.zeros((1, 1))
    g = jnp.array([[0., 1.], [0., 0.], [1., 0.]])
    x, y = _onevar(f, g, -1., 1., -1., 1.)
    np.testing.assert_allclose(x, [1.], rtol=0, atol=10*np.finfo(float).eps)
    np.testing.assert_allclose(y, [0.], rtol=0, atol=10*np.finfo(float).eps)
    assert float(jnp.max(jnp.abs(2*y*y-1+x))) < 10*np.finfo(float).eps


def test_generalized_singular_pencil_classification():
    roots = np.asarray(_generalized_eigvals(jnp.diag(jnp.array([2., 3., 0.])),
                                          jnp.diag(jnp.array([1., 0., 0.]))))
    assert np.count_nonzero(np.isnan(roots)) == 1
    assert np.count_nonzero(np.isinf(roots)) == 1
    np.testing.assert_allclose(roots[np.isfinite(roots)], [2.], rtol=0, atol=1e-14)


def test_scalar_chebyshev_pencil_and_colleague():
    # T2(x)-1/2 has roots +/-sqrt(3)/2.
    coeffs = jnp.array([1., 0., -.5])
    expected = np.array([-np.sqrt(3)/2, np.sqrt(3)/2])
    np.testing.assert_allclose(_chebT1rts(coeffs), expected, rtol=0, atol=1e-14)
    roots = np.asarray(_chebT1rts_matgep(coeffs.reshape(1, 1, 3)))
    np.testing.assert_allclose(np.sort(roots.real), expected, rtol=0, atol=1e-14)
    assert np.max(np.abs(roots.imag)) < 1e-14


def test_default_engine_selection_and_vector_forwarding(monkeypatch):
    from chebfunjax.chebfun2d import chebfun2, resultant, zerocurves
    from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

    calls = []

    def record(engine):
        def solve(f, g):
            calls.append(engine)
            return jnp.array([[.25, -.25]])
        return solve

    monkeypatch.setattr(resultant, "resultant_common_zeros", record("resultant"))
    monkeypatch.setattr(zerocurves, "common_zeros", record("ms"))
    low = chebfun2(lambda x, y: x+y)
    high = chebfun2(lambda x, y: jnp.cos(230*x)+y)
    low.roots(low)
    Chebfun2v([low.approx, low.approx]).roots()
    high.roots(low)
    low.roots(low, method="ms")
    high.roots(low, method="resultant")
    assert calls == ["resultant", "resultant", "ms", "ms", "resultant"]
