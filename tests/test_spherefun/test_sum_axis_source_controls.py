# uses-numpy: Independent analytic reference values and numerical assertions.
"""Independent public sum/sum2 controls for Chebfun 7574c77.

Analytic coefficient fixtures implement the source even-cosine contraction;
these are independent controls, not substituted original source assertions.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.tech.trigtech import Trigtech


def factors(cols, rows=None, pivots=None, plus=None):
    cols = [Trigtech.from_coeffs(jnp.asarray(c), is_real=True) for c in cols]
    if rows is None:
        rows = [[1.]] * len(cols)
    rows = [Trigtech.from_coeffs(jnp.asarray(r), is_real=True) for r in rows]
    if pivots is None:
        pivots = [1.] * len(cols)
    return Spherefun(cols=cols, rows=rows, pivots=jnp.asarray(pivots),
                     idx_plus=tuple(range(len(cols))) if plus is None else plus,
                     idx_minus=())


def test_axis_api_and_mean_retention():
    # cos(2 theta) carried solely by the unsplit even Nyquist coefficient.
    f = factors([[1., 0., 0., 0.]])
    lam = jnp.array([-jnp.pi, -.5, 0., .8, jnp.pi])
    theta = jnp.array([0., .3, jnp.pi / 2, 2.7, jnp.pi])
    first, second = f.sum(), f.sum(2)
    assert first.is_transposed and not second.is_transposed
    np.testing.assert_allclose(first(lam), -2./3., rtol=0, atol=8*np.finfo(float).eps)
    np.testing.assert_allclose(second(theta), 2*np.pi*jnp.cos(2*theta), rtol=0, atol=5e-14)
    np.testing.assert_allclose(f.mean()(lam), first(lam)/np.pi, rtol=0, atol=2e-15)
    np.testing.assert_allclose(f.mean(2)(theta), second(theta)/(2*np.pi), rtol=0, atol=5e-15)


def test_sum2_even_nyquist_and_common_length():
    # Integral cos(2 theta) plus constant 3 is 4*pi*(-1/3+3).
    # Different lengths force the old Nyquist slot to split exactly once.
    f = factors([[1., 0., 0., 0.], [0., 0., 0., 3., 0., 0., 0.]])
    expected = 4*np.pi*(3.-1./3.)
    assert abs(float(f.sum2())-expected) < 32*np.finfo(float).eps*abs(expected)
    assert abs(float(f.mean2())-(3.-1./3.)) < 16*np.finfo(float).eps


def test_sum2_plus_selection_and_cdr_zero_reciprocal():
    f = factors([[1.], [1.], [1.]], pivots=[0., 2., 1.], plus=(0, 1))
    # Source cdr zero reciprocal is replaced by zero; non-plus excluded.
    # MATLAB7574c77 default SUM reduces the single frequency row [2,2]
    # across both plus factors to4 before broadcasting against [0,1/2].
    # Fresh public CDR probe gives4*pi; the old2*pi reference wrongly
    # integrated these factors independently. Keep the original bound.
    assert abs(float(f.sum2())-4*np.pi) < 8*np.finfo(float).eps*np.pi
    assert float(factors([[1.]], plus=()).sum2()) == 0.


def test_empty_and_invalid_dimension_order():
    empty = Spherefun.empty()
    assert np.asarray(empty.sum(7)).size == 0
    empty_mean = empty.mean(7)
    assert isinstance(empty_mean, Chebfun) and empty_mean.isempty()
    assert float(empty.sum2()) == 0.
    with pytest.raises(ValueError, match='CHEBFUN:SPHEREFUN:sum:unknown'):
        factors([[1.]]).sum(7)
