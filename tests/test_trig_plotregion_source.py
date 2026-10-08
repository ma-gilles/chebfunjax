"""Pinned @trigtech/plotregionData and physical-domain mapping controls."""
import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.plotting import plotregion


@pytest.mark.parametrize("count", [5, 6])
def test_odd_even_strip_and_auxiliary(count):
    # Nonzero full coefficients keep both odd/even degree branches exercised.
    f = cj.chebfun(jnp.arange(1., count + 1), domain=[2, 8], trig=True)
    if count % 2 == 0:
        # Source simplify returns unhappy techs unchanged; happy even techs
        # become odd after their Nyquist coefficient is split.
        from dataclasses import replace
        piece = f.funs[0]
        tech = replace(piece.tech, ishappy=False)
        f = cj.Chebfun(funs=[type(piece)(tech=tech, interval=piece.interval)], domain=f.domain)
    m = f.simplify().funs[0].tech.n
    assert m == count
    fig, ax = plotregion(f, eps=1e-12)
    n = (count - 1) / 2 if count % 2 else count / 2 - 1
    height = 3 * np.log(1e12) / (np.pi * n)
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [2, 8, np.nan, 2, 8])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [height, height, np.nan, -height, -height])
    np.testing.assert_allclose(ax.lines[1].get_xdata(), [2, 2, np.nan, 8, 8])
    assert ax.lines[1].get_linestyle() == ':'
    np.testing.assert_allclose(ax.get_xlim(), [1.7, 8.3])
    np.testing.assert_allclose(ax.get_ylim(), [-1.1 * height, 1.1 * height])
    np.testing.assert_allclose(ax.lines[2].get_xdata(), [2, 8])
    plt.close(fig)


def test_simplify_before_strip_default_epsilon():
    f = cj.chebfun(lambda x: jnp.cos(3 * jnp.pi * x), trig=True)
    fig, ax = plotregion(f)
    n = (f.simplify().funs[0].tech.n - 1) / 2
    assert abs(ax.lines[0].get_ydata()[0] - np.log(1/np.finfo(float).eps)/(np.pi*n)) < 1e-14
    plt.close(fig)


def test_constant_does_not_get_fabricated_finite_strip():
    f = cj.chebfun(lambda x: 1 + 0*x, trig=True)
    with pytest.raises(ValueError, match="limits cannot be NaN or Inf"):
        plotregion(f)
    plt.close('all')


def test_empty_source_returns_empty_line():
    fig, ax = plotregion(cj.chebfun())
    assert len(ax.lines) == 1 and len(ax.lines[0].get_xdata()) == 0
    plt.close(fig)
