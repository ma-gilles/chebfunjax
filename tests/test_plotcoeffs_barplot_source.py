"""Literal padData controls from @chebtech/plotcoeffs.m at 7574c77."""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np  # uses-numpy: Matplotlib host-output assertions
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize('cls', [Chebtech1, Chebtech2])
@pytest.mark.parametrize('loglog', [False, True])
def test_literal_bar_data_and_default_marker_size(cls, loglog):
    tech = cls(coeffs=jnp.asarray([1., -.5, 0.]), ishappy=True)
    f = Chebfun(funs=[_Piece(tech=tech, interval=(-1., 1.))],
                domain=Domain((-1., 1.)))
    fig, ax = cj.plotcoeffs(f, source=True, barplot=True, loglog=loglog)
    line = ax.lines[0]
    np.testing.assert_array_equal(line.get_xdata(), [.5, 0, 0, 1.5, .5, .5, 2.5, 1.5, 1.5])
    np.testing.assert_array_equal(line.get_ydata(), [1, 1, np.nan, .5, .5, np.nan, 0, 0, np.nan])
    assert line.get_markersize() == pytest.approx(2.5 + 50 / np.sqrt(17), rel=8*np.finfo(float).eps)
    assert ax.get_yscale() == 'log'
    assert ax.get_xscale() == ('log' if loglog else 'linear')
    plt.close(fig)


def test_strict_small_coefficient_threshold():
    eps = np.finfo(float).eps
    f = cj.chebfun.from_coeffs(jnp.asarray([1., eps/200, eps/100, eps/50]))
    assert float(f.funs[0].tech.vscale) == 1.
    fig, ax = cj.plotcoeffs(f, source=True, barplot=True, markersize=7)
    np.testing.assert_array_equal(ax.lines[0].get_ydata()[::3], [1, 0, eps/100, eps/50])
    assert ax.lines[0].get_markersize() == 7
    plt.close(fig)


def test_zero_bar_data_preserves_epsilon():
    fig, ax = cj.plotcoeffs(cj.chebfun.from_coeffs(jnp.asarray([0.])),
                            source=True, barplot=True)
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [.5, 0, 0])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(),
                                  [np.finfo(float).eps, np.finfo(float).eps, np.nan])
    plt.close(fig)
