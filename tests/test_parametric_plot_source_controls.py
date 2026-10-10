"""Analytic source plotData controls, Chebfun7574c77."""
import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from chebfunjax import Chebfun, chebfun
from chebfunjax.chebfun1d.chebfun import _Piece
from chebfunjax.domain import Domain
from chebfunjax.plotting import matlab_plot


@pytest.fixture(autouse=True)
def close():
    yield
    plt.close("all")

def test_common_oversampling_degree():
    f = chebfun(lambda x: x)
    coeffs = jnp.zeros(61).at[-1].set(1)
    g = Chebfun(funs=[_Piece.from_coeffs(coeffs, -1., 1.)], domain=Domain((-1.,1.)))
    _, ax = matlab_plot(f, g)
    n = int(np.floor(4*np.pi*61 + .5))
    x = -np.cos(np.pi*np.arange(n)/(n-1))
    line = ax.lines[0]
    assert len(line.get_xdata()) == n+1
    np.testing.assert_allclose(line.get_xdata()[1:], x, rtol=0, atol=3e-14)
    np.testing.assert_allclose(line.get_ydata()[1:], np.cos(60*np.arccos(x)), rtol=0, atol=3e-12)
    np.testing.assert_allclose(ax.get_xlim(), [-1,1], rtol=0, atol=1e-14)

@pytest.mark.parametrize("markers", [False, True])
def test_overlap_grids_and_markers(markers):
    f = chebfun(lambda x: x, domain=(-1,0,1))
    g = chebfun(lambda x: x*x, domain=(-1,.25,1))
    _, ax = matlab_plot(f,g, **({"marker":"o"} if markers else {}))
    line = ax.lines[0]
    x,y = np.asarray(line.get_xdata()),np.asarray(line.get_ydata())
    assert len(x)==1506
    np.testing.assert_array_equal(np.flatnonzero(np.isnan(x)), [0,502,1004])
    np.testing.assert_allclose(y[~np.isnan(x)], x[~np.isnan(x)]**2, rtol=0, atol=2e-14)
    if markers:
        points=next(line for line in ax.lines if line.get_marker()=="o")
        assert len(points.get_xdata()) == 12

def test_held_manual_limits_union_with_parametric_extents():
    f=chebfun(lambda x: 3*x)
    _,ax=plt.subplots()
    ax.set_xlim(-2,2)
    ax.set_ylim(-1,1)
    matlab_plot(f,2*f,ax=ax)
    np.testing.assert_allclose(ax.get_xlim(),[-3,3],rtol=0,atol=1e-14)
    np.testing.assert_allclose(ax.get_ylim(),[-6,6],rtol=0,atol=1e-14)


def test_first_kind_limits_include_endpoints():
    from chebfunjax.tech.chebtech import Chebtech1
    f=Chebfun(funs=[_Piece(tech=Chebtech1(coeffs=jnp.array([0.,1.])),interval=(-1.,1.))],domain=Domain((-1.,1.)))
    _,ax=matlab_plot(f,f)
    np.testing.assert_array_equal(ax.get_xlim(),[-1.,1.])
    assert np.nanmax(ax.lines[0].get_xdata()) < 1


def test_arrowplot_uses_parametric_source_limits():
    from chebfunjax.plotting import arrowplot
    t=chebfun(lambda x: x)
    _,ax=arrowplot(3*t,6*t)
    np.testing.assert_allclose(ax.get_xlim(),[-3.,3.],rtol=0,atol=1e-14)
