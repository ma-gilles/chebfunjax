"""Source delta stem base, endpoint, color and derivative omission."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from chebfunjax import chebfun
from chebfunjax.plotting import matlab_plot


def test_stem_starts_at_continuous_density_and_marker_matches():
    x=chebfun('x',domain=(-1,1))
    f=2+x+3*x.dirac()
    fig,ax=plt.subplots()
    matlab_plot(f,'k',ax=ax,deltaline='r')
    stem,marker=ax.lines[-2:]
    np.testing.assert_allclose(stem.get_ydata(),[2,5],rtol=0,atol=1e-13)
    np.testing.assert_allclose(marker.get_ydata(),[5],rtol=0,atol=1e-13)
    assert to_rgba(stem.get_color())==to_rgba(marker.get_color())==to_rgba('r')
    assert to_rgba(marker.get_markerfacecolor())==to_rgba('r')
    plt.close(fig)


def test_default_delta_color_follows_function():
    x=chebfun('x',domain=(-1,1))
    fig,ax=plt.subplots()
    matlab_plot(x.dirac(),'g',ax=ax)
    assert to_rgba(ax.lines[-2].get_color())==to_rgba(ax.lines[-1].get_color())==to_rgba('g')
    plt.close(fig)


def test_delta_derivative_is_not_drawn():
    x=chebfun('x',domain=(-1,1))
    fig,ax=plt.subplots()
    matlab_plot(x.dirac(1),'k',ax=ax)
    assert all(line.get_marker() not in ('^','v') for line in ax.lines)
    plt.close(fig)


def test_negative_delta_uses_downward_marker():
    x=chebfun('x',domain=(-1,1))
    fig,ax=plt.subplots()
    matlab_plot(2-x.dirac(),'k',ax=ax)
    np.testing.assert_allclose(ax.lines[-2].get_ydata(),[2,1],rtol=0,atol=1e-13)
    assert ax.lines[-1].get_marker()=='v'
    plt.close(fig)
