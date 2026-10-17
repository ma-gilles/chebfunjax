"""Analytic controls for source scalar Chebtech physical plotting grids."""
import matplotlib

matplotlib.use('Agg')
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

import chebfunjax.plotting as P
from chebfunjax import Chebfun
from chebfunjax.chebfun1d.chebfun import _Piece
from chebfunjax.domain import Domain
from chebfunjax.fun.unbndfun import Unbndfun
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


def mapped(kind, domain):
    cls = Chebtech1 if kind == 1 else Chebtech2
    return Unbndfun.from_chebtech(cls.from_coeffs(jnp.array([1., .25])), Domain(domain))


def wrap(piece):
    return Chebfun(funs=[piece], domain=Domain(piece.interval))


@pytest.mark.parametrize('kind', [1, 2])
@pytest.mark.parametrize('domain', [(2., np.inf), (-np.inf, 4.), (-np.inf, np.inf)])
def test_unbounded_source_grid_order_map_and_jacobian(kind, domain):
    piece = mapped(kind, domain)
    x, y = P._sample_pieces(wrap(piece), numpts=17, _source_grid=True)[0]
    assert len(x) == len(y) == 501
    assert np.all(np.diff(x) > 0)
    # Recover reference coordinates independently from the physical output.
    finite = np.isfinite(x)
    xx = x[finite]
    if np.isfinite(domain[0]):
        t = (xx - domain[0] - 15) / (xx - domain[0] + 15)
    elif np.isfinite(domain[1]):
        t = (15 + xx - domain[1]) / (15 - xx + domain[1])
    else:
        t = 2 * xx / (5 + np.sqrt(25 + 4 * xx**2))
    np.testing.assert_allclose(y[finite], 1 + .25*t, rtol=0, atol=2e-14)
    if kind == 2:
        assert np.isinf(x[0]) == np.isinf(domain[0])
        assert np.isinf(x[-1]) == np.isinf(domain[1])
    t = jnp.array([-.5, 0., .5])
    _, derivative = jax.jvp(piece.forward_map, (t,), (jnp.ones_like(t),))
    if np.isfinite(domain[0]):
        expected = 30 / (1-np.asarray(t))**2
    elif np.isfinite(domain[1]):
        expected = 30 / (1+np.asarray(t))**2
    else:
        expected = 5*(1+np.asarray(t)**2)/(1-np.asarray(t)**2)**2
    np.testing.assert_allclose(derivative, expected, rtol=2e-15, atol=2e-14)
    expected_xlim = ((domain[0], domain[0]+10) if np.isfinite(domain[0])
                     else (domain[1]-10, domain[1]) if np.isfinite(domain[1])
                     else (-10., 10.))
    xlim, ylim, default = P._function_lims(wrap(piece))
    assert tuple(xlim) == expected_xlim and default
    mask = (x > domain[0]) & (x < domain[1])
    np.testing.assert_array_equal(ylim, [y[mask].min(), y[mask].max()])


def test_unbounded_extrema_use_physical_domain_not_visible_window():
    piece = mapped(2, (0., np.inf))
    x, y = P._sample_pieces(wrap(piece), _source_grid=True)[0]
    _, ylim, _ = P._function_lims(wrap(piece))
    assert ylim[1] > .99 * 1.25  # Maximum lies far beyond x=10.
    assert y[x <= 10].max() < 1.


@pytest.mark.parametrize('kind', [1, 2])
def test_bounded_piece_extrema_and_held_axes_do_not_evaluate_whole_function(kind, monkeypatch):
    cls = Chebtech1 if kind == 1 else Chebtech2
    pieces = [_Piece(cls.from_coeffs(jnp.array([3., 0., 1.])), (-2., 0.)),
              _Piece(cls.from_coeffs(jnp.array([-5., 0., -2.])), (0., 4.))]
    f = Chebfun(funs=pieces, domain=Domain((-2., 0., 4.)))
    expected = []
    for piece in pieces:
        _, y = P._sample_pieces(Chebfun(funs=[piece], domain=Domain(piece.interval)),
                                _source_grid=True)[0]
        expected.extend([y.min(), y.max()])
    def forbidden(*args, **kw):
        raise AssertionError('Source plotData reads each tech, not wholefunction feval')
    monkeypatch.setattr(Chebfun, '__call__', forbidden)
    xlim, ylim, default = P._function_lims(f)
    assert tuple(xlim) == (-2., 4.) and default
    np.testing.assert_array_equal(ylim, [min(expected), max(expected)])
    fig, ax = plt.subplots()
    ax.set(xlim=(-3., -1.), ylim=(-2., 2.))
    P.matlab_plot(f, ax=ax)
    assert ax.get_xlim() == (-3., 4.)
    np.testing.assert_array_equal(ax.get_ylim(), [min(-2., min(expected)), max(2., max(expected))])
    plt.close(fig)


def test_nan_omission_and_infinite_values_are_source_min_max_semantics():
    assert P._source_plot_value_limits(np.array([np.nan, -np.inf, 1., np.inf])) == (-np.inf, np.inf)
    assert np.isnan(P._source_plot_value_limits(np.array([np.nan]))).all()


def test_explicit_unbounded_window_retains_existing_sampling_adapter():
    piece = mapped(2, (0., np.inf))
    x, _ = P._sample_pieces(wrap(piece), numpts=17, interval=(0., 1.), _source_grid=True)[0]
    assert len(x) == 17 and x[0] > 0 and x[-1] < 1
    np.testing.assert_allclose(np.diff(x), np.diff(x)[0], rtol=2e-14, atol=0)


@pytest.mark.parametrize('coeffs', [jnp.array([1.+1j, .25]), jnp.array([[1., 2.], [.25, .5]])])
def test_complex_and_array_techs_retain_unbounded_adapter(coeffs):
    piece = Unbndfun.from_chebtech(Chebtech2.from_coeffs(coeffs), Domain((0., np.inf)))
    x, _ = P._sample_pieces(wrap(piece), numpts=17, _source_grid=True)[0]
    assert len(x) == 17 and x[0] > 0 and x[-1] < 10


@pytest.mark.parametrize('length,expected_count', [(44, 553), (6000, 65537)])
def test_unbounded_source_oversampling_and_factory_cap(length, expected_count):
    c = np.zeros(length)
    c[0], c[-1] = 1., .1
    piece = Unbndfun.from_chebtech(Chebtech2.from_coeffs(jnp.asarray(c)), Domain((0., np.inf)))
    x, y = P._sample_pieces(wrap(piece), numpts=17, _source_grid=True)[0]
    assert len(x) == len(y) == expected_count
    assert x[0] == 0 and np.isposinf(x[-1])
    assert np.isfinite(y).all()
