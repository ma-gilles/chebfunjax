"""MATLAB parity tests for ``arrowplot``.

Pins the placement semantics of ``@chebfun/arrowplot.m`` (commit 7574c77):
arrowheads sit at ``linspace(a, b, multi+1)`` minus the first point, so the
default ``multi=1`` puts a single head at the *end* of the curve. An earlier
implementation drew 12 heads spread along the curve, which silently made
every replica figure that used it wrong.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.plotting import arrowplot


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _t():
    # MATLAB: t = chebfun('t', [0 6])
    return chebfun(lambda t: t, domain=(0, 6))


def test_default_is_a_single_arrowhead():
    t = _t()
    _fig, ax = arrowplot(t.sin(), t.cos())
    assert len(ax.texts) == 1


def test_single_head_sits_at_the_end_of_the_curve():
    t = _t()
    f, g = t.sin(), t.cos()
    _fig, ax = arrowplot(f, g)
    tail = np.asarray(ax.texts[0].xyann)
    assert tail == pytest.approx(
        [float(f(6.0)), float(g(6.0))], abs=1e-12)


def test_head_points_along_the_derivative():
    t = _t()
    f, g = t.sin(), t.cos()
    _fig, ax = arrowplot(f, g)
    ann = ax.texts[0]
    d = np.asarray(ann.xy) - np.asarray(ann.xyann)
    expect = np.array([float(f.diff()(6.0)), float(g.diff()(6.0))])
    expect = expect / np.linalg.norm(expect)
    assert d / np.linalg.norm(d) == pytest.approx(expect, abs=1e-9)


def test_ystretch_scales_only_the_vertical_slope():
    t = _t()
    f, g = t.sin(), t.cos()
    _fig, a1 = arrowplot(f, g)
    _fig, a2 = arrowplot(f, g, ystretch=2.0)
    d1 = np.asarray(a1.texts[0].xy) - np.asarray(a1.texts[0].xyann)
    d2 = np.asarray(a2.texts[0].xy) - np.asarray(a2.texts[0].xyann)
    assert d2[0] == pytest.approx(d1[0], abs=1e-15)
    assert d2[1] == pytest.approx(2.0 * d1[1], abs=1e-15)


@pytest.mark.parametrize("multi", [1, 2, 5])
def test_multi_places_that_many_heads_at_the_matlab_points(multi):
    t = _t()
    f, g = t.sin(), t.cos()
    _fig, ax = arrowplot(f, g, multi=multi)
    assert len(ax.texts) == multi
    want = np.linspace(0.0, 6.0, multi + 1)[1:]
    got = np.array([ann.xyann for ann in ax.texts])
    expect = np.column_stack([np.asarray(f(want)), np.asarray(g(want))])
    assert got == pytest.approx(expect, abs=1e-12)


def test_complex_chebfun_plots_real_versus_imag():
    t = _t()
    h = ((-0.2 + 3j) * t).exp()
    _fig, ax = arrowplot(h)
    assert len(ax.lines) == 1 and len(ax.texts) == 1
    tail = np.asarray(ax.texts[0].xyann)
    hv = complex(h(6.0))
    assert tail == pytest.approx([hv.real, hv.imag], abs=1e-12)


def test_quasimatrix_plots_one_curve_and_one_head_each():
    t = _t()
    A = [((-0.1 * k + 1j) * t).exp() for k in (1, 2, 3)]
    _fig, ax = arrowplot(A)
    assert len(ax.lines) == 3
    assert len(ax.texts) == 3
    _fig, ax = arrowplot(A, multi=5)
    assert len(ax.texts) == 15


def test_zero_chebfun_gets_no_arrowhead():
    # MATLAB skips the annotation when the column is a zero chebfun.
    z = chebfun(lambda t: 0.0 * t, domain=(0, 6))
    _fig, ax = arrowplot(z, z)
    assert len(ax.texts) == 0


def test_mismatched_component_counts_raise():
    t = _t()
    with pytest.raises(ValueError, match="y-components"):
        arrowplot([t.sin(), t.cos()], t.cos())


def test_multi_must_be_positive():
    t = _t()
    with pytest.raises(ValueError, match="multi"):
        arrowplot(t.sin(), t.cos(), multi=0)
