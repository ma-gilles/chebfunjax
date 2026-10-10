"""Source plotcoeffs regression policies and rendering adapters.

Provenance
----------
MATLAB source : @chebfun/plotcoeffs.m, @classicfun/plotcoeffs.m,
    @singfun/plotcoeffs.m, @chebtech/plotcoeffs.m, @trigtech/plotcoeffs.m
Chebfun commit: 7574c77
"""
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np  # uses-numpy: assertions on Matplotlib host rendering data
import pytest

import chebfunjax as cj

_ORACLE = json.loads((Path(__file__).parent / "fixtures" /
                      "plotcoeffs_source_oracle_2025b.json").read_text())
_CASES = {case["name"]: case for case in _ORACLE["cases"]}


def _assert_axes(ax, case):
    assert ax.get_title() == case["title"]
    assert ax.get_xlabel() == case["xlabel"]
    assert ax.get_ylabel() == case["ylabel"]
    assert ax.get_xscale() == case["xscale"]
    assert ax.get_yscale() == case["yscale"]
    assert all(line.get_visible() for line in ax.get_xgridlines())
    assert all(line.get_visible() for line in ax.get_ygridlines())


def test_cheb_coeffs_and_legacy_default_keep_source_and_adapter_contracts():
    case = _CASES["cheb_coeffs"]
    f = cj.chebfun.from_coeffs(jnp.asarray([1.0, 0.5, 0.125]))
    fig, ax = cj.plotcoeffs(f, source=True, color=(0.066, 0.443, 0.745))
    assert np.array_equal(np.asarray(ax.lines[0].get_xdata()), case["lines"][0]["x"])
    assert np.array_equal(np.asarray(ax.lines[0].get_ydata()), case["lines"][0]["y"])
    _assert_axes(ax, case)
    assert np.allclose(ax.lines[0].get_color(), (0.066, 0.443, 0.745), rtol=0, atol=0)
    assert ax.lines[0].get_marker() == case["lines"][0]["marker"]
    assert ax.lines[0].get_markersize() == pytest.approx(case["lines"][0]["markersize"], rel=8 * np.finfo(float).eps, abs=0.0)
    plt.close(fig)
    fig, ax = cj.plotcoeffs(f)
    assert ax.get_title() == "Chebyshev coefficients"
    plt.close(fig)


def test_trig_odd_even_modes_follow_source_domain_and_fixture():
    for name, n, domain in (("trig_even", 4, (-np.pi, np.pi)),
                            ("trig_odd", 5, (-4.0, 4.0))):
        case = _CASES[name]
        vals = jnp.asarray([1.0, 2.0, 3.0, 4.0] if n == 4 else [1.0, 2.0, 3.0, 4.0, 5.0])
        f = cj.chebfun(vals, domain=domain, coeffs=True, trig=True)
        fig, ax = cj.plotcoeffs(f, source=True)
        assert np.array_equal(np.asarray(ax.lines[0].get_xdata()), case["lines"][0]["x"])
        assert np.array_equal(np.asarray(ax.lines[0].get_ydata()), case["lines"][0]["y"])
        _assert_axes(ax, case)
        assert ax.lines[0].get_markersize() == pytest.approx(case["lines"][0]["markersize"], rel=8 * np.finfo(float).eps, abs=0.0)
        plt.close(fig)


def test_trig_loglog_preserves_source_split_and_outer_xscale():
    case = _CASES["trig_even_loglog"]
    f = cj.chebfun(jnp.asarray([1.0, 2.0, 3.0, 4.0]), domain=(-np.pi, np.pi),
                   coeffs=True, trig=True)
    fig, ax = cj.plotcoeffs(f, source=True, loglog=True)
    x = np.asarray(ax.lines[0].get_xdata())
    y = np.asarray(ax.lines[0].get_ydata())
    assert np.array_equal(np.isnan(x), [False, False, True, False, False, False])
    assert np.array_equal(np.isnan(y), [False, False, True, False, False, False])
    assert np.array_equal(x[~np.isnan(x)], case["lines"][0]["x"][:2] + case["lines"][0]["x"][3:])
    assert np.array_equal(y[~np.isnan(y)], case["lines"][0]["y"][:2] + case["lines"][0]["y"][3:])
    _assert_axes(ax, case)
    plt.close(fig)


def test_zero_constant_array_columns_and_custom_color():
    zero = cj.chebfun.from_coeffs(jnp.asarray([0.0]))
    const = cj.chebfun.from_coeffs(jnp.asarray([3.0]))
    fig0, ax0 = cj.plotcoeffs(zero, source=True)
    fig3, ax3 = cj.plotcoeffs(const, source=True, color=(0.1, 0.2, 0.3))
    assert np.asarray(ax0.lines[0].get_ydata())[0] == np.finfo(float).eps
    assert np.asarray(ax3.lines[0].get_ydata())[0] == 3.0
    assert np.allclose(ax3.lines[0].get_color(), (0.1, 0.2, 0.3), rtol=0, atol=0)
    plt.close(fig0)
    plt.close(fig3)
    coeffs = jnp.asarray([[1.0, 2.0], [0.0, 0.0], [1.0, 2.0]])
    f = cj.chebfun(coeffs, coeffs=True)
    fig, ax = cj.plotcoeffs(f, source=True)
    assert len(ax.lines) == 2
    assert ax.lines[0].get_color() != ax.lines[1].get_color()
    plt.close(fig)
    mixed = cj.chebfun(jnp.asarray([[0.0, 3.0], [0.0, 0.0], [0.0, 1.0]]), coeffs=True)
    fig, ax = cj.plotcoeffs(mixed, source=True)
    assert np.all(np.asarray(ax.lines[0].get_ydata()) == np.finfo(float).eps)
    assert np.array_equal(np.asarray(ax.lines[1].get_ydata()), [3.0, 0.0, 1.0])
    plt.close(fig)


def test_piecewise_curves_reuse_column_color_and_empty_is_empty_plot():
    f = cj.chebfun(lambda x: jnp.where(x < 0.0, 1.0 + x, 2.0 - x),
                   domain=(-2.0, 0.0, 3.0))
    fig, ax = cj.plotcoeffs(f, source=True)
    assert len(ax.lines) == len(f.funs)
    assert ax.lines[0].get_color() == ax.lines[1].get_color()
    plt.close(fig)
    from chebfunjax.chebfun1d.chebfun import Chebfun
    fig, ax = cj.plotcoeffs(Chebfun.empty(), source=True)
    assert len(ax.lines) == 1 and len(ax.lines[0].get_xdata()) == 0
    plt.close(fig)


def test_barplot_and_short_format_color_policy():
    with pytest.raises(ValueError, match="at most one"):
        cj.plotcoeffs(cj.chebfun.from_coeffs(jnp.asarray([1.0, 0.5])),
                      source=True, fmt="rr.")
    fig, ax = cj.plotcoeffs(cj.chebfun.from_coeffs(jnp.asarray([1.0, 0.5])),
                            source=True, fmt="r.", color=(0.1, 0.2, 0.3))
    assert ax.lines[0].get_color() == "r"
    plt.close(fig)


def test_singfun_coefficients_forward_to_smooth_part():
    from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
    from chebfunjax.domain import Domain
    from chebfunjax.fun.singfun import Singfun
    from chebfunjax.tech.chebtech import Chebtech2
    # Independent cubic smooth factor; singular weighting is not displayed.
    smooth = Chebtech2.from_coeffs(jnp.asarray([1.0, -0.5, 0.0, 0.125]))
    singular = Singfun(smoothPart=smooth, exponents=(-1.0, 0.0))
    f = Chebfun(funs=[_Piece(tech=singular, interval=(-2.0, 1.0))],
                domain=Domain((-2.0, 1.0)))
    fig, ax = cj.plotcoeffs(f, source=True)
    assert np.array_equal(np.asarray(ax.lines[0].get_ydata()), [1.0, 0.5, 0.0, 0.125])
    plt.close(fig)



@pytest.mark.parametrize("hold", [False, True])
def test_source_signed_trig_hold_state_retains_or_clears_existing_artist(hold):
    # MATLAB trigtech only resets signed limits when initial ishold is false.
    fig, ax = plt.subplots()
    prior, = ax.plot([-5.0, 5.0], [1.0, 2.0])
    ax.set_xlim(-9.0, 9.0)
    f = cj.chebfun(jnp.asarray([1.0, 2.0, 3.0, 4.0]), coeffs=True,
                   trig=True, domain=(0.0, 2.0 * jnp.pi))
    cj.plotcoeffs(f, ax=ax, source=True, hold=hold)
    assert len(ax.lines) == (2 if hold else 1)
    assert (prior in ax.lines) == hold
    assert ax.get_xlim() == ((-9.0, 9.0) if hold else (-2.0, 2.0))
    assert ax.get_yscale() == ("linear" if hold else "log")
    plt.close(fig)


def test_source_chebyshev_each_piece_extends_previously_manual_limits():
    # The first tech's xlim includes degree-count2; hold is then on. The
    # independent T99 polynomial's next tech extends to100, not autoscale103.95.
    from chebfunjax.chebfun1d.chebfun import Chebfun
    from chebfunjax.domain import Domain
    first = Chebfun.from_coeffs(jnp.asarray([1.0, 0.5]), (-1.0, 0.0))
    second_coeffs = jnp.zeros(100).at[0].set(1.0).at[-1].set(0.25)
    second = Chebfun.from_coeffs(second_coeffs, (0.0, 1.0))
    f = Chebfun(funs=first.funs + second.funs, domain=Domain((-1.0, 0.0, 1.0)))
    fig, ax = cj.plotcoeffs(f, source=True)
    assert len(ax.lines) == 2
    assert ax.get_xlim() == (0.0, 100.0)
    plt.close(fig)


def test_original_eleven_source_smoke_assertions_through_source_adapter(monkeypatch):
    # Exercise the unchanged MATLAB port's11 assertions through this source
    # renderer, including array/quasimatrix/pieces/trig/loglog and line styles.
    import importlib.util

    import chebfunjax.plotting as plotting
    path = Path(__file__).parent / "test_matlab_port/chebfun/test_plotcoeffs_matlab.py"
    spec = importlib.util.spec_from_file_location("_source_plotcoeffs_original", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = plotting.plotcoeffs
    def source_renderer(*args, **kwargs):
        return original(*args, source=True, **kwargs)
    monkeypatch.setattr(plotting, "plotcoeffs", source_renderer)
    module.TestChebfunPlotcoeffs().test_all_matlab_assertions()


def test_legacy_default_coefficient_data_envelope_and_color_stay_unchanged():
    f = cj.chebfun.from_coeffs(jnp.asarray([1.0, -0.5, 0.125]))
    fig, ax = cj.plotcoeffs(f)
    assert len(ax.lines) == 2
    for line in ax.lines:
        assert np.array_equal(line.get_xdata(), [0, 1, 2])
        assert np.array_equal(line.get_ydata(), [1.0, 0.5, 0.125])
        assert line.get_color() == "#0072BD"
    assert ax.lines[0].get_marker() == "."
    assert ax.lines[0].get_markersize() == 4
    assert ax.lines[1].get_alpha() == 0.3
    assert ax.get_xlabel() == "degree $n$"
    assert ax.get_ylabel() == "$|a_n|$"
    plt.close(fig)
