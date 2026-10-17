"""Source-contract controls for spincomp's comparison adapter."""

import importlib

import jax.numpy as jnp

spincomp_module = importlib.import_module("chebfunjax.operators.spincomp")
from chebfunjax.operators.spinop import Spinop
from chebfunjax.operators.spinop2 import Spinop2
from chebfunjax.operators.spinop3 import Spinop3
from chebfunjax.operators.spinpref import SpinPref, SpinPref2, SpinPref3


def test_spincomp_reference_order_shape_normalization_and_plot(monkeypatch):
    calls = []
    plotted = []

    def fake_solver(S, dim, N, dt, scheme, pref, *, timing=True):
        calls.append((dim, N, dt, scheme, timing))
        return (scheme, dt), 1.0

    def fake_sample(solution, grid, dim):
        scheme, dt = solution
        if scheme == "pecec736":
            return [jnp.asarray([2.0])]
        values = {
            ("a", 0.1): 3.0,
            ("b", 0.1): 1.0,
            ("a", 0.2): 4.0,
            ("b", 0.2): 0.0,
        }
        return [jnp.asarray([values[(scheme, dt)]])]

    monkeypatch.setattr(spincomp_module, "_solver", fake_solver)
    monkeypatch.setattr(spincomp_module, "_sample", fake_sample)
    monkeypatch.setattr(
        spincomp_module, "_plot_results",
        lambda *args: plotted.append(args),
    )

    op = Spinop("AC")
    op.tspan = (0.0, 1.0)
    pref = SpinPref(scheme=["a", "b"])
    errors, elapsed = spincomp_module.spincomp(op, 16, [0.1, 0.2], pref)

    assert calls == [
        (1, 16, 0.05, "pecec736", False),
        (1, 16, 0.1, "a", True),
        (1, 16, 0.1, "b", True),
        (1, 16, 0.2, "a", True),
        (1, 16, 0.2, "b", True),
    ]
    assert errors.shape == (2, 2)
    assert jnp.array_equal(errors, jnp.asarray([[0.5, 0.5], [1.0, 1.0]]))
    assert elapsed.shape == (2, 2)
    assert jnp.array_equal(elapsed, jnp.ones((2, 2)))
    assert len(plotted) == 1


def test_spincomp_solver_routing_preserves_preferences(monkeypatch):
    routed = []

    def fake_spin(S, N, dt, **kwargs):
        routed.append((1, N, dt, kwargs))
        return "one", 0.1

    def fake_spin2(S, N, dt, pref, **kwargs):
        routed.append((2, N, dt, pref, kwargs))
        return "two", 0.2

    def fake_spin3(S, N, dt, pref, **kwargs):
        routed.append((3, N, dt, pref, kwargs))
        return "three", 0.3

    monkeypatch.setattr(spincomp_module, "spin", fake_spin)
    monkeypatch.setattr(spincomp_module, "spin2", fake_spin2)
    monkeypatch.setattr(spincomp_module, "spin3", fake_spin3)

    pref1 = SpinPref(M=19, dealias="on")
    pref2 = SpinPref2(M=21, dealias="on")
    pref3 = SpinPref3(M=23, dealias="on")
    assert spincomp_module._solver(Spinop("AC"), 1, 8, 0.1, "pecec736", pref1) == ("one", 0.1)
    assert spincomp_module._solver(Spinop2(), 2, 8, 0.1, "pecec736", pref2) == ("two", 0.2)
    assert spincomp_module._solver(Spinop3(), 3, 8, 0.1, "pecec736", pref3) == ("three", 0.3)

    assert routed[0][3] == {
        "scheme": "pecec736", "M": 19, "dealias": False,
        "return_computing_time": True,
    }
    assert routed[1][3].scheme == "pecec736"
    assert routed[1][3].plot == "off"
    assert routed[1][3].M == 21
    assert routed[1][3].dealias == "off"
    assert routed[1][4] == {"return_computing_time": True}
    assert routed[2][3].scheme == "pecec736"
    assert routed[2][3].plot == "off"
    assert routed[2][3].M == 23
    assert routed[2][3].dealias == "off"
    assert routed[2][4] == {"return_computing_time": True}
    assert pref2.scheme == "etdrk4" and pref3.scheme == "etdrk4"


def test_spincomp_plot_has_two_comparison_panels():
    import matplotlib.pyplot as plt

    figure = spincomp_module._plot_results(
        jnp.asarray([0.1, 0.2]),
        jnp.asarray([[0.01, 0.02], [0.03, 0.04]]),
        jnp.asarray([[1e-5, 2e-5], [3e-5, 4e-5]]),
        ["etdrk4", "krogstad"],
        1.0,
    )
    try:
        left, right = figure.axes
        assert len(left.lines) == 2 and len(right.lines) == 2
        assert left.get_xlabel() == "Relative time-step"
        assert left.get_ylabel() == "Relative error at t = 1.000"
        assert right.get_xlabel() == "Computer time (s)"
        assert right.get_ylabel() == "Relative error at t = 1.000"
    finally:
        plt.close(figure)


def test_spincomp_zero_error_plot_uses_native_lower_bound():
    """MATLAB's log10(0) path yields the explicit 1e-12 lower error limit."""
    import matplotlib.pyplot as plt

    figure = spincomp_module._plot_results(
        jnp.asarray([0.1, 0.2]),
        jnp.asarray([[0.01], [0.02]]),
        jnp.zeros((2, 1)),
        ["etdrk4"],
        1.0,
    )
    try:
        error_axis, runtime_axis = figure.axes
        assert error_axis.get_ylim() == (1e-12, 1.0)
        assert runtime_axis.get_ylim() == (1e-12, 1.0)
    finally:
        plt.close(figure)


def test_spincomp_nan_detection_visits_all_dimension_representations():
    contains_nan = spincomp_module._contains_nan
    import jax.numpy as jnp

    from chebfunjax.chebfun1d.chebfun import Chebfun, Domain, _Piece
    from chebfunjax.chebfun2d.chebfun2 import Chebfun2
    from chebfunjax.chebfun2d.separable_approx import SeparableApprox
    from chebfunjax.chebfun3d.chebfun3 import Chebfun3
    from chebfunjax.tech.chebtech import Chebtech2

    coeffs = Chebtech2(jnp.asarray([1.0, jnp.nan]))
    finite_coeffs = Chebtech2(jnp.asarray([1.0, 2.0]))
    one_d = Chebfun([_Piece(coeffs, (-1.0, 1.0))], Domain((-1.0, 1.0)))
    two_d = Chebfun2(SeparableApprox(
        [finite_coeffs], [coeffs], jnp.asarray([1.0]),
        (-1.0, 1.0, -1.0, 1.0)))
    three_d = Chebfun3(
        [finite_coeffs], [finite_coeffs], [finite_coeffs],
        jnp.asarray([[[jnp.nan]]]), (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0))
    finite = Chebfun(
        [_Piece(finite_coeffs, (-1.0, 1.0))], Domain((-1.0, 1.0)))
    assert contains_nan(one_d)
    assert contains_nan(two_d)
    assert contains_nan(three_d)
    assert not contains_nan(finite)


def test_spincomp_plot_omits_nan_errors_in_limits():
    """Native R2025b probe: min([NaN,.1]) and max([NaN,.1]) both equal .1."""
    import matplotlib.pyplot as plt

    figure = spincomp_module._plot_results(
        jnp.asarray([0.1, 0.2]),
        jnp.asarray([[0.01], [0.02]]),
        jnp.asarray([[jnp.nan], [0.1]]),
        ["etdrk4"],
        1.0,
    )
    try:
        for axis in figure.axes:
            assert axis.get_ylim() == (1e-10, 1.0)
            assert jnp.isnan(axis.lines[0].get_ydata()[0])
    finally:
        plt.close(figure)
