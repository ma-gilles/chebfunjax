"""Port of MATLAB Chebfun tests/chebop/test_quiver.m.

The MATLAB test only checks that nothing crashes for each of the four
shapes of call. We keep those four as-is and then check the field values
themselves, since a quiver plot that runs but points the wrong way would
pass the MATLAB test unnoticed.

``@chebop/quiver.m`` supports three shapes of problem: a second-order
scalar ODE (phase plane ``(u, u')``), a coupled pair of first-order
equations (plane ``(u, v)``), and a first-order scalar (a *slope field*
on ``(t, u)``, cf. Chebfun issue #2238), plus the options ``xpts``,
``ypts`` (default 20 each), ``normalize`` (false) and ``scale`` (1).
Only the second-order case was implemented here, and the order sniffer
feeding it raised AttributeError on any operator containing an
elementwise call -- so ``y'' + sin(y)``, the very first case the MATLAB
test exercises, could not even be classified.

Provenance
----------
MATLAB source : tests/chebop/test_quiver.m, @chebop/quiver.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from chebfunjax.operators.chebop import Chebop


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


class TestChebopQuiver:
    """The four calls of tests/chebop/test_quiver.m."""

    def test_pendulum_second_order_default_options(self):
        N = Chebop(lambda t, y: y.diff(2) + y.sin(),
                   domain=(0, 10 * np.pi))
        assert N.quiver([-2, 2, -1, 1]) is not None

    def test_pendulum_with_more_options(self):
        N = Chebop(lambda t, y: y.diff(2) + y.sin(),
                   domain=(0, 10 * np.pi))
        assert N.quiver([0, 1, 0, 2], xpts=25, ypts=25, scale=0.4,
                        normalize=True) is not None

    def test_lotka_volterra_first_order_coupled_system(self):
        N = Chebop(lambda t, u, v: [u.diff() - 2 * u + u * v,
                                    v.diff() + v - u * v], domain=(0, 4))
        assert N.quiver([0, 2, 0, 4], normalize=True, scale=.5,
                        linewidth=2) is not None

    def test_slopefield_for_a_first_order_problem(self):
        N = Chebop(lambda t, u: u.diff() - t.sin() * u)
        assert N.quiver([-1.2 * np.pi, 1.2 * np.pi, -1, 1]) is not None

    # The two commented-out cases of the MATLAB test: quiver must reject
    # anything above a second-order scalar or two first-order equations.
    def test_third_order_ode_is_rejected(self):
        N = Chebop(lambda t, y: y.diff(3) + y.sin(),
                   domain=(0, 10 * np.pi))
        with pytest.raises(ValueError, match="order"):
            N.quiver([-2, 2, -1, 1])

    def test_second_order_coupled_system_is_rejected(self):
        N = Chebop(lambda t, u, v: [u.diff(2) - 2 * u + u * v,
                                    v.diff() + v - u * v], domain=(0, 4))
        with pytest.raises(ValueError):
            N.quiver([-2, 2, -1, 1])


class TestQuiverFieldValues:
    """The field itself, which the MATLAB crash test cannot see."""

    def test_second_order_scalar_phase_plane(self):
        # van der Pol: plane (u, v = u'), field (v, 3(1-u^2)v - u)
        N = Chebop(lambda t, u: u.diff(2) - 3 * (1 - u**2) * u.diff() + u,
                   domain=(0, 100))
        q = N.quiver([-2.0, 2.0, -3.0, 3.0], xpts=5, ypts=5).collections[0]
        X, Y = np.asarray(q.X), np.asarray(q.Y)
        assert q.U == pytest.approx(Y.ravel(), abs=1e-9)
        assert q.V == pytest.approx((3 * (1 - X**2) * Y - X).ravel(),
                                    rel=1e-8, abs=1e-8)

    def test_coupled_first_order_system_phase_plane(self):
        # Lotka-Volterra: u' = u - uv, v' = -v + uv
        N = Chebop(lambda t, u, v: [u.diff() - u + u * v,
                                    v.diff() + v - u * v], domain=(0, 10))
        q = N.quiver([0.5, 3.0, 0.5, 3.0], xpts=4, ypts=4).collections[0]
        X, Y = np.asarray(q.X).ravel(), np.asarray(q.Y).ravel()
        assert q.U == pytest.approx(X - X * Y, rel=1e-8, abs=1e-8)
        assert q.V == pytest.approx(-Y + X * Y, rel=1e-8, abs=1e-8)

    def test_first_order_scalar_is_a_slope_field(self):
        # Plane is (t, u) and the field is (1, u'), so U is all ones.
        N = Chebop(lambda t, u: u.diff() - t.sin() * u, domain=(-3.8, 3.8))
        q = N.quiver([-3.0, 3.0, -1.0, 1.0], xpts=4, ypts=4).collections[0]
        assert q.U == pytest.approx(np.ones_like(q.U), abs=1e-12)


class TestQuiverOptions:
    def test_xpts_and_ypts_set_the_grid_independently(self):
        N = Chebop(lambda t, u: u.diff(2) + u.sin(), domain=(0, 50))
        q = N.quiver([-1, 1, -1, 1], xpts=7, ypts=3).collections[0]
        assert len(np.asarray(q.X).ravel()) == 21
        assert len(np.unique(np.asarray(q.X))) == 7
        assert len(np.unique(np.asarray(q.Y))) == 3

    def test_default_grid_is_twenty_by_twenty(self):
        N = Chebop(lambda t, u: u.diff(2) + u.sin(), domain=(0, 50))
        q = N.quiver([-1, 1, -1, 1]).collections[0]
        assert len(np.asarray(q.X).ravel()) == 400

    def test_normalize_makes_every_arrow_unit_length(self):
        # Away from the fixed point at the origin, where the field
        # vanishes and MATLAB's u./nrm is 0/0 too.
        N = Chebop(lambda t, u: u.diff(2) - 3 * (1 - u**2) * u.diff() + u,
                   domain=(0, 100))
        q = N.quiver([-2, 2, -3, 3], xpts=4, ypts=4,
                     normalize=True).collections[0]
        nrm = np.hypot(q.U, q.V)
        assert np.all(np.isfinite(nrm))
        assert nrm == pytest.approx(np.ones_like(nrm), abs=1e-12)

    def test_scale_stretches_the_arrows_proportionally(self):
        # MATLAB fits the longest arrow in a grid cell then stretches by
        # S. matplotlib's `scale` is the reciprocal (data units per unit
        # arrow length), so doubling S must halve it.
        N = Chebop(lambda t, u: u.diff(2) - 3 * (1 - u**2) * u.diff() + u,
                   domain=(0, 100))
        got = {}
        for sc in (0.25, 0.5, 1.0, 2.0):
            got[sc] = N.quiver([-2, 2, -3, 3], xpts=5, ypts=5,
                               normalize=True, scale=sc).collections[0].scale
        assert got[0.25] / got[0.5] == pytest.approx(2.0, rel=1e-12)
        assert got[0.5] / got[1.0] == pytest.approx(2.0, rel=1e-12)
        assert got[1.0] / got[2.0] == pytest.approx(2.0, rel=1e-12)

    def test_scale_zero_draws_the_raw_vectors(self):
        N = Chebop(lambda t, u: u.diff(2) - 3 * (1 - u**2) * u.diff() + u,
                   domain=(0, 100))
        q = N.quiver([-2, 2, -3, 3], xpts=5, ypts=5,
                     scale=0).collections[0]
        assert q.scale == pytest.approx(1.0)
        assert q.angles == "xy" and q.scale_units == "xy"

    def test_default_axis_limits_are_the_unit_square(self):
        N = Chebop(lambda t, u: u.diff(2) + u.sin(), domain=(0, 50))
        ax = N.quiver(xpts=3, ypts=3)
        assert ax.get_xlim() == (-1.0, 1.0)
        assert ax.get_ylim() == (-1.0, 1.0)


class TestOperatorOrderSniffing:
    """``_op_order`` feeds quiver; elementwise calls used to break it."""

    @pytest.mark.parametrize("op, order", [
        (lambda t, u: u.diff(2) + u.sin(), 2),
        (lambda t, u: u.diff(2) + 0.25 * u.diff() + u.sin(), 2),
        (lambda t, u: u.diff() - t.sin() * u, 1),
        (lambda t, u: u.diff(2) + u.exp(), 2),
        (lambda t, u: u.diff(3) + u.cosh(), 3),
        (lambda t, u: u.diff(2) - 3 * (1 - u**2) * u.diff() + u, 2),
    ])
    def test_order_survives_elementwise_calls(self, op, order):
        assert Chebop(op, domain=(0, 1))._op_order() == order

    def test_sniffer_does_not_masquerade_as_an_array(self):
        # _TrigX treats anything carrying 'dtype' as a plain array, so a
        # sniffer that faked it would be consumed by t.sin() * u instead
        # of getting its reflected operator called.
        from chebfunjax.operators.chebop import _OrderSniffer
        s = _OrderSniffer()
        for marker in ("dtype", "shape", "ndim", "size",
                       "__array_struct__", "__array__"):
            assert not hasattr(s, marker), marker
        assert s.sin().cos().exp() is s      # real methods still absorbed

    def test_nonlocal_terms_still_bail_out(self):
        # The absorption above must stay an ALLOWLIST. _sniff_order
        # relies on the sniffer raising for a nonlocal term so the
        # caller falls back to the general column probe; absorbing
        # everything would silently assemble u'' + cumsum(u) as a plain
        # second-order differential operator.
        from chebfunjax.operators.chebop import _OrderSniffer
        s = _OrderSniffer()
        for nonlocal_name in ("cumsum", "sum", "mean", "norm", "innerProduct"):
            with pytest.raises(AttributeError):
                getattr(s, nonlocal_name)
