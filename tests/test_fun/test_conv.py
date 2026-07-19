"""Core tests for fun-level convolution and change-of-map (Fable 5).

Covers :meth:`Bndfun.conv`, :meth:`Bndfun.change_map`,
:meth:`Deltafun.conv`, and :meth:`Deltafun.change_map`.  These exercise the
smooth-single-interval quadrature route and the Dirac-delta translation rule
``delta^{(k)}(. - x0) * phi = phi^{(k)}(. - x0)`` independently of the
MATLAB golden-reference harness.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.deltafun import Deltafun

D1 = Domain((-1.0, 1.0))


def _max_abs_on_grid(fun, n=200) -> float:
    a, b = float(fun.domain.a), float(fun.domain.b)
    xs = jnp.asarray(np.linspace(a, b, n), dtype=jnp.float64)
    return float(jnp.max(jnp.abs(fun(xs))))


class TestBndfunChangeMap:
    def test_relocates_shape(self):
        f = Bndfun.from_function(lambda x: x ** 2, D1)
        g = f.change_map((3.0, 5.0))
        assert float(g.domain.a) == 3.0 and float(g.domain.b) == 5.0
        # G(x) = f mapped: endpoints match f(-1)=1 and f(1)=1; midpoint f(0)=0.
        assert abs(float(g(jnp.float64(3.0))) - 1.0) < 1e-12
        assert abs(float(g(jnp.float64(5.0))) - 1.0) < 1e-12
        assert abs(float(g(jnp.float64(4.0))) - 0.0) < 1e-12

    def test_change_map_accepts_domain(self):
        f = Bndfun.from_function(jnp.sin, D1)
        g = f.change_map(Domain((0.0, 2.0)))
        assert float(g.domain.a) == 0.0 and float(g.domain.b) == 2.0
        # Same underlying coefficients -> identical onefun evaluation.
        assert abs(float(g(jnp.float64(1.0))) - float(f(jnp.float64(0.0)))) < 1e-12


class TestBndfunConv:
    def test_box_box_is_triangle(self):
        # Convolution of two unit boxes on [-1, 1] is a triangular hat on
        # [-2, 2]: h(0) = 2, h(+-2) = 0, and it is piecewise linear.
        one = Bndfun.from_function(lambda x: jnp.ones_like(x), D1)
        pieces = one.conv(one)
        assert len(pieces) >= 1
        # Reconstruct h(x) by locating the piece that contains x.
        def h(xv):
            for p in pieces:
                if float(p.domain.a) - 1e-12 <= xv <= float(p.domain.b) + 1e-12:
                    return float(p(jnp.float64(xv)))
            return 0.0
        assert abs(h(0.0) - 2.0) < 1e-10
        assert abs(h(1.0) - 1.0) < 1e-10
        assert abs(h(-1.0) - 1.0) < 1e-10
        assert abs(h(2.0) - 0.0) < 1e-9
        assert abs(h(-2.0) - 0.0) < 1e-9

    def test_conv_with_zero_is_zero(self):
        f = Bndfun.from_function(lambda x: x, D1)
        z = Bndfun.from_function(lambda x: jnp.zeros_like(x), D1)
        pieces = f.conv(z)
        for p in pieces:
            assert _max_abs_on_grid(p) < 1e-12

    def test_empty_operand(self):
        f = Bndfun.from_function(lambda x: x, D1)
        assert f.conv(Bndfun.empty()) == []
        assert Bndfun.empty().conv(f) == []


class TestDeltafunChangeMap:
    def test_delta_location_moves(self):
        # A delta at 0.5 on [-1, 1] maps to the corresponding point on [0, 4].
        fp = Bndfun.from_function(lambda x: jnp.zeros_like(x), D1)
        df = Deltafun(fp, delta_locs=[0.5], delta_mags=[[2.0]])
        g = df.change_map((0.0, 4.0))
        # affine map: -1->0, 1->4, so 0.5 -> 3.0
        assert abs(float(np.asarray(g.delta_locs)[0]) - 3.0) < 1e-12
        assert float(g.domain.a) == 0.0 and float(g.domain.b) == 4.0


class TestDeltafunConv:
    def test_delta_translates_function(self):
        # delta at x0=0.3 convolved with a smooth f translates f by x0.
        f = Bndfun.from_function(lambda x: x ** 2, D1)
        d_f = Deltafun.from_fun(f)
        fp0 = Bndfun.from_function(lambda x: jnp.zeros_like(x), D1)
        delta = Deltafun(fp0, delta_locs=[0.3], delta_mags=[[1.0]])
        pieces = d_f.conv(delta)
        # The result should evaluate to f(x - 0.3) on the interior.
        found = False
        for p in pieces:
            if float(p.domain.a) <= 0.3 <= float(p.domain.b):
                val = float(p(jnp.float64(0.3)))  # f(0) = 0
                assert abs(val - 0.0) < 1e-9
                found = True
        assert found

    def test_delta_prime_differentiates(self):
        # delta' convolved with f=x recovers f' = 1.
        f = Bndfun.from_function(lambda x: x, D1)
        d_f = Deltafun.from_fun(f)
        fp0 = Bndfun.from_function(lambda x: jnp.zeros_like(x), D1)
        delta = Deltafun(fp0, delta_locs=[0.0], delta_mags=[[1.0]])
        pieces = d_f.conv(delta.diff(1))
        diff = pieces[0] - 1.0
        assert _max_abs_on_grid(diff) < 1e-9

    def test_empty_propagates(self):
        f = Bndfun.from_function(lambda x: x, D1)
        d_f = Deltafun.from_fun(f)
        assert d_f.conv(Deltafun.empty()) == []
        assert Deltafun.empty().conv(d_f) == []
