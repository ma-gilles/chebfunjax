"""Port of MATLAB Chebfun tests/singfun/test_plus.m (Opus 4.8).

Literal predicates from the pinned source. Constructor defaults are exercised.
Probe inputs use deterministic MT19937 seed666 reconstruction; a native seed666
capture is still pending, so these are source-shaped deterministic controls,
not a claim of native random-input parity.

Provenance
----------
MATLAB source : tests/singfun/test_plus.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np  # uses-numpy: deterministic source-shaped RNG test inputs

from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech2

EPS = float(np.finfo(np.float64).eps)

X = jnp.asarray(2 * np.random.RandomState(666).rand(100) - 1)


def _sf(f, exps=None, **kwargs):
    return Singfun.from_function(f, exps, **kwargs)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))




class TestSingfunPlus:
    def test_empty(self):
        f = Singfun.empty()
        g = _sf(lambda x: 1.0 / (1 + x), (-1.0, 0.0))
        assert (f + f).isempty()
        assert (f + g).isempty()
        assert (g + f).isempty()

    def test_smooth_plus_smooth_not_singfun(self):
        f = _sf(lambda x: jnp.sin(x))
        g = _sf(lambda x: jnp.cos(x))
        assert not isinstance(f + g, Singfun)

    def test_smoothfun_plus_singfun(self):
        # MATLAB: smoothfun + (smooth) singfun isa smoothfun, both ways.
        f = Chebtech2.from_function(lambda x: jnp.sin(x))
        g = _sf(lambda x: jnp.cos(x))
        assert isinstance(f + g, Chebtech2) and not isinstance(f + g, Singfun)
        assert isinstance(g + f, Chebtech2) and not isinstance(g + f, Singfun)

    def test_add_complex_scalar(self):
        alpha = -0.194758928283640 + 0.075474485412665j
        f = _sf(lambda x: 1.0 / ((1 + x) * (1 - x)), (-1.0, -1.0))
        g1 = f + alpha
        g2 = alpha + f
        assert g1.isequal(g2)
        exact = 1.0 / ((1 + X) * (1 - X)) + alpha
        assert _ninf(g1(X) - exact) < 1e3 * EPS

    def test_add_zero_to_zero(self):
        f = _sf(lambda x: jnp.zeros_like(x), (0.0, 0.0))
        h1 = f + f
        h2 = f + f
        assert h1.isequal(h2)
        assert _ninf(h1(X)) <= 2e3 * EPS

    def test_add_same_exponents(self):
        def fh(x):
            return jnp.sin(np.pi * x) / (1 - x)

        def gh(x):
            return jnp.cos(np.pi * x) / (1 - x)

        f = _sf(fh)
        g = _sf(gh)
        h1 = f + g
        h2 = g + f
        assert h1.isequal(h2)
        exact = fh(X) + gh(X)
        assert _ninf(h1(X) - exact) <= 2e3 * EPS

    def test_add_integer_exponent_diff(self):
        def fh(x):
            return jnp.sin(np.pi * x) / (1 - x)

        def gh(x):
            return jnp.cos(1e2 * x)

        f = _sf(fh)
        g = _sf(gh)
        h1 = f + g
        h2 = g + f
        assert h1.isequal(h2)
        exact = fh(X) + gh(X)
        assert _ninf(h1(X) - exact) <= 2e3 * EPS

    def test_add_complex_function(self):
        # FIXED (Fable 5): the Chebtech1 complex-transform fix made
        # complex smooth parts work in Singfun too.
        def fh(x):
            return jnp.sin(np.pi * x) / (1 - x)

        def gh(t):
            return jnp.sinh(t * np.exp(2 * np.pi * 1j / 6))

        f = _sf(fh)
        g = _sf(gh)
        h = f + g
        assert h.isequal(g + f)
        exact = fh(X) + gh(X)
        assert _ninf(h(X) - exact) <= 2e3 * EPS

    def test_plus_vs_direct_construction(self):
        f = _sf(lambda x: x)
        g = _sf(lambda x: jnp.cos(x) - 1)
        h1 = f + g  # smooth sum -> demoted to a bare Chebtech2
        h2 = _sf(lambda x: x + jnp.cos(x) - 1)
        vs1 = h1.vscale
        vs2 = h2.vscale
        tol = 10 * max(vs1 * EPS, vs2 * EPS)
        assert _ninf(h1(X) - h2(X)) < tol

    def test_noninteger_exponent_diff_small_result(self):
        def op(x):
            return ((x + 1) / 2) ** np.pi

        f = _sf(op, (np.pi - 3, 0.0))
        g = _sf(op, n=256)
        h = f - g
        assert h.ishappy and len(h) < 1024
