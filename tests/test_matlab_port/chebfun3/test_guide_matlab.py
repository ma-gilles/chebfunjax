"""Original Chebfun3 guide predicates1--19, Chebfun7574c77.

First10 retain native functions, public construction/length routes,
quasimatrix curve input, comparison predicates and tolerances.
Previously qualified HOSVD assertions11--19 are unchanged.
"""

# uses-numpy: inherited earlier guide test array helpers.
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3
from chebfunjax.chebpref import ChebfunPref

EPS = float(np.finfo(np.float64).eps)
TOL = 1e3 * ChebfunPref().cheb3Prefs.chebfun3eps
_EXACT_SUM3 = 4.28685406230184188268


def _mode_length(factors):
    return max(len(np.asarray(t.coeffs)) for t in factors)


class TestChebfun3Guide:
    def test_pass1to4_basics(self):
        f = chebfun3(lambda x, y, z: 1/(1+x**2+y**2+z**2))
        assert abs(f(0., .5, .5)-2/3) < TOL
        assert abs(f.sum3()-_EXACT_SUM3) < TOL
        assert abs(f.mean3()-(1/8)*_EXACT_SUM3) < TOL
        assert abs(f.max3()[0]-1) < TOL

    def test_pass5to7_degenerate_lengths(self):
        from chebfunjax.chebfun1d.chebfun import chebfun

        f1 = chebfun(lambda x: jnp.exp(x))
        len1d = len(f1)
        f3 = chebfun3(lambda x, y, z: jnp.exp(x))
        m3, n3, p3 = f3.length()
        assert m3 < 2*len1d and n3 == 1 and p3 == 1
        f3 = chebfun3(lambda x, y, z: jnp.exp(y))
        m3, n3, p3 = f3.length()
        assert m3 == 1 and n3 < 2*len1d and p3 == 1
        f3 = chebfun3(lambda x, y, z: jnp.exp(z))
        m3, n3, p3 = f3.length()
        assert m3 == 1 and n3 == 1 and p3 < 2*len1d

    def test_pass8_max3_submultiplicative(self):
        f = chebfun3(lambda x, y, z: jnp.sin(x+y*z))
        g = chebfun3(lambda x, y, z: jnp.cos(15*jnp.exp(z))
                    / (5+x**3+2*y**2+z))
        assert (f*g).max3()[0] <= f.max3()[0]*g.max3()[0]

    def test_pass9_helix_line_integral(self):
        from chebfunjax.chebfun1d.chebfun import chebfun

        length = 8*np.pi
        curve = chebfun(lambda t: jnp.stack(
            [jnp.cos(t), jnp.sin(t), t/length], axis=-1), domain=(0., length))
        f = chebfun3(lambda x, y, z: x+y*z)
        integral = f.integral(curve)
        exact = -jnp.sqrt(1+length**2)/length
        assert abs(integral-exact) < TOL

    def test_pass10_trig_ctor(self):
        from chebfunjax.chebfun3d.chebfun3 import chebfun3
        ff = lambda x, y, z: jnp.tanh(3 * jnp.sin(x)) - jnp.sin(y + 0.5) ** 2 + jnp.cos(6 * z)  # noqa: E731
        dom = (-np.pi, np.pi, -np.pi, np.pi, -np.pi, np.pi)
        m, n, p = chebfun3(ff, dom, trig=True).length()
        mc, nc, pc = chebfun3(ff, dom).length()
        assert m <= mc and n <= nc and p <= pc                               # pass(10)

    def test_pass11to19_hosvd(self):
        # Literal native assertions11--19, including its three tautologies.
        f = Chebfun3.from_function(lambda x, y, z: jnp.sin(x+2*y+3*z))
        values, core, cols, rows, tubes = f.hosvd(return_factors=True)
        for mode in values:
            assert mode[1] <= mode[1]
        for panel in (cols, rows, tubes):
            gram = panel.ctranspose() @ panel
            assert jnp.linalg.norm(jnp.eye(gram.shape[0])-gram, ord=2) < TOL
        assert jnp.linalg.norm(core[0, :, :]*core[1, :, :], ord=2) < TOL
        assert jnp.linalg.norm(core[:, 0, :]*core[:, 1, :], ord=2) < TOL
        assert jnp.linalg.norm(core[:, :, 0]*core[:, :, 1], ord=2) < TOL
