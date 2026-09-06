"""Port of MATLAB Chebfun tests/chebfun/test_polyval.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_polyval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun, polyval

jax.config.update("jax_enable_x64", True)

EPS = np.finfo(float).eps


class TestChebfunPolyval:
    def test_all_matlab_assertions(self):
        c = np.random.RandomState(666).rand(3)
        x = chebfun("x", domain=(0.0, np.pi))
        f = polyval(c, x)
        err = float((c[0] * x ** 2 + c[1] * x + c[2] - f).norm(jnp.inf))
        assert err < 10 * f.vscale * EPS                            # pass(1)

        x1, x2 = x, x ** 2
        xx = [x1, x2]
        f2 = polyval(c, xx)
        g1 = c[0] * x1 ** 2 + c[1] * x1 + c[2]
        g2 = c[0] * x2 ** 2 + c[1] * x2 + c[2]
        err1 = float((g1 - f2[0]).norm(jnp.inf))
        err2 = float((g2 - f2[1]).norm(jnp.inf))
        vs = max(f2[0].vscale, f2[1].vscale)
        assert err1 < 10 * EPS * vs and err2 < 10 * EPS * vs        # pass(2)

        c2 = np.random.RandomState(667).rand(3)
        cc = np.stack([c, c2], axis=1)
        f3 = polyval(cc, x)
        g4 = c2[0] * x ** 2 + c2[1] * x + c2[2]
        err3 = float((g1 - f3[0]).norm(jnp.inf))
        err4 = float((g4 - f3[1]).norm(jnp.inf))
        vs = max(f3[0].vscale, f3[1].vscale)
        assert err3 < 10 * EPS * vs and err4 < 10 * EPS * vs        # pass(3)

        with pytest.raises(ValueError):
            polyval(cc, xx)                                         # pass(4)
