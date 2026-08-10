"""Port of MATLAB Chebfun tests/chebfun/test_erfX.m (Fable 5).

erf/erfc/erfinv of sin vs direct construction; erfcx/erfcinv skipped
(no counterparts).

Provenance
----------
MATLAB source : tests/chebfun/test_erfX.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import erf, erfc, erfinv

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
X = jnp.asarray(np.linspace(-0.97, 0.97, 100))


class TestChebfunErfX:
    @pytest.mark.parametrize("name,meth,ref", [
        ("erf", "erf", erf), ("erfc", "erfc", erfc),
        ("erfinv", "erfinv", erfinv)])
    def test_erf_family_of_sin(self, name, meth, ref):
        f = cj.chebfun(jnp.sin)
        g = getattr(f, meth)()
        exact = jnp.asarray(ref(np.sin(np.asarray(X))))
        err = jnp.abs(g(X) - exact)
        assert float(jnp.max(err)) < 1e3 * EPS * max(g.vscale, 1.0), name

    def test_erfcx(self):
        # MATLAB test_erfX.m F{4} = @erfcx, composed with sin.
        from scipy.special import erfcx
        f = cj.chebfun(jnp.sin)
        g = f.erfcx()
        exact = jnp.asarray(erfcx(np.sin(np.asarray(X))))
        err = jnp.abs(g(X) - exact)
        assert float(jnp.max(err)) < 1e3 * EPS * max(g.vscale, 1.0)

    def test_erfcinv(self):
        # MATLAB test_erfX.m F{5} = @(x) erfcinv(x+1): the argument must
        # lie in [0, 2], so erfcinv is applied to sin + 1.
        from scipy.special import erfcinv
        f = cj.chebfun(jnp.sin) + 1.0
        g = f.erfcinv()
        exact = jnp.asarray(erfcinv(np.sin(np.asarray(X)) + 1.0))
        err = jnp.abs(g(X) - exact)
        assert float(jnp.max(err)) < 1e3 * EPS * max(g.vscale, 1.0)

    def test_erfcinv_complex_raises(self):
        # MATLAB erfcinv.m errors 'CHEBFUN:CHEBFUN:erfcinv:notreal'.
        f = cj.chebfun(lambda x: 1.0 + 1j * x)
        with pytest.raises(ValueError):
            f.erfcinv()
