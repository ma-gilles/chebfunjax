"""Port of MATLAB Chebfun tests/chebfun/test_exp.m (Fable 5).

exp of a piecewise sign/abs base function under splitting.  expm1 is
skipped (no chebfunjax counterpart).

Provenance
----------
MATLAB source : tests/chebfun/test_exp.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(7681)
XR = jnp.asarray(2 * RNG.uniform(size=1000) - 1)


def base_op(x):
    return jnp.sign(x - 0.1) * jnp.abs(x + 0.2) * jnp.sin(3 * x)


class TestChebfunExp:
    def test_exp_of_piecewise(self):
        # MATLAB builds f = chebfun(base_op, [-1 -.2 .1 1]) under
        # splitting, which resolves the sign/abs jumps with one-sided
        # samples.  The equivalent exact construction here uses the
        # smooth restriction of base_op on each piece (MATLAB's cell
        # syntax): base = -(x+.2)sin(3x) on [-1,.1], +(x+.2)sin(3x)
        # after the sign flip at .1, with the |.| kink at -.2.
        from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
        from chebfunjax.domain import Domain
        ops = [lambda x: (x + 0.2) * jnp.sin(3 * x),
               lambda x: -(x + 0.2) * jnp.sin(3 * x),
               lambda x: (x + 0.2) * jnp.sin(3 * x)]
        brks = [-1.0, -0.2, 0.1, 1.0]
        funs = [_Piece.from_function(op, a, b)
                for op, a, b in zip(ops, brks[:-1], brks[1:])]
        f = Chebfun(funs=funs, domain=Domain(tuple(brks)))
        g = f.exp()
        exact = jnp.exp(base_op(XR))
        mask = (jnp.abs(XR + 0.2) > 1e-6) & (jnp.abs(XR - 0.1) > 1e-6)
        err = jnp.abs(g(XR) - exact)[mask]
        assert float(jnp.max(err)) < 1e2 * g.vscale * EPS

    def test_expm1(self):
        pytest.skip("chebfunjax Chebfun has no expm1")
