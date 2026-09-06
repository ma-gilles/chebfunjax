"""Port of MATLAB Chebfun tests/chebfun/test_doubleLength.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_doubleLength.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun

jax.config.update("jax_enable_x64", True)


class TestChebfunDoubleLength:
    def test_all_matlab_assertions(self):
        f = chebfun(jnp.sin)
        g = chebfun(jnp.sin, doubleLength=True)
        assert 2 * len(f) - 1 == len(g)                              # pass(1)

        op = lambda x: jnp.stack([jnp.sin(x), jnp.tanh(x)], axis=-1)  # noqa: E731
        f = chebfun(op)
        g = chebfun(op, doubleLength=True)
        assert 2 * len(f) - 1 == len(g)                              # pass(2)

        f = chebfun(jnp.asarray([3.0, 2.0, 1.0]), doubleLength=True)
        assert len(f) == 5                                           # pass(3)
        f = chebfun(jnp.asarray([3.0, 2.0, 1.0]), coeffs=True,
                    doubleLength=True)
        assert len(f) == 5                                           # pass(4)

        op = lambda x: jnp.exp(jnp.sin(np.pi * x))  # noqa: E731
        f = chebfun(op)
        g = chebfun(op, doubleLength=True)
        assert len(g) == 2 * len(f) - 1                              # pass(5)

        with pytest.raises(ValueError, match="doubleLengthSplitting"):
            chebfun(jnp.sign, doubleLength=True, splitting=True)     # pass(6)
        with pytest.raises(ValueError, match="doubleLengthBreakpoints"):
            chebfun(jnp.exp, domain=(0.0, 0.2, 1.0), doubleLength=True)  # pass(7)
