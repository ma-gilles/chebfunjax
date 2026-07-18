"""Port of MATLAB Chebfun tests/chebfun/test_all.m (Fable 5).

``all(f)`` is True where the function has no roots.  For array-valued
chebfuns it returns a per-column ``(m,)`` boolean array.

Provenance
----------
MATLAB source : tests/chebfun/test_all.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

_DOM = (-1, -0.5, 0, 0.5, 1)


class TestChebfunAll:
    def test_scalar_sin_has_root(self):
        # pass(1): ~all(sin(x)) -- sin has a root at 0.
        f = cj.chebfun(jnp.sin, domain=_DOM)
        assert not f.all()

    def test_scalar_shifted_sin_has_root(self):
        # pass(2): ~all(sin(x - 0.1)) -- root at 0.1.
        f = cj.chebfun(lambda x: jnp.sin(x - 0.1), domain=_DOM)
        assert not f.all()

    # FIXED (Fable 5, Big-Three array-valued epic): complex-coefficient
    # rootfinding no longer casts to the real part (the colleague
    # matrix stays complex), so exp(2 pi i x) has no spurious roots.
    def test_scalar_complex_never_zero(self):
        # pass(3): all(exp(2 pi i x)) == True (modulus 1, never zero).
        f = cj.chebfun(lambda x: jnp.exp(2j * jnp.pi * x), domain=_DOM)
        assert f.all()

    def test_array_valued(self):
        # pass(4): all([sin(x) sin(x-0.1) exp(2 pi i x)]) == [0 0 1].
        f = cj.chebfun(
            lambda x: jnp.stack(
                [jnp.sin(x), jnp.sin(x - 0.1),
                 jnp.exp(2j * jnp.pi * x)], axis=-1),
            domain=_DOM)
        np.testing.assert_array_equal(
            np.asarray(f.all()), [False, False, True])

    def test_singular(self):
        # pass(5): ~all(sin(x)/(x+1)) with an endpoint singularity.
        pytest.skip("chebfunjax has no SingFun (endpoint 'exps') support")

    def test_unbounded(self):
        # pass(6): all(x^2(1-exp(-x^2))+3) on [-inf, inf].
        pytest.skip("chebfunjax has no unbounded-domain support")
