"""Port of MATLAB Chebfun tests/chebfun/test_constructor_inputs_periodic.m
(Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_constructor_inputs_periodic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun

jax.config.update("jax_enable_x64", True)

EPS = np.finfo(float).eps


class TestChebfunConstructorInputsPeriodic:
    def test_all_matlab_assertions(self):
        chebfun(lambda x: jnp.cos(np.pi * x) ** 2, periodic=True,
                vectorize=True)                                     # pass(1)

        with pytest.raises(ValueError, match="parseInputs:periodic"):
            chebfun(lambda x: jnp.cos(np.pi * x), domain=(-1.0, 0.0, 1.0),
                    periodic=True)                                  # pass(2)

        f = chebfun(lambda x: jnp.exp(jnp.sin(np.pi * x)), periodic=True, n=10)
        assert len(f.funs) == 1 and len(f) == 10                    # pass(3)

        f = chebfun(jnp.asarray(np.random.RandomState(0).rand(10, 3)),
                    periodic=True, equi=True)
        assert len(f.funs) == 1 and len(f) == 10                    # pass(4)

        f = chebfun(jnp.asarray([0.5, 0.0, 0.5]), periodic=True, coeffs=True)
        assert np.allclose(np.asarray(f.funs[0].tech.coeffs), [0.5, 0, 0.5])  # pass(5)

        f = chebfun("1", periodic=True)
        assert np.all(np.asarray(f(jnp.linspace(-1, 1, 10))) == 1)  # pass(6)

        f = chebfun(lambda x: 1 + jnp.sin(np.pi * jnp.sin(10 * np.pi * x)),
                    periodic=True, trunc=11)
        c = np.asarray(f.funs[0].tech.coeffs)
        assert abs(1 - c[5]) < EPS                                  # pass(7)

        f = chebfun(lambda x: jnp.cos(np.pi * x), trig=True).T
        g = chebfun(f)
        assert g.is_transposed is False                             # pass(8)
