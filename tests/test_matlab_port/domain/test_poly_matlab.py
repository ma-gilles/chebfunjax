"""Port of MATLAB Chebfun tests/domain/test_poly.m (Fable 5).

Provenance
----------
MATLAB source : tests/domain/test_poly.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import poly

jax.config.update("jax_enable_x64", True)

EPS = np.finfo(float).eps


def _vs(f):
    return float(np.max(np.atleast_1d(np.asarray(f.vscale))))


class TestDomainPoly:
    def test_all_matlab_assertions(self):
        f = poly(0.0, (-1, 1))
        assert abs(float(f(0.0))) < 100 * EPS * _vs(f)                       # pass(1)
        f = poly(0.1, (-1, 1))
        assert len(f) == 2 and abs(float(f(0.1))) < 100 * EPS * _vs(f)       # pass(2)
        x = np.array([-.1, 0.1])
        f = poly(x, (-1, 1))
        assert len(f) == 3 and np.linalg.norm(np.asarray(f(jnp.asarray(x)))) < 100 * EPS * _vs(f)  # pass(3)
        x = np.array([[-.1, 0.1], [-.5, .9]])
        for dom in ((-1, 1), (-1, 2)):
            fs = poly(x, dom)
            err = np.linalg.norm(np.column_stack([np.asarray(fs[0](jnp.asarray(x[:, 0]))),
                                                  np.asarray(fs[1](jnp.asarray(x[:, 1])))]))
            vs = max(_vs(fs[0]), _vs(fs[1]))
            assert len(fs[0]) == 3 and len(fs) == 2 and err < 100 * EPS * vs  # pass(4)-(5)
