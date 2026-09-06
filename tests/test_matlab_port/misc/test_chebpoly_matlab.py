"""Port of MATLAB Chebfun tests/misc/test_chebpoly.m (Fable 5).

MATLAB's chebfun-valued ``chebpoly`` is
:func:`chebfunjax.chebfun1d.chebfun.chebpoly`.

Provenance
----------
MATLAB source : tests/misc/test_chebpoly.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebpoly

jax.config.update("jax_enable_x64", True)

TOL = 1e-14


class TestMiscChebpoly:
    def test_all_matlab_assertions(self):
        n = np.array([0, 1, 2, 3, 5, 8])
        x = np.linspace(-1, 1, 10)[[1, -2]]
        T1 = chebpoly(n, kind=1)
        T1x = np.asarray(T1(jnp.asarray(x)))
        T2 = np.cos(np.outer(np.arccos(x), n))
        assert np.max(np.abs(T2 - T1x)) < TOL                       # pass(1)

        U1 = chebpoly(n, kind=2)
        U1x = np.asarray(U1(jnp.asarray(x)))
        U2 = np.sin(np.outer(np.arccos(x), n + 1)) / np.sin(np.arccos(x))[:, None]
        assert np.max(np.abs(U2 - U1x)) < TOL                       # pass(2)

        T = chebpoly(27)
        assert abs(T.max()[1] - 1) < TOL                            # pass(3)
        T = chebpoly(44)
        assert abs(T.min()[1] + 1) < TOL                            # pass(4)

        T = chebpoly([7, 8])
        v = T.extract_columns(0) + T.extract_columns(1)             # T*[1;1]
        assert abs(float(v(jnp.asarray(-1.0)))) < TOL               # pass(5)

        T = chebpoly(100000)
        assert abs(float(T(jnp.asarray(1.0))) - 1) < TOL            # pass(6)
