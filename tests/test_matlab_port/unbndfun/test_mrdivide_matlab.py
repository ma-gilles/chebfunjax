"""Port of MATLAB Chebfun tests/unbndfun/test_mrdivide.m (Opus 4.8).

``A/B`` divides an array-valued unbndfun by a scalar (case 1) or a numerical
matrix (case 2).  chebfunjax's ``Unbndfun`` wraps a single scalar Chebtech2
and has no array-valued representation, so neither the column layout nor the
matrix right-division is available.

Provenance
----------
MATLAB source : tests/unbndfun/test_mrdivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.domain import Domain
from chebfunjax.fun.unbndfun import Unbndfun

EPS = float(np.finfo(np.float64).eps)
INF = np.inf


class TestUnbndfunMrdivide:
    def test_array_valued_over_scalar(self):
        # pass(1): A/3 where A = [exp x*exp (1-exp)/x] on [-inf, 3pi].
        # FIXED (Fable 5, Big-Three array-valued epic): Unbndfun now supports
        # (n, m) array-valued funs; division by a scalar acts column-wise.
        op = lambda x: jnp.stack(
            [jnp.exp(x), x * jnp.exp(x), (1 - jnp.exp(x)) / x], axis=-1
        )
        A = Unbndfun.from_function(op, Domain((-INF, 3 * np.pi)))
        X = A / 3.0
        x = jnp.asarray(np.linspace(-1e6, 3 * np.pi, 100))
        op_exact = lambda xx: jnp.stack(
            [jnp.exp(xx) / 3, xx * jnp.exp(xx) / 3, (1 - jnp.exp(xx)) / (3 * xx)],
            axis=-1,
        )
        err = float(jnp.max(jnp.abs(X(x) - op_exact(x))))
        assert err < 1e1 * EPS * X.vscale

    def test_array_valued_over_matrix(self):
        # pass(2): A/B with B a 3x3 matrix; residual X*B - A ~ 0 (mtimes).
        op = lambda x: jnp.stack(
            [jnp.exp(x), x * jnp.exp(x), (1 - jnp.exp(x)) / x], axis=-1
        )
        A = Unbndfun.from_function(op, Domain((-INF, 3 * np.pi)))
        rng = np.random.default_rng(6178)
        Bm = rng.random((3, 3))
        X = A / Bm
        res = X @ Bm - A
        # MATLAB samples random INTERIOR points; the exp columns peak at the
        # finite endpoint x=3*pi, so stop just short of it.
        x = jnp.asarray(np.linspace(-1e6, 3 * np.pi, 100)[:-1])
        err = float(np.max(np.abs(np.asarray(res(x)))))
        assert err < 1e1 * EPS * float(X.vscale)
