"""Port of MATLAB Chebfun tests/chebfun/test_range.m (Fable 5).

MATLAB quasimatrices (``cheb2quasi``) map to :class:`Quasimatrix`; the
transposed quasimatrix cases use the ``dim`` argument directly since a
Quasimatrix carries no orientation flag.

Provenance
----------
MATLAB source : tests/chebfun/test_range.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
from chebfunjax.chebfun1d.linalg import Quasimatrix

jax.config.update("jax_enable_x64", True)

EPS = np.finfo(float).eps


def _r_exact(x):
    x = np.asarray(x)
    return np.where(x <= 1, x - x ** 2, x ** 2 - x)


class TestChebfunRange:
    def test_all_matlab_assertions(self):
        x = jnp.asarray(2 * np.random.RandomState(6178).rand(100) - 1)

        assert Chebfun.empty().range().size == 0                    # pass(1)

        f = chebfun(lambda t: t, domain=(-1.0, 0.0, 1.0))
        assert abs(f.range() - 2) < 10 * f.vscale * EPS             # pass(2)
        ft = f.T
        assert abs(ft.range() - 2) < 10 * ft.vscale * EPS           # pass(3)

        fa = chebfun(lambda t: jnp.stack([t, t ** 2], axis=-1),
                     domain=(0.0, 1.0, 2.0))
        fq = Quasimatrix([fa.extract_columns(0), fa.extract_columns(1)],
                         fa.domain)
        assert np.max(np.abs(np.asarray(fa.range()) - [2, 4])) < 10 * fa.vscale * EPS  # pass(4)
        assert np.max(np.abs(np.asarray(fq.range()) - [2, 4])) < 10 * fa.vscale * EPS  # pass(5)

        xx = jnp.asarray(2.0 * np.asarray(x) + 0.0)  # x in [-1, 1] -> use [0, 2] domain points
        xx = jnp.asarray(np.asarray(x) + 1.0)
        ra = fa.range(2)
        assert np.max(np.abs(np.asarray(ra(xx)) - _r_exact(xx))) < 10 * ra.vscale * EPS  # pass(6)
        rq = fq.range(2)
        assert np.max(np.abs(np.asarray(rq(xx)) - _r_exact(xx))) < 10 * rq.vscale * EPS  # pass(7)

        fat = fa.T
        assert np.max(np.abs(np.asarray(fat.range(2)) - [2, 4])) < 10 * fa.vscale * EPS  # pass(8)
        assert np.max(np.abs(np.asarray(fq.range(1)) - [2, 4])) < 10 * fa.vscale * EPS   # pass(9)
        rat = fat.range(1)
        assert np.max(np.abs(np.asarray(rat(xx)) - _r_exact(xx))) < 10 * rat.vscale * EPS  # pass(10)

        assert f.range(3).iszero()                                  # pass(11)
