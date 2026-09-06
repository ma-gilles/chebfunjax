"""Port of MATLAB Chebfun tests/misc/test_randnfun.m (Fable 5).

MATLAB's ``rng(k)`` is ``numpy.random.seed(...)`` here, with MATLAB's
``rng(0)`` mapped to the Mersenne-Twister seed 5489 it corresponds to
(the normal streams still differ, so the assertions are the statistical
ones MATLAB makes).  Array-valued outputs (``n > 1`` columns) are array-valued
Chebfuns; ``cov(A)`` is the column covariance via a Quasimatrix.

Provenance
----------
MATLAB source : tests/misc/test_randnfun.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.chebfun1d.randfuns import randnfun

jax.config.update("jax_enable_x64", True)


def _cols(f):
    return Quasimatrix([f.extract_columns(j) for j in range(f.n_columns)],
                       f.domain)


def _n(f):
    return float(jnp.linalg.norm(jnp.atleast_1d(jnp.asarray(f.norm(2)))))


class TestMiscRandnfun:
    def test_all_matlab_assertions(self):
        np.random.seed(5489)
        f = randnfun(.01)
        assert abs(float(f.std()) - 1) < .1                          # pass(1)
        assert abs(float(f.mean())) < .1                             # pass(2)
        np.random.seed(5489)
        g = randnfun(.01, 1, [5, 7])
        assert abs(float(f.sum()) - float(g.sum())) < 1e-3           # pass(3)

        A = randnfun(1, 10, [0, 10])
        X = np.asarray(_cols(A).cov())
        assert np.linalg.norm(X - X.T) == 0                          # pass(4)

        np.random.seed(5489)
        f1 = randnfun("big", 1 / 64, [0, 3])
        np.random.seed(5489)
        f2 = np.sqrt(2 * 64) * randnfun(1 / 64, 1, [0, 3])
        assert _n(f1 - f2) < .2 * _n(f1)                             # pass(5)

        np.random.seed(5489)
        f1 = randnfun(1, [0, 4])
        np.random.seed(5489)
        f2 = randnfun(1 / 4, [0, 1])
        assert np.linalg.norm(np.asarray(f1(jnp.asarray([.8, 1.2])))
                              - np.asarray(f2(jnp.asarray([.2, .3])))) == 0  # pass(6)

        f = randnfun(1e6)
        assert _n(f.diff()) < 1e-4                                   # pass(7)
        f = randnfun(np.inf)
        assert _n(f.diff()) == 0                                     # pass(8)

        np.random.seed(5489)
        f = randnfun(.01, "trig")
        assert abs(float(f.std()) - 1) < .1                          # pass(9)
        assert abs(float(f.mean())) < .1                             # pass(10)
        np.random.seed(5489)
        g = randnfun(.01, "trig", 1, [5, 7])
        assert float(f.sum()) - float(g.sum()) < 1e-3                # pass(11)

        A = randnfun("trig", 1, 10, [0, 10])
        X = np.asarray(_cols(A).cov())
        assert np.linalg.norm(X - X.T) == 0                          # pass(12)

        f = randnfun("trig") + 1j * randnfun("trig")
        assert abs(complex(f(jnp.asarray(-.999))) - complex(f(jnp.asarray(.999)))) < .1  # pass(13)

        np.random.seed(5489)
        f1 = randnfun(1 / 16, "big", "trig") / (4 * np.sqrt(2))
        np.random.seed(5489)
        f2 = randnfun(1 / 16, "trig")
        assert _n(f1 - f2) < .2 * _n(f1)                             # pass(14)

        np.random.seed(5489)
        f1 = randnfun(1, [0, 4], "trig")
        np.random.seed(5489)
        f2 = randnfun(1 / 4, [0, 1], "trig")
        assert np.linalg.norm(np.asarray(f1(jnp.asarray([.8, 1.2])))
                              - np.asarray(f2(jnp.asarray([.2, .3])))) == 0  # pass(15)

        np.random.seed(5489)
        f1 = randnfun([0, 4], "trig")
        np.random.seed(5489)
        f2 = randnfun([4, 8], "trig")
        assert np.linalg.norm(np.asarray(f1(jnp.asarray([.8, 1.2])))
                              - np.asarray(f2(jnp.asarray([4.8, 5.2])))) < 1e-14  # pass(16)

        f = randnfun(6, "trig")
        assert _n(f.diff()) == 0                                     # pass(17)

        np.random.seed(1)
        f1 = randnfun([7, 17], .3, "big").cumsum()
        np.random.seed(1)
        f2 = randnfun([7, 17], .1, "big").cumsum()
        assert float((f1 - f2).mean()) < .5                          # pass(18)

        np.random.seed(5489)
        f = randnfun(.3, "complex", [0, 77])
        assert abs(float(f.std()) - 1) < .1                          # pass(19)
        assert abs(complex(f.mean())) < .1                           # pass(20)

        nsamp = 600
        f = randnfun(.1, [0, 1], "big", nsamp)
        b = f.cumsum()
        b1 = np.asarray(b(jnp.asarray(1.0)))                         # s = sum(b.^2, 2)/nsamp; s(1)
        s1 = float(np.sum(b1 ** 2)) / nsamp
        assert abs(s1 - 1) < .2                                      # pass(21)
        f = randnfun(.1, [0, 1], "big", "complex", nsamp)
        b = f.cumsum()
        b1 = np.asarray(b(jnp.asarray(1.0)))
        s1 = float(np.real(np.sum(np.conj(b1) * b1))) / nsamp
        assert abs(s1 - 1) < .2                                      # pass(22)

        np.random.seed(7)
        f = randnfun("norm")
        np.random.seed(7)
        g = randnfun("big")
        assert float(f(jnp.asarray(.2))) == float(g(jnp.asarray(.2)))  # pass(23)
