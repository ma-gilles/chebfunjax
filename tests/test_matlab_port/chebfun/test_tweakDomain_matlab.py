"""Port of MATLAB Chebfun tests/chebfun/test_tweakDomain.m (Fable 5).

``tweak_domain`` returns 0-based indices of the moved breakpoints;
MATLAB's are 1-based, so the expected indices are shifted by one.

Provenance
----------
MATLAB source : tests/chebfun/test_tweakDomain.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun, tweak_domain

jax.config.update("jax_enable_x64", True)

EPS = np.finfo(float).eps


def _test(df, dg):
    f = chebfun(1.0, domain=tuple(df))
    g = chebfun(1.0, domain=tuple(dg))
    f, g, j, k = tweak_domain(f, g)
    return (np.asarray(list(f.domain.breakpoints)),
            np.asarray(list(g.domain.breakpoints)),
            [i + 1 for i in j], [i + 1 for i in k])


class TestChebfunTweakDomain:
    def test_all_matlab_assertions(self):
        df, dg, j, k = _test([-1 + EPS, 1], [-1, 1])
        assert np.array_equal(df, dg) and j == [1] and k == [1]    # pass(1)

        df, dg, j, k = _test([-1, 1], [-1, 1 + EPS])
        assert np.array_equal(df, dg) and j == [2] and k == [2]    # pass(2)

        df, dg, j, k = _test([-1, 0, 1], [-1, EPS, 1])
        assert np.array_equal(df, dg) and j == [2] and k == [2]    # pass(3)

        df, dg, j, k = _test([-1, -.5, 0, .5, 1], [-1, EPS, 1])
        assert np.array_equal(dg, [-1, 0, 1]) and j == [3] and k == [2]  # pass(4)

        df, dg, j, k = _test([-1, .5 + EPS, 1], [-1, .5 - EPS, 1])
        assert np.array_equal(df, dg) and j == [2] and k == [2]    # pass(5)

        a = [-1 - EPS, -.5, -.2 + EPS, EPS, .4, .7, 1]
        b = [-1, -.2 + EPS, EPS, .4 + 2 * EPS, .5, 1 + EPS]
        df, dg, j, k = _test(a, b)
        df2 = [-1, -.5, -.2 + EPS, EPS, .4 + EPS, .7, 1]
        dg2 = [-1, -.2 + EPS, EPS, .4 + EPS, .5, 1]
        assert np.array_equal(df, df2) and np.array_equal(dg, dg2)  # pass(6)
        assert j == [1, 5, 7] and k == [1, 4, 6]
