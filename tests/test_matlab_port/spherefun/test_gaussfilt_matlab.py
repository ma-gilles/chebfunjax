"""Literal assertions from MATLAB spherefun Gaussian-filter tests.

Provenance
----------
MATLAB source : tests/spherefun/test_gaussfilt.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Retain all source predicates, including the repeated sig=100 loop and the
final mean predicate that overwrites a source pass-array slot.
"""
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun


class TestSpherefunGaussfilt:
    def test_constant_unchanged(self):
        tol = 1e3*ChebfunPref().cheb2Prefs.chebfun2eps
        for sig in (None, 2):
            f = Spherefun.from_function(lambda lam, th: 1.0+0*lam)
            g = f.gaussfilt() if sig is None else f.gaussfilt(sig)
            assert float((f-g).norm()) < tol

    def test_norm_decreases(self):
        f = Spherefun.sphharm(13, 7)
        for _ in (1, 10, 100):
            g = f.gaussfilt(100)
            assert float(g.norm()) < float(f.norm())
            f = g

    def test_mean_preserved(self):
        tol = 1e3*ChebfunPref().cheb2Prefs.chebfun2eps
        f = 2 + Spherefun.sphharm(12, 5)
        g = f.gaussfilt(2)
        assert abs(complex(g.mean2())-2) < tol
