"""All eighteen literal native PROD predicates, with separate array/quasi cohorts."""

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebfunPref

TOL = 100 * np.finfo(float).eps


class TestNativeProdArray:
    @pytest.mark.parametrize("row", [False, True])
    def test_scalar(self, row):
        f = cj.chebfun(lambda x: x + 2, pref=ChebfunPref())
        if row:
            f = f.transpose()
        assert abs(complex(np.asarray(f.prod())) - 27 * np.exp(-2)) < TOL

    @pytest.mark.parametrize("row", [False, True])
    def test_array_continuous(self, row):
        f = cj.chebfun(lambda x: x + 2, pref=ChebfunPref())
        fa = type(f).horzcat(f, f)
        if row:
            fa = fa.transpose()
        actual = np.asarray(fa.prod())
        exact = np.full((2, 1) if row else (2,), 27 * np.exp(-2))
        error = np.max(np.abs(actual - exact)) if row else np.sum(np.abs(actual - exact))
        print({"native_clause": 5 if row else 3, "error": error, "bound": TOL}, flush=True)
        assert error < TOL
        assert np.array_equal(actual, np.asarray(fa.prod(2 if row else 1)))

    @pytest.mark.parametrize("row", [False, True])
    def test_array_discrete(self, row):
        f = cj.chebfun(lambda x: x, pref=ChebfunPref())
        fa = type(f).horzcat(f, f)
        if row:
            f = f.transpose()
            fa = fa.transpose()
        actual = fa.prod(1 if row else 2)
        assert actual.is_transposed == row
        error = float((f**2 - actual).norm(jnp.inf))
        print({"native_clause": 14 if row else 12, "error": error, "bound": TOL}, flush=True)
        assert error < TOL


from chebfunjax.chebfun1d.linalg import Quasimatrix


class TestNativeProdQuasi:
    @pytest.mark.parametrize("row", [False, True])
    def test_continuous(self, row):
        f = cj.chebfun(lambda x: x + 2, pref=ChebfunPref())
        fa = type(f).horzcat(f, f)
        # Explicit Python Quasimatrix adapter of native cheb2quasi column extraction.
        fq = Quasimatrix(fa.mat2cell(), fa.domain)
        if row:
            fq = fq.transpose()
        actual = np.asarray(fq.prod())
        exact = np.full((2, 1) if row else (2,), 27 * np.exp(-2))
        error = np.max(np.abs(actual - exact)) if row else np.sum(np.abs(actual - exact))
        print({"native_clause": 9 if row else 7, "error": error, "bound": TOL}, flush=True)
        assert error < TOL
        assert np.array_equal(actual, np.asarray(fq.prod(2 if row else 1)))

    @pytest.mark.parametrize("row", [False, True])
    def test_discrete(self, row):
        f = cj.chebfun(lambda x: x, pref=ChebfunPref())
        fa = type(f).horzcat(f, f)
        fq = Quasimatrix(fa.mat2cell(), fa.domain)
        if row:
            pfat2 = fa.transpose().prod(1)
            f = f.transpose()
            fq = fq.transpose()
        actual = fq.prod(1 if row else 2)
        # Native17 literally checks previous array output pfat2, not pfqt2.
        if row:
            assert pfat2.is_transposed
        else:
            assert not actual.is_transposed
        error = float((f**2 - actual).norm(jnp.inf))
        print({"native_clause": 18 if row else 16, "error": error, "bound": TOL}, flush=True)
        assert error < TOL
