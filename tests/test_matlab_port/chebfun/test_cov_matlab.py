"""Port of MATLAB Chebfun tests/chebfun/test_cov.m (Fable 5).

FIXED: Quasimatrix.cov added in the Fable 5 audit (array-valued
cases via the Quasimatrix counterpart; the exact closed-form
covariance matrix of [sin cos exp] is matched entrywise).

Provenance
----------
MATLAB source : tests/chebfun/test_cov.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix

EPS = np.finfo(float).eps


class TestChebfunCov:
    def test_single_column_cov_is_var(self):
        # pass(2): cov of one function equals var (single-interval
        # complex function; Quasimatrix requires one interval)
        f = cj.chebfun(lambda x: jnp.exp(4 * np.pi * 1j * x))
        Q = Quasimatrix([f], f.domain)
        C = np.asarray(Q.cov())
        assert abs(complex(C[0, 0])
                   - complex(np.asarray(f.var()))) < 100 * EPS

    def test_covariance_matrix(self):
        # pass(4): exact closed-form covariance of [sin cos exp]
        Q = Quasimatrix(
            [cj.chebfun(jnp.sin), cj.chebfun(jnp.cos),
             cj.chebfun(jnp.exp)],
            cj.chebfun(jnp.sin).domain)
        C = np.asarray(Q.cov())
        css = (1 - np.sin(2) / 2) / 2
        csc = 0.0
        cse = (np.sin(1) + np.cos(1)
               + np.exp(2) * (np.sin(1) - np.cos(1))) / (4 * np.e)
        ccc = (1 + np.sin(1) * np.cos(1)) / 2 - np.sin(1) ** 2
        cce = (np.sin(1) - np.cos(1)
               + np.exp(2) * (np.sin(1) + np.cos(1))) / (4 * np.e) \
            - np.sin(1) * (np.e - np.exp(-1)) / 2
        cee = (1 - np.exp(-2)) / 2
        exact = np.array([[css, csc, cse],
                          [csc, ccc, cce],
                          [cse, cce, cee]])
        assert np.max(np.abs(C - exact)) < 100 * EPS
