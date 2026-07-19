"""Port of MATLAB Chebfun tests/chebfun2/test_norm.m (Fable 5).

FIXED: Chebfun2.norm now supports the Frobenius ('fro'), spectral (2 /
'op'), nuclear ('nuc'), max/inf, and even-p norms, backed by
Chebfun2.svd (@separableApprox/svd.m).

Provenance
----------
MATLAB source : tests/chebfun2/test_norm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 1000 * EPS


class TestChebfun2Norm:
    def test_fro_of_x(self):
        f = Chebfun2.from_function(lambda x, y: x)
        assert abs(float(f.norm()) - np.sqrt(4 / 3)) < TOL

    def test_inf_norm_of_x(self):
        f = Chebfun2.from_function(lambda x, y: x)
        assert abs(float(f.norm("inf")) - 1.0) < TOL
        assert abs(float(f.norm(jnp.inf)) - 1.0) < TOL

    def test_p4_norm_of_x(self):
        f = Chebfun2.from_function(lambda x, y: x)
        assert abs(float(f.norm(4)) - (4 / 5) ** 0.25) < TOL

    def test_fro_of_ix(self):
        f = Chebfun2.from_function(lambda x, y: 1j * x)
        assert abs(float(f.norm()) - np.sqrt(4 / 3)) < TOL

    def test_inf_norm_of_ix(self):
        f = Chebfun2.from_function(lambda x, y: 1j * x)
        assert abs(float(f.norm("inf")) - 1.0) < TOL

    def test_p4_norm_of_ix(self):
        f = Chebfun2.from_function(lambda x, y: 1j * x)
        assert abs(float(f.norm(4)) - (4 / 5) ** 0.25) < TOL

    def test_fro_of_expxy(self):
        f = Chebfun2.from_function(lambda x, y: jnp.exp(x * y))
        assert abs(float(f.norm()) - 2.236768845167052) < TOL
        assert abs(float(f.norm("fro")) - 2.236768845167052) < TOL

    def test_spectral_norm(self):
        f = Chebfun2.from_function(lambda x, y: jnp.exp(x * y))
        assert abs(float(f.norm(2)) - 2.119814813637055) < TOL

    def test_nuclear_norm(self):
        f = Chebfun2.from_function(lambda x, y: jnp.exp(x * y))
        assert abs(float(f.norm("nuc")) - 2.925303491814361) < TOL
