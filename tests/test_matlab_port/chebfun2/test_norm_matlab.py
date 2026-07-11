"""Port of MATLAB Chebfun tests/chebfun2/test_norm.m (Fable 5).

Frobenius-norm assertions ported; inf-norm, p=4, and spectral 2-norm are
xfailed: chebfunjax Chebfun2.norm supports only 'fro' (and treats 2 as
Frobenius, whereas MATLAB's norm(f,2) is the spectral norm).

Provenance
----------
MATLAB source : tests/chebfun2/test_norm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 1000 * EPS


class TestChebfun2Norm:
    def test_fro_of_x(self):
        f = Chebfun2.from_function(lambda x, y: x)
        assert abs(float(f.norm()) - np.sqrt(4 / 3)) < TOL

    def test_fro_of_ix(self):
        f = Chebfun2.from_function(lambda x, y: 1j * x)
        assert abs(float(f.norm()) - np.sqrt(4 / 3)) < TOL

    def test_fro_of_expxy(self):
        f = Chebfun2.from_function(lambda x, y: jnp.exp(x * y))
        assert abs(float(f.norm()) - 2.236768845167052) < TOL
        assert abs(float(f.norm("fro")) - 2.236768845167052) < TOL

    @pytest.mark.xfail(reason="Chebfun2.norm supports only 'fro'; no inf "
                       "norm")
    def test_inf_norm(self):
        f = Chebfun2.from_function(lambda x, y: x)
        assert abs(float(f.norm(jnp.inf)) - 1.0) < TOL

    @pytest.mark.xfail(reason="Chebfun2.norm supports only 'fro'; no p=4 "
                       "norm")
    def test_p4_norm(self):
        f = Chebfun2.from_function(lambda x, y: x)
        assert abs(float(f.norm(4)) - (4 / 5) ** 0.25) < TOL

    @pytest.mark.xfail(reason="MATLAB norm(f,2) is the SPECTRAL norm "
                       "(largest singular value); chebfunjax treats p=2 "
                       "as Frobenius")
    def test_spectral_norm(self):
        f = Chebfun2.from_function(lambda x, y: jnp.exp(x * y))
        assert abs(float(f.norm(2)) - 2.119814813637055) < TOL
