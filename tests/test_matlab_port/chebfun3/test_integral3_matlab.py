"""Port of MATLAB Chebfun tests/chebfun3/test_integral3.m (Fable 5).

MATLAB integral3(f) delegates to sum3(f); integral3(f, dom) restricts
then integrates over the requested contained cuboid.

Provenance
----------
MATLAB source : tests/chebfun3/test_integral3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebpref import ChebfunPref


def _source_tol():
    return 1e4 * ChebfunPref().cheb3Prefs.chebfun3eps


class TestChebfun3Integral3:
    def test_original_empty_source_predicate(self):
        # MATLAB pass(1): isempty(integral3(chebfun3())).
        assert jnp.size(Chebfun3.empty().integral3()) == 0

    def test_original_restricted_cuboid_source_predicate(self):
        # MATLAB pass(2): integral x over [0,1] x [0,2] x [0,3] is 3.
        f = Chebfun3.from_function(lambda x, y, z: x,
                                   domain=(0.0, 2.0, 0.0, 3.0, -1.0, 4.0))
        assert abs(float(f.integral3((0.0, 1.0, 0.0, 2.0, 0.0, 3.0))) - 3.0) < _source_tol()

    def test_supplemental_odd_integrand(self):
        f = Chebfun3.from_function(lambda x, y, z: x * y * z)
        assert abs(float(f.sum3())) < _source_tol()

    def test_supplemental_polynomial_on_box(self):
        dom = (0.0, 3.0, -1.0, 1.0, 0.0, 1.0)
        f = Chebfun3.from_function(
            lambda x, y, z: x ** 2 * jnp.cos(y) + z, domain=dom)
        exact = 9 * 2 * np.sin(1.0) + 0.5 * 3 * 2
        assert abs(float(f.sum3()) - exact) < 10 * _source_tol()


def test_supplemental_default_domain_and_error_order():
    f = Chebfun3.from_function(lambda x, y, z: x,
                              domain=(0., 2., 0., 3., -1., 4.))
    assert f.integral3() == f.sum3()
    assert jnp.size(Chebfun3.empty().integral3([1.])) == 0
    with pytest.raises(ValueError, match="integral3:baddomain"):
        f.integral3([0., 1.])
    # MATLAB's six linear indices follow column-major order even for a matrix.
    box = jnp.asarray([[0., 0., 0.], [1., 2., 3.]])
    assert abs(float(f.integral3(box))-3.) < _source_tol()
