"""All original norm predicates from Chebfun tests/chebfun3/test_diffy.m.

Pinned source 7574c77. Functions, domains, construction order and thresholds
are unchanged. MATLAB diffy(f, k) delegates diff(f, k, 2); the Python
adapter is f.diff(dim=2, k=k). Qualification splits cases into serial
fresh processes while retaining the native function/domain iteration order.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref

TOL = 1e6 * ChebfunPref().cheb3Prefs.chebfun3eps
DOMAINS = [(-1., 1., -1., 1., -1., 1.), (-3., 1., -1., 2., 3., 5.)]

FF = [
    lambda x, y, z: x,
    lambda x, y, z: jnp.cos(x)*jnp.exp(y)*jnp.sin(z),
    lambda x, y, z: jnp.cos(x*y*z),
    lambda x, y, z: x**2+x*y**2*z**3,
]

FD1 = [
    lambda x, y, z: 0*x,
    lambda x, y, z: jnp.cos(x)*jnp.exp(y)*jnp.sin(z),
    lambda x, y, z: -x*z*jnp.sin(x*y*z),
    lambda x, y, z: 2*x*y*z**3,
]

FD2 = [
    lambda x, y, z: 0*x,
    lambda x, y, z: jnp.cos(x)*jnp.exp(y)*jnp.sin(z),
    lambda x, y, z: -x**2*z**2*jnp.cos(x*y*z),
    lambda x, y, z: 2*x*z**3,
]


@pytest.mark.parametrize("function_index,domain_index", [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)])
def test_native_diffy(function_index, domain_index):
    domain = DOMAINS[domain_index]
    f = chebfun3(FF[function_index], domain)
    first_reference = chebfun3(FD1[function_index], domain)
    second_reference = chebfun3(FD2[function_index], domain)
    first = f.diff(dim=2, k=1)
    second = f.diff(dim=2, k=2)
    assert (first-first_reference).norm() < TOL
    assert (second-second_reference).norm() < TOL
