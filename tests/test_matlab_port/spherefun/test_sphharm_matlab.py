"""Original assertions from MATLAB tests/spherefun/test_sphharm.m.

Provenance
----------
MATLAB source : tests/spherefun/test_sphharm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun

from ._cart import sph_xyz


def test_source_sphharm_predicates_one_through_five():
    # Source tol = 1e2 * chebfunpref().cheb2Prefs.chebfun2eps.
    tol = 100 * ChebfunPref().cheb2Prefs.chebfun2eps

    f = Spherefun.sphharm(2, 1)
    g = sph_xyz(lambda x, y, z: -jnp.sqrt(15.0 / (4.0 * jnp.pi))*x*z)
    assert float((f-g).norm("inf")) < tol  # pass(1)

    f = Spherefun.sphharm(2, -1)
    g = sph_xyz(lambda x, y, z: -jnp.sqrt(15.0 / (4.0 * jnp.pi))*y*z)
    assert float((f-g).norm("inf")) < tol  # pass(2)

    f = Spherefun.sphharm(3, 1)
    g = sph_xyz(lambda x, y, z: (0.25*jnp.sqrt(21.0/(2.0*jnp.pi))
                                 * x*(1.0-5.0*z**2)))
    assert float((f-g).norm("inf")) < tol  # pass(3)

    # Literal source norm-square predicates; these are not replaced by
    # quadrature over a separately sampled grid.
    f = Spherefun.sphharm(11, 9)
    assert abs(float((f**2).sum2()) - 1.0) < tol  # pass(4)

    f = Spherefun.sphharm(20, -8)
    assert abs(float((f**2).sum2()) - 1.0) < tol  # pass(5)


@pytest.mark.parametrize("m", range(1, 11))
def test_source_sphharm_laplacian_predicates_six_through_forty_five(m):
    # Source resets tol = 1e4 * current chebfun2eps before this loop. One
    # fresh process per source outer-loop m bounds accumulated JAX compilation
    # memory without changing any of the four assertions for this m.
    tol = 10000 * ChebfunPref().cheb2Prefs.chebfun2eps
    pass_index = 6 + 4*(m-1)
    for ell in range(m, m+4):
        u = Spherefun.sphharm(ell, m)
        v = u.laplacian()
        assert float((v + ell*(ell+1)*u).norm("inf")) < tol, (
            f"MATLAB sphharm pass({pass_index}), ell={ell}, m={m}")
        pass_index += 1
    assert pass_index == 6 + 4*m  # Four original pass clauses for this m.
