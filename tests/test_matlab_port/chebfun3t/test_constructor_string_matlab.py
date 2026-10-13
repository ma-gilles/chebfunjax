"""Port MATLAB chebfun3t constructor string passes 4–5.

Source: ``tests/chebfun3t/test_constructor.m`` at Chebfun commit 7574c77.
The expression parser and ``chebfun3t`` forwarding path are already public;
these assertions specifically qualify their composition through the Tucker
wrapper.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3t import chebfun3t
from chebfunjax.chebpref import ChebfunPref


def test_string_constructor_matches_native_passes_4_and_5():
    ff = lambda x, y, z: jnp.cos(x + z) + jnp.sin(x * y * z)  # noqa: E731
    fstr = "cos(x+z) + sin(x.*y.*z)"
    domain = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    tol = 1e2 * ChebfunPref().cheb3Prefs.chebfun3eps

    f1 = chebfun3t(ff)
    f3 = chebfun3t(fstr)
    f4 = chebfun3t(fstr, domain)

    # MATLAB [xx,yy,zz] = meshgrid(linspace(-1,1)); preserve its 100^3
    # query grid and the two original maximum-absolute-error predicates.
    x = np.linspace(-1.0, 1.0, 100)
    xx, yy, zz = np.meshgrid(x, x, x)
    err_4 = np.max(np.abs(np.asarray(f1(xx, yy, zz)) - np.asarray(f3(xx, yy, zz))))
    err_5 = np.max(np.abs(np.asarray(f3(xx, yy, zz)) - np.asarray(f4(xx, yy, zz))))
    assert err_4 < tol  # native pass(4)
    assert err_5 < tol  # native pass(5)
