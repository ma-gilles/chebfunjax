"""The original continuous-norm predicate from test_complex.m, 7574c77.

Construct real, imaginary and independent complex references in source order.
The former sampled-surrogate checks are replaced by the native assertion.
"""

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3
from chebfunjax.chebpref import ChebfunPref


def test_native_complex():
    tol = 1000*ChebfunPref().cheb3Prefs.chebfun3eps
    f = chebfun3(lambda x, y, z: jnp.sin(x*y*z))
    g = chebfun3(lambda x, y, z: jnp.cos(x*y*z))
    h = chebfun3(lambda x, y, z: jnp.sin(x*y*z)+1j*jnp.cos(x*y*z))
    assert (h-Chebfun3.complex(f, g)).norm() < tol
