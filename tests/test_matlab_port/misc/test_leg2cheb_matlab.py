"""Port of MATLAB Chebfun tests/misc/test_leg2cheb.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_leg2cheb.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.special import eval_legendre

from chebfunjax.utils.transforms import leg2cheb

TOL = 1e-11


class TestLeg2cheb:
    def test_single_legendre_mode_values(self):
        xs = np.linspace(-0.95, 0.95, 40)
        for k in [0, 1, 5, 10]:
            e = jnp.zeros(12, dtype=jnp.float64).at[k].set(1.0)
            cc = np.asarray(leg2cheb(e))
            got = C.chebval(xs, cc)
            exact = eval_legendre(k, xs)
            assert float(np.max(np.abs(got - exact))) < TOL, f"k={k}"
