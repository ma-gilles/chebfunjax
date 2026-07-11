"""Port of MATLAB Chebfun tests/misc/test_jac2cheb.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_jac2cheb.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.special import eval_jacobi

from chebfunjax.utils.transforms import jac2cheb

TOL = 1e-11


class TestJac2cheb:
    def test_single_jacobi_mode_values(self):
        # jac2cheb(e_k, a, b) must be the Chebyshev coefficients of
        # P_k^{(a,b)}; check values against scipy's eval_jacobi.
        a, b = 0.3, -0.2
        xs = np.linspace(-0.95, 0.95, 40)
        for k in [0, 1, 4, 9]:
            e = jnp.zeros(12, dtype=jnp.float64).at[k].set(1.0)
            cc = np.asarray(jac2cheb(e, a, b))
            got = C.chebval(xs, cc)
            exact = eval_jacobi(k, a, b, xs)
            assert float(np.max(np.abs(got - exact))) < TOL, f"k={k}"
