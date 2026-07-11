"""Port of MATLAB Chebfun tests/chebfun/test_ellipj.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_ellipj.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.special import ellipj as sellipj

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


class TestChebfunEllipj:
    def test_sn_cn_dn_of_identity(self):
        m = 0.5
        u = cj.chebfun(lambda x: x, domain=(0.0, 2.0))
        sn, cn, dn = u.ellipj(m)
        xs = np.linspace(0.05, 1.95, 40)
        s, c, d, _ = sellipj(xs, m)
        for got, want in [(sn, s), (cn, c), (dn, d)]:
            err = np.abs(np.asarray(got(jnp.asarray(xs))) - want)
            assert float(np.max(err)) < 1e4 * EPS
