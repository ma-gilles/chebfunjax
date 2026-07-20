"""Port of MATLAB Chebfun tests/diskfun/test_cdr.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_cdr.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun

_EPS = float(jnp.finfo(jnp.float64).eps)
_TOL = 1000 * _EPS


class TestDiskfunCdr:
    def test_reconstruction(self):
        # MATLAB: f = diskfun(@(x,y) exp(-((x-.4).^2+(y-.9).^2)));
        # [C,D,R] = cdr(f);  norm(f - C*D*R') < tol.
        f = Diskfun.from_function(
            lambda t, r: jnp.exp(-((r * jnp.cos(t) - 0.4) ** 2
                                   + (r * jnp.sin(t) - 0.9) ** 2))
        )
        C, D, R = f.cdr()
        D = np.asarray(D)

        # Sample on a polar grid and rebuild f = sum_j C[j](r) D[j,j] R[j](theta).
        th = np.linspace(-np.pi, np.pi, 41)
        rr = np.linspace(0.0, 1.0, 33)
        tt, rrr = np.meshgrid(th, rr)
        tt_j = jnp.asarray(tt)
        rr_j = jnp.asarray(rrr)

        recon = np.zeros_like(tt, dtype=np.complex128)
        for j in range(len(C)):
            cj = np.asarray(C[j](rr_j))            # column at r in [0,1]
            rj = np.asarray(R[j](tt_j / np.pi))    # row at th_ref = theta/pi
            recon += D[j, j] * cj * rj

        f_vals = np.asarray(f(tt_j, rr_j))
        assert np.linalg.norm(recon.real - f_vals) < _TOL
        assert np.max(np.abs(recon.imag)) < _TOL

    def test_diagonal_is_inv_pivots(self):
        f = Diskfun.from_function(lambda t, r: r * jnp.sin(t))
        _, D, _ = f.cdr()
        D = np.asarray(D)
        piv = np.asarray(f.pivots)
        assert np.allclose(np.diag(D), 1.0 / piv, rtol=0, atol=_TOL)
