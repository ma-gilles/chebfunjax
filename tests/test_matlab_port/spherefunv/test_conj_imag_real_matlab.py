"""Port of MATLAB Chebfun tests/spherefunv/test_conj_imag_real.m (Fable 5).

FIXED: Spherefun.gradient now returns a 3-Cartesian-component Spherefunv, so
conj/real/imag can be exercised on the gradient field exactly as MATLAB does.

Provenance
----------
MATLAB source : tests/spherefunv/test_conj_imag_real.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun

from ._helpers import EPS

TOL = 100 * EPS

# rng(7) point in MATLAB is just an interior sample; any interior (lam, th)
# exercises the identical algebra.
LAM0, TH0 = jnp.asarray(0.5488135), jnp.asarray(0.7151894)


def _vnorm_at(u, lam, th) -> float:
    vals = u(lam, th)
    return float(np.sqrt(sum(abs(np.asarray(v)) ** 2 for v in vals)))


def _vdiff_at(a, b, lam, th) -> float:
    av, bv = a(lam, th), b(lam, th)
    return float(np.sqrt(sum(abs(np.asarray(u) - np.asarray(v)) ** 2
                             for u, v in zip(av, bv))))


class TestSpherefunvConjImagReal:
    def test_conj_real_imag_of_gradient(self):
        f = Spherefun.from_function(
            lambda lam, th: jnp.cos((jnp.cos(lam) * jnp.sin(th) + 0.1)
                                    * (jnp.sin(lam) * jnp.sin(th))
                                    * jnp.cos(th)))
        u = f.gradient()

        # pass(1): conj(u) == u for a real field.
        assert _vdiff_at(u, u.conj(), LAM0, TH0) < TOL
        # pass(2): real(u) == u for a real field.
        assert _vdiff_at(u, u.real(), LAM0, TH0) < TOL
        # pass(3): imag(u) == 0 for a real field.
        assert _vnorm_at(u.imag(), LAM0, TH0) < TOL
