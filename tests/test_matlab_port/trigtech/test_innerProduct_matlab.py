"""Port of MATLAB Chebfun tests/trigtech/test_innerProduct.m (Fable 5).

FIXED: Trigtech.innerProduct added in the Fable 5 audit (exact Fourier
orthogonality with MATLAB's isequal force-real branch).

Provenance
----------
MATLAB source : tests/trigtech/test_innerProduct.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)
ALPHA = -0.194758928283640 + 0.075474485412665j
BETA = -0.526634844879922 - 0.685484380523668j


def _tt(f):
    return Trigtech.from_function(f)


class TestTrigtechInnerProduct:
    def test_orthogonal_modes(self):
        f = _tt(lambda x: jnp.sin(2 * jnp.pi * x))
        g = _tt(lambda x: jnp.cos(2 * jnp.pi * x))
        assert abs(complex(f.innerProduct(g))) \
            < 10 * EPS * max(f.vscale, g.vscale)
        g4 = _tt(lambda x: jnp.cos(4 * jnp.pi * x))
        assert abs(complex(f.innerProduct(g4))) \
            < 10 * EPS * max(f.vscale, g4.vscale)

    def test_smooth_pair_reference(self):
        # <exp(cos(pi x)), cos(2 pi x)> via scipy quadrature
        from scipy.integrate import quad
        f = _tt(lambda x: jnp.exp(jnp.cos(jnp.pi * x)))
        g = _tt(lambda x: jnp.cos(2 * jnp.pi * x))
        ref = quad(lambda x: np.exp(np.cos(np.pi * x))
                   * np.cos(2 * np.pi * x), -1, 1,
                   epsabs=1e-14, epsrel=1e-14)[0]
        assert abs(float(jnp.real(f.innerProduct(g))) - ref) < 1e-12

    def test_linearity_and_conjugate_symmetry(self):
        f = _tt(lambda x: jnp.exp(1j * jnp.pi * x))
        g = _tt(lambda x: jnp.cos(jnp.pi * x))
        ip_fg = complex(f.innerProduct(g))
        ip_gf = complex(g.innerProduct(f))
        assert abs(ip_fg - np.conj(ip_gf)) < 100 * EPS
        h = _tt(lambda x: ALPHA * jnp.exp(1j * jnp.pi * x))
        assert abs(complex(h.innerProduct(g))
                   - np.conj(ALPHA) * ip_fg) < 100 * EPS

    def test_self_inner_real_nonnegative(self):
        g = _tt(lambda x: jnp.exp(1j * jnp.pi * x) * (1 + 0.3j))
        ip = g.innerProduct(g)
        v = complex(ip)
        assert abs(v.imag) < 100 * EPS and v.real > 0
