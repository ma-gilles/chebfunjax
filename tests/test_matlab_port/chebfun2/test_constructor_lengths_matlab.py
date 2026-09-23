"""chebfun2 constructor parity with MATLAB R2025b (Fable 5).

MATLAB reference values were produced with Chebfun 7574c77 for the
functions below (``[m, n] = length(f)``, ``rank(f)``, ``f.pivotValues``,
``f.pivotLocations``).  The Greeks example's payoff-density slice on
[0, 5000] x [0.41, 0.5] exercises the sampleTest restart (MATLAB restarts
phase 1 on a 17^2 and then a 33^2 grid) and the gradient-scaled getTol.
Slice lengths are pinned to within 2 coefficients: the final
``simplify(g, relTol)`` chops at a plateau of round-off noise, and the
FFT round-off of MATLAB (FFTW) and JAX differ by ~5e-17, which moves the
standardChop cutoff by up to 2 in either direction for
exp(-20((x-.3)^2 + 2y^2)), depending even on whether the handle uses
numpy or jax.numpy exp (MATLAB's
own standardChop applied to our coefficients returns our cutoffs).

Provenance
----------
MATLAB source : @chebfun2/constructor.m, @separableApprox/sampleTest.m,
    @separableApprox/simplify.m, @chebtech/standardCheck.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.separable_approx import _halton_1d

jax.config.update("jax_enable_x64", True)


def _len(f):
    m, n = f.length()
    return int(m), int(n)


class TestConstructorLengths:
    def test_halton_matches_matlab(self):
        # MATLAB halton (sampleTest.m): radical inverses of 1, 2, 3, ...
        np.testing.assert_allclose(_halton_1d(6, 2),
                                   [0.5, 0.25, 0.75, 0.125, 0.625, 0.375])
        np.testing.assert_allclose(_halton_1d(4, 3),
                                   [1 / 3, 2 / 3, 1 / 9, 4 / 9])

    def test_reference_functions(self):
        f = Chebfun2.from_function(lambda x, y: 1 / (1 + x ** 2 + 2 * y ** 2))
        assert _len(f) == (43, 57) and len(f.approx.pivots) == 10
        f = Chebfun2.from_function(lambda x, y: jnp.cos(10 * x * y) + jnp.sin(3 * y))
        assert _len(f) == (35, 35) and len(f.approx.pivots) == 13
        f = Chebfun2.from_function(lambda x, y: jnp.sin(x + y) * jnp.cos(x * y))
        assert _len(f) == (19, 19) and len(f.approx.pivots) == 12
        f = Chebfun2.from_function(
            lambda x, y: jnp.exp(-((x - .3) ** 2 + 2 * y ** 2) * 20))
        m, n = _len(f)
        assert abs(m - 61) <= 2 and abs(n - 85) <= 2 and len(f.approx.pivots) == 1

    def test_greeks_slice_sample_test_restart(self):
        St, tau, r, K = 100.0, 0.5, 0.01, 100.0

        def pdf(ST, St, vol, tau, r):
            return (jnp.exp(-(jnp.log(ST / St) - (r - 0.5 * vol ** 2) * tau) ** 2
                            / (2 * vol ** 2 * tau))
                    / (vol * ST * jnp.sqrt(2 * jnp.pi * tau)))

        f = Chebfun2.from_function(
            lambda S, v: jnp.exp(-r * tau) * pdf(S + K, St, v, tau, r),
            domain=(0.0, 5000.0, 0.41, 0.5))
        piv = 1.0 / np.asarray(f.approx.pivots)
        ref = [0.01358088512, 0.0008556977, -2.890258922e-05, 5.08663719e-07,
               -1.173040364e-07, 7.464556832e-10, -1.019179687e-11]
        np.testing.assert_allclose(piv, ref, rtol=1e-8)
        locs = np.asarray(f.pivot_locations)
        np.testing.assert_allclose(locs[:, 0], [0, 48.0368, 107.6492, 12.0382,
                                                190.3012, 295.1968, 421.3260],
                                   atol=5e-4)
        np.testing.assert_allclose(locs[:, 1], [0.4100, 0.5000, 0.4506, 0.4265,
                                                0.4835, 0.4681, 0.4153],
                                   atol=5e-4)
        m, n = _len(f)
        assert abs(m - 165) <= 2 and n == 14
