"""Port of MATLAB Chebfun tests/trigtech/test_happinessCheck.m (Opus 4.8).

The happiness check must report a resolved (happy) representation and the
tail location (number of retained Fourier coefficients).  For a pure
frequency sin(omega*pi*x) sampled on a 33-point grid the tail is exactly
2*omega+1.

Provenance
----------
MATLAB source : tests/trigtech/test_happinessCheck.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.tech.trigtech import Trigtech, trigpts


class TestTrigtechHappinessCheck:
    def _check(self, fop, npts=33):
        x = trigpts(npts)
        vals = fop(x)
        c = Trigtech.vals2coeffs(vals)
        return Trigtech.happiness_check(c, vals)

    def test_scalar_tail(self):
        omega = 8
        ishappy, tail = self._check(lambda x: jnp.sin(omega * jnp.pi * x))
        assert tail == 2 * omega + 1

    def test_scalar_ishappy(self):
        omega = 8
        ishappy, tail = self._check(lambda x: jnp.sin(omega * jnp.pi * x))
        assert ishappy

    def test_array_tail(self):
        # pass(3): array-valued [sin(pi x) cos(3 pi x) (sin(7pi x)+cos(7pi x))]
        # at 33 pts => tail == 2*omega+1 with omega=7.
        # FIXED (Fable 5, Big-Three array-valued epic): happiness_check on (n, m)
        # coeffs takes the per-column max cutoff.
        omega = 7

        def fop(x):
            return jnp.stack(
                [
                    jnp.sin(jnp.pi * x),
                    jnp.cos((omega // 2) * jnp.pi * x),
                    jnp.sin(omega * jnp.pi * x) + jnp.cos(omega * jnp.pi * x),
                ],
                axis=-1,
            )

        ishappy, tail = self._check(fop)
        assert tail == 2 * omega + 1

    def test_array_ishappy(self):
        # pass(4): array-valued case is happy.
        # FIXED (Fable 5, Big-Three array-valued epic).
        omega = 7

        def fop(x):
            return jnp.stack(
                [
                    jnp.sin(jnp.pi * x),
                    jnp.cos((omega // 2) * jnp.pi * x),
                    jnp.sin(omega * jnp.pi * x) + jnp.cos(omega * jnp.pi * x),
                ],
                axis=-1,
            )

        ishappy, tail = self._check(fop)
        assert ishappy

    def test_aliased_happy_without_sampletest(self):
        # pass(5): sin((k+m+1)*pi*x) sampled on k+1 points aliases to look like
        # a resolved frequency, so the chop-only check is (wrongly) happy with
        # tail == 2*m+1.  FIXED (Fable 5): sampleTest ported (see pass 6).
        k = 4 * 8
        m = k // 4

        def fop(x):
            return jnp.sin((k + m + 1) * jnp.pi * x)

        x = trigpts(k + 1)
        vals = fop(x)
        c = Trigtech.vals2coeffs(vals)
        ishappy, tail = Trigtech.happiness_check(c, vals)  # sampleTest off
        assert ishappy and tail == 2 * m + 1

    def test_unhappy_with_sampletest(self):
        # pass(6): with sampleTest on (op supplied), the aliasing is detected
        # and the representation is unhappy with tail reverted to k+1 == 33.
        # FIXED (Fable 5): Trigtech.happiness_check gained an `op=` sample test
        # (MATLAB @trigtech/sampleTest.m).
        k = 4 * 8
        m = k // 4

        def fop(x):
            return jnp.sin((k + m + 1) * jnp.pi * x)

        x = trigpts(k + 1)
        vals = fop(x)
        c = Trigtech.vals2coeffs(vals)
        ishappy, tail = Trigtech.happiness_check(c, vals, op=fop)  # sampleTest
        assert (not ishappy) and tail == k + 1
