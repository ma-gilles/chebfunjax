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
import pytest

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

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_tail(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_ishappy(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(
        reason="chebfunjax happiness_check has no sampleTest option; it cannot detect "
        "the aliasing-fooled happy case that MATLAB's sampleTest=0 branch checks"
    )
    def test_aliased_happy_without_sampletest(self):
        raise AssertionError("sampleTest preference not implemented")

    @pytest.mark.xfail(
        reason="chebfunjax happiness_check has no sampleTest option to reject the "
        "aliasing-fooled case"
    )
    def test_unhappy_with_sampletest(self):
        raise AssertionError("sampleTest preference not implemented")
