"""Port of MATLAB Chebfun tests/deltafun/test_real.m (Fable 5).

``real`` takes the real part of both the funPart and the delta magnitudes,
then cleans; a delta-free result demotes to a bare Bndfun (mirroring MATLAB's
SMOOTHFUN demotion).  The empty-Deltafun case (pass 1) is skipped: chebfunjax
has no empty Deltafun representation.

Provenance
----------
MATLAB source : tests/deltafun/test_real.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.deltafun import Deltafun

DOM = Domain((-1.0, 1.0))


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestDeltafunReal:
    def test_real_empty(self):
        pytest.skip("chebfunjax has no empty Deltafun representation")

    def test_real_of_imag_delta_not_deltafun(self):
        # pass(2): ~isa(real(d), 'deltafun') when deltaMag = 1i
        f = Bndfun.from_function(jnp.sin, DOM)
        d = Deltafun(f, jnp.array([0.0]), jnp.array([1j]))
        assert not isinstance(d.real(), Deltafun)

    def test_real_of_imaginary_delta_is_deltafun(self):
        # pass(3): isa(real(1i*d), 'deltafun')
        f = Bndfun.from_function(jnp.sin, DOM)
        d = Deltafun(f, jnp.array([0.0]), jnp.array([1j]))
        h = (1j * d).real()
        assert isinstance(h, Deltafun)

    def test_real_magnitude_value(self):
        # pass(4): h.deltaMag == -1 for h = real(1i*d), deltaMag = 1i
        f = Bndfun.from_function(jnp.sin, DOM)
        d = Deltafun(f, jnp.array([0.0]), jnp.array([1j]))
        h = (1j * d).real()
        assert _ninf(h.delta_mags - np.array([[-1.0]])) == 0.0

    def test_real_of_complex_matrix(self):
        # pass(5): all(all(real(d).deltaMag == A)) for deltaMag = A + 1i*B
        f = Bndfun.from_function(jnp.exp, DOM)
        A = np.random.rand(4, 4)
        B = np.random.rand(4, 4)
        d = Deltafun(f, jnp.array([-0.5, -0.25, 0.0, 1.0]),
                     jnp.asarray(A + 1j * B))
        h = d.real()
        assert _ninf(np.asarray(h.delta_mags) - A) == 0.0
