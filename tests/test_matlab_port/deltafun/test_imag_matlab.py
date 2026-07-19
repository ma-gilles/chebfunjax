"""Port of MATLAB Chebfun tests/deltafun/test_imag.m (Fable 5).

``imag`` takes the imaginary part of both the funPart and the delta
magnitudes, then cleans; a delta-free result demotes to a bare Bndfun.  The
empty-Deltafun case (pass 1) is skipped: chebfunjax has no empty Deltafun
representation.

Provenance
----------
MATLAB source : tests/deltafun/test_imag.m
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


class TestDeltafunImag:
    def test_imag_empty(self):
        pytest.skip("chebfunjax has no empty Deltafun representation")

    def test_imag_of_real_delta_not_deltafun(self):
        # pass(2): ~isa(imag(d), 'deltafun') when deltaMag is real
        f = Bndfun.from_function(jnp.sin, DOM)
        d = Deltafun(f, jnp.array([0.0]), jnp.array([1.0]))
        assert not isinstance(d.imag(), Deltafun)

    def test_imag_of_imaginary_delta_is_deltafun(self):
        # pass(3): isa(imag(1i*d), 'deltafun')
        f = Bndfun.from_function(jnp.sin, DOM)
        d = Deltafun(f, jnp.array([0.0]), jnp.array([1.0]))
        h = (1j * d).imag()
        assert isinstance(h, Deltafun)

    def test_imag_magnitude_value(self):
        # pass(4): h.deltaMag == 1 for h = imag(1i*d)
        f = Bndfun.from_function(jnp.sin, DOM)
        d = Deltafun(f, jnp.array([0.0]), jnp.array([1.0]))
        h = (1j * d).imag()
        assert _ninf(h.delta_mags - np.array([[1.0]])) == 0.0

    def test_imag_of_complex_matrix(self):
        # pass(5): all(all(imag(d).deltaMag == B)) for deltaMag = A + 1i*B
        f = Bndfun.from_function(jnp.exp, DOM)
        A = np.random.rand(4, 4)
        B = np.random.rand(4, 4)
        d = Deltafun(f, jnp.array([-0.5, -0.25, 0.0, 1.0]),
                     jnp.asarray(A + 1j * B))
        h = d.imag()
        assert _ninf(np.asarray(h.delta_mags) - B) == 0.0
