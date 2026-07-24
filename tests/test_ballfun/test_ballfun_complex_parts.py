"""Core mirrors for structural Ballfun real/imag/conj + complex ctor.

Pins the coefficient-space conjugation identity (index reflection with
Fourier padding) and the constructor fix for complex-valued inputs
(the r=0 BMC mean must keep its imaginary part; a real() cast used to
plant a jump at the origin that drove the radial grid to max_sample).

Provenance
----------
Mirrors of MATLAB @ballfun/{real,imag,conj}.m; Chebfun commit 7574c77.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp

from chebfunjax.ballfun.ballfun import Ballfun


class TestBallfunComplexParts:
    def test_complex_ctor_converges(self):
        # 1j*cos(y) has f(0) = 1j: the r=0 mean must stay complex.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            f = Ballfun.from_function(lambda x, y, z: 1j * jnp.cos(y))
        assert f.coeffs.shape[0] < 40  # was 1643 with the real() cast

    def test_real_imag_machine_precision(self):
        f = Ballfun.from_function(
            lambda x, y, z: jnp.sin(z) + 1j * jnp.cos(y))
        re_exact = Ballfun.from_function(lambda x, y, z: jnp.sin(z))
        im_exact = Ballfun.from_function(lambda x, y, z: jnp.cos(y))
        assert float((f.real() - re_exact).norm()) < 1e-13
        assert float((f.imag() - im_exact).norm()) < 1e-13

    def test_conj_roundtrip(self):
        f = Ballfun.from_function(lambda x, y, z: x + 1j * y)
        g = f.conj().conj()
        assert float((g - f).norm()) < 1e-13

    def test_real_input_short_circuits(self):
        f = Ballfun.from_function(lambda x, y, z: x * y)
        assert f.real() is f
        assert f.conj() is f
        assert float(f.imag().norm()) == 0.0
