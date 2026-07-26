"""Port of MATLAB Chebfun tests/chebtech/test_real.m (Fable 5).

Chebtech now provides a ``real()`` method and represents complex-valued
functions (scalar and array-valued) via ``from_function``, so the MATLAB
assertions are ported as real checks against direct construction.

MATLAB ``norm(h.coeffs - g.coeffs, inf)`` -> the inf-norm of the coefficient
difference after zero-padding to a common length.

FIXED (Fable 5): ``vals2coeffs`` now mirrors MATLAB's purely-imaginary branch
(``coeffs = 1i*real(ifft(imag(tmp)))``), so a purely imaginary construction has
a bit-exactly-zero real part and ``real()`` collapses the all-zero result to a
length-1 constant on Chebtech2 as well.  Passes 3 and 4 hold on both techs.

Provenance
----------
MATLAB source : tests/chebtech/test_real.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)

def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _coeff_ninf_diff(a, b):
    """||a - b||_inf after zero-padding along axis 0 to a common length."""
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    n = max(a.shape[0], b.shape[0])
    dt = jnp.result_type(a.dtype, b.dtype)
    ap = jnp.zeros((n,) + a.shape[1:], dt).at[: a.shape[0]].set(a)
    bp = jnp.zeros((n,) + b.shape[1:], dt).at[: b.shape[0]].set(b)
    return _ninf(ap - bp)


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechReal:
    # FIXED (Fable 5, Big-Three array-valued epic): real now exists on Chebtech.
    def test_real_scalar(self, Tech):
        # pass(n, 1): real(exp(1i*x) + 1i*sin(x)) == cos(x).
        f = Tech.from_function(lambda x: jnp.exp(1j * x) + 1j * jnp.sin(x))
        g = Tech.from_function(lambda x: jnp.cos(x))
        h = f.real()
        assert _coeff_ninf_diff(h.coeffs, g.coeffs) < 10 * h.vscale * EPS

    def test_real_array(self, Tech):
        # pass(n, 2): real([exp(1i*x)+1i*sin(x), -exp(1i*x)]) == [cos(x), -real(exp(1i*x))].
        f = Tech.from_function(
            lambda x: jnp.stack(
                [jnp.exp(1j * x) + 1j * jnp.sin(x), -jnp.exp(1j * x)], axis=-1
            )
        )
        g = Tech.from_function(
            lambda x: jnp.stack(
                [jnp.cos(x), -jnp.real(jnp.exp(1j * x))], axis=-1
            )
        )
        h = f.real()
        assert _coeff_ninf_diff(h.coeffs, g.coeffs) < 10 * h.vscale * EPS

    def test_real_of_imaginary_is_zero(self, Tech):
        # pass(n, 3): real(1i*cos(x)) has a single zero coeff.
        # FIXED (Fable 5): vals2coeffs now mirrors MATLAB's purely-imaginary
        # branch (coeffs = 1i*real(ifft(imag(tmp)))), so the real part of a
        # purely imaginary construction is bit-exactly zero and real()
        # collapses to a length-1 constant on Chebtech2 as well.
        f = Tech.from_function(lambda x: 1j * jnp.cos(x))
        g = f.real()
        assert g.coeffs.size == 1 and bool(jnp.all(jnp.asarray(g.coeffs) == 0))

    def test_real_array_of_imaginary_is_zero(self, Tech):
        # pass(n, 4): real(1i*[cos sin exp]) is a [1, 3] block of zeros.
        # FIXED (Fable 5): purely-imaginary vals2coeffs branch (see pass 3).
        f = Tech.from_function(
            lambda x: 1j * jnp.stack([jnp.cos(x), jnp.sin(x), jnp.exp(x)], axis=-1)
        )
        g = f.real()
        assert g.coeffs.shape == (1, 3) and bool(
            jnp.all(jnp.asarray(g.coeffs) == 0)
        )
