"""Source coefficient predicates for Chebfun tests/chebtech/test_imag.m.

Chebfun commit 7574c77. All eight native predicates are runnable: new literal
coefficient comparisons cover source clauses 1/2 for both tech classes, and
the existing zero-result checks cover clauses 3/4. Historical padded probes
remain supplemental. REAL uses direct coefficient shapes; IMAG explicitly
prolongs the result to the reference length. Native array comparisons use
the matrix infinity norm and preserve the original vscale/epsilon bounds.
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
class TestChebtechImag:
    # FIXED (Fable 5, Big-Three array-valued epic): imag now exists on Chebtech.
    def test_imag_scalar(self, Tech):
        # pass(n, 1): imag(exp(x) + 1i*sin(x)) == sin(x).
        f = Tech.from_function(lambda x: jnp.exp(x) + 1j * jnp.sin(x))
        g = Tech.from_function(lambda x: jnp.sin(x))
        h = f.imag()
        assert _coeff_ninf_diff(h.coeffs, g.coeffs) < 10 * h.vscale * EPS

    def test_imag_array(self, Tech):
        # pass(n, 2): imag([exp(x)+1i*sin(x), -exp(1i*x)]) == [sin(x), -imag(exp(1i*x))].
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.exp(x) + 1j * jnp.sin(x), -jnp.exp(1j * x)], axis=-1)
        )
        g = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), -jnp.imag(jnp.exp(1j * x))], axis=-1)
        )
        h = f.imag()
        assert _coeff_ninf_diff(h.coeffs, g.coeffs) < 10 * h.vscale * EPS

    def test_imag_of_real_is_zero(self, Tech):
        # pass(n, 3): imag(cos(x)) collapses to a single coeff.
        f = Tech.from_function(lambda x: jnp.cos(x))
        g = f.imag()
        assert g.coeffs.size == 1

    def test_imag_array_of_real_is_zero(self, Tech):
        # pass(n, 4): imag([cos sin exp]) is a [1, 3] block of zeros.
        f = Tech.from_function(lambda x: jnp.stack([jnp.cos(x), jnp.sin(x), jnp.exp(x)], axis=-1))
        g = f.imag()
        assert g.coeffs.shape == (1, 3) and bool(jnp.all(jnp.asarray(g.coeffs) == 0))


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
@pytest.mark.parametrize("array", [False, True])
def test_native_tech_imag_literal_coefficients(Tech, array):
    """Native1/2×two techclasses; existingzeroresult methods cover3/4."""
    if array:
        f = Tech.from_function(
            lambda x: jnp.stack((jnp.exp(x) + 1j * jnp.sin(x), -jnp.exp(1j * x)), axis=-1)
        )
        g = Tech.from_function(
            lambda x: jnp.stack((jnp.sin(x), -jnp.imag(jnp.exp(1j * x))), axis=-1)
        )
    else:
        f = Tech.from_function(lambda x: jnp.exp(x) + 1j * jnp.sin(x))
        g = Tech.from_function(lambda x: jnp.sin(x))
    h = f.imag()
    before_shape = h.coeffs.shape
    h = h.prolong(len(g))
    print(
        {
            "native_type": Tech.__name__,
            "component": "imag",
            "array": array,
            "input_shape": list(f.coeffs.shape),
            "unprolonged_shape": list(before_shape),
            "actual_shape": list(h.coeffs.shape),
            "reference_shape": list(g.coeffs.shape),
        },
        flush=True,
    )
    assert h.coeffs.shape == g.coeffs.shape
    error = float(jnp.linalg.norm(h.coeffs - g.coeffs, ord=jnp.inf))
    bound = (
        10 * float(jnp.max(jnp.asarray(h.vscale) * EPS)) if array else 10 * float(h.vscale) * EPS
    )
    print(
        {
            "native_type": Tech.__name__,
            "component": "imag",
            "array": array,
            "error": error,
            "bound": bound,
        },
        flush=True,
    )
    assert error < bound
