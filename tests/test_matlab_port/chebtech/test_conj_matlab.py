"""All4 literal tests/chebtech/test_conj.m predicates, Chebfun7574c77.

Native checks use direct equal coefficient shapes and vector/matrix infinity
norms at the source bounds. Earlier padded/max-element probes are retained
separately as supplemental controls.

Provenance
----------
MATLAB source : tests/chebtech/test_conj.m
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
class TestChebtechConj:
    # FIXED (Fable 5, Big-Three array-valued epic): conj now exists on Chebtech.
    def test_conj_scalar(self, Tech):
        # pass(n, 1): conj(cos(x) + 1i*sin(x)) == cos(x) - 1i*sin(x).
        f = Tech.from_function(lambda x: jnp.cos(x) + 1j * jnp.sin(x))
        g = Tech.from_function(lambda x: jnp.cos(x) - 1j * jnp.sin(x))
        h = f.conj()
        assert _coeff_ninf_diff(h.coeffs, g.coeffs) < 10 * h.vscale * EPS

    def test_conj_array(self, Tech):
        # pass(n, 2): conj([cos+1i*sin, -exp(1i*x)]) == [cos-1i*sin, -conj(exp(1i*x))].
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.cos(x) + 1j * jnp.sin(x), -jnp.exp(1j * x)], axis=-1)
        )
        g = Tech.from_function(lambda x: jnp.cos(x) - 1j * jnp.sin(x))
        h = f.conj()
        gc = jnp.asarray(g.coeffs)
        target = jnp.stack([gc, -gc], axis=-1)
        assert _coeff_ninf_diff(h.coeffs, target) < 10 * h.vscale * EPS


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
@pytest.mark.parametrize("array", [False, True])
def test_native_tech_conj_literal_coefficients(Tech, array):
    """All4 literal sourcepredicates: directcoeffshape and actualinf norm."""
    if array:
        f = Tech.from_function(
            lambda x: jnp.stack((jnp.cos(x) + 1j * jnp.sin(x), -jnp.exp(1j * x)), axis=-1)
        )
    else:
        f = Tech.from_function(lambda x: jnp.cos(x) + 1j * jnp.sin(x))
    g = Tech.from_function(lambda x: jnp.cos(x) - 1j * jnp.sin(x))
    h = f.conj()
    expected = jnp.stack((g.coeffs, -g.coeffs), axis=-1) if array else g.coeffs
    # Native raw subtraction does not pad unequal coefficient lengths.
    assert h.coeffs.shape == expected.shape
    error = float(jnp.linalg.norm(h.coeffs - expected, ord=jnp.inf))
    bound = (
        10 * float(jnp.max(jnp.asarray(h.vscale) * EPS)) if array else 10 * float(h.vscale) * EPS
    )
    print(
        {
            "native_type": Tech.__name__,
            "array": array,
            "coefficient_shape": list(h.coeffs.shape),
            "error": error,
            "bound": bound,
        },
        flush=True,
    )
    assert error < bound
