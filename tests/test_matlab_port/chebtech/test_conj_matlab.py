"""Port of MATLAB Chebfun tests/chebtech/test_conj.m (Fable 5).

Chebtech now provides a ``conj()`` method and represents complex-valued
functions (scalar and array-valued) via ``from_function``, so both MATLAB
assertions are ported as real checks against direct construction.

MATLAB ``norm(h.coeffs - g.coeffs, inf)`` -> the inf-norm of the coefficient
difference after zero-padding to a common length.

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
            lambda x: jnp.stack(
                [jnp.cos(x) + 1j * jnp.sin(x), -jnp.exp(1j * x)], axis=-1
            )
        )
        g = Tech.from_function(lambda x: jnp.cos(x) - 1j * jnp.sin(x))
        h = f.conj()
        gc = jnp.asarray(g.coeffs)
        target = jnp.stack([gc, -gc], axis=-1)
        assert _coeff_ninf_diff(h.coeffs, target) < 10 * h.vscale * EPS
