"""Port of MATLAB Chebfun tests/chebtech/test_mrdivide.m (Fable 5).

Array-valued techs and a tech-level ``qr``/``mrdivide`` both exist now, so
every MATLAB assertion is ported directly at MATLAB's tolerances.  The MATLAB
file loops ``for n = 1:2`` over ``{chebtech1(), chebtech2()}``; we parametrize
over ``[Chebtech1, Chebtech2]``.

MATLAB overloads ``/`` on both operand orders.  Python's ``__truediv__`` is
already elementwise (MATLAB ``./``) on techs, so the ``double / chebtech``
branch of ``mrdivide.m`` is reached through the ``rmrdivide`` static method.

No gaps: all nine MATLAB passes are exercised.

Provenance
----------
MATLAB source : tests/chebtech/test_mrdivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)

# MATLAB: seedRNG(6178); x = 2*rand(100,1) - 1.
X = jnp.asarray(np.linspace(-1.0, 1.0, 100))

# MATLAB: alpha = -0.194758928283640 + 0.075474485412665i.
ALPHA = -0.194758928283640 + 0.075474485412665j

BOTH = [Chebtech1, Chebtech2]


def _sin_cos(x):
    return jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)


def _max_abs(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestChebtechMrdivide:
    @pytest.mark.parametrize("Tech", BOTH)
    def test_divide_by_zero_is_nan(self, Tech):
        # pass(n, 1): isnan(f / 0).
        f = Tech.from_function(_sin_cos)
        assert bool(jnp.all(jnp.isnan(f.mrdivide(0).coeffs)))

    @pytest.mark.parametrize("Tech", BOTH)
    def test_divide_by_complex_scalar(self, Tech):
        # pass(n, 2): f / alpha.
        f = Tech.from_function(_sin_cos)
        g = f.mrdivide(ALPHA)
        assert _max_abs(g(X) - _sin_cos(X) / ALPHA) < 10 * g.vscale * EPS

    @pytest.mark.parametrize("Tech", BOTH)
    def test_divide_by_identity(self, Tech):
        # pass(n, 3): g = f / eye(2); g*eye(2) == f.
        f = Tech.from_function(_sin_cos)
        eye2 = jnp.eye(2)
        g = f.mrdivide(eye2)
        err = (g @ eye2) - f
        assert _max_abs(err(X)) < 10 * g.vscale * EPS

    @pytest.mark.parametrize("Tech", BOTH)
    def test_nontrivial_least_squares(self, Tech):
        # pass(n, 4): g = f / [1 1] == (sin + cos)/2.
        f = Tech.from_function(_sin_cos)
        g = f.mrdivide(jnp.array([[1.0, 1.0]]))
        exact = (jnp.sin(X) + jnp.cos(X)) / 2.0
        assert _max_abs(g(X) - exact) < 10 * g.vscale * EPS

    @pytest.mark.parametrize("Tech", BOTH)
    def test_scalar_divided_by_tech(self, Tech):
        # pass(n, 5): g = alpha / f; innerProduct(f, g) == alpha.
        f = Tech.from_function(jnp.sin)
        g = Tech.rmrdivide(ALPHA, f)
        ip = complex(jnp.reshape(jnp.asarray(f.inner(g)), ()))
        assert abs(ip - ALPHA) < 10 * g.vscale * EPS

    @pytest.mark.parametrize("Tech", BOTH)
    def test_row_vector_divided_by_tech(self, Tech):
        # pass(n, 6): [1 1] / [sin(2*pi*x) cos(2*pi*x)].
        f = Tech.from_function(
            lambda x: jnp.stack(
                [jnp.sin(2 * jnp.pi * x), jnp.cos(2 * jnp.pi * x)], axis=-1))
        g = Tech.rmrdivide(jnp.array([[1.0, 1.0]]), f)
        exact = jnp.sin(2 * jnp.pi * X) + jnp.cos(2 * jnp.pi * X)
        assert _max_abs(g(X) - exact) < 1e2 * g.vscale * EPS

    @pytest.mark.parametrize("Tech", BOTH)
    def test_error_dimension_mismatch(self, Tech):
        # pass(n, 7): f / [1 2 3] raises mrdivide:size.
        f = Tech.from_function(
            lambda x: jnp.stack(
                [jnp.sin(2 * jnp.pi * x), jnp.cos(2 * jnp.pi * x)], axis=-1))
        with pytest.raises(ValueError, match="mrdivide:size"):
            f.mrdivide(jnp.array([[1.0, 2.0, 3.0]]))

    @pytest.mark.parametrize("Tech", BOTH)
    def test_error_tech_divided_by_tech(self, Tech):
        # pass(n, 8): f / g raises chebtechDivChebtech.
        f = Tech.from_function(jnp.sin)
        g = Tech.from_function(jnp.cos)
        with pytest.raises(ValueError, match="chebtechDivChebtech"):
            f.mrdivide(g)

    @pytest.mark.parametrize("Tech", BOTH)
    def test_error_bad_argument_type(self, Tech):
        # pass(n, 9): f / true raises mrdivide:badArg.
        f = Tech.from_function(jnp.sin)
        with pytest.raises(ValueError, match="mrdivide:badArg"):
            f.mrdivide(True)
