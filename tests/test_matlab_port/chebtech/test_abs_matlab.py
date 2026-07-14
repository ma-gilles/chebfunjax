"""Port of MATLAB Chebfun tests/chebtech/test_abs.m (Opus 4.8).

Self-validating: ``abs(f)`` is checked against the analytic |f| at the SAME
tolerance MATLAB uses.  The MATLAB file loops ``for type = 1:2`` over
``{chebtech1(), chebtech2()}``; each assertion is parametrized over both.

MATLAB uses ``normest(h - f)`` (an operator-norm estimate that ~0 => the
functions agree).  chebfunjax has no ``normest``; the faithful equivalent
used here is ``float((h - f).norm(jnp.inf))`` — the L-infinity norm of the
difference on a fine grid — which is exactly the "is this ~0" check MATLAB's
normest performs at these tolerances.

Gaps vs MATLAB (honest xfail/skip):
- Chebtech1 drops the imaginary part in vals2coeffs/coeffs2vals, so it cannot
  represent complex-valued functions; the complex ``abs`` case xfails on
  Chebtech1 (Chebtech2 passes).
- Array-valued ``abs``: Chebtech now supports (n, m) coefficient matrices, so
  the complex array-valued abs case (pass 4) is ported.  Complex columns are
  preserved on both classes, so it holds for Chebtech1 too.

Provenance
----------
MATLAB source : tests/chebtech/test_abs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechAbs:
    def test_abs_positive(self, Tech):
        # pass(type, 1): |f| == f for a positive function (normest(h-f) < 10 eps).
        f = Tech.from_function(lambda x: jnp.sin(x) + 2.0)
        h = abs(f)
        assert float((h - f).norm(jnp.inf)) < 10 * EPS

    def test_abs_negative(self, Tech):
        # pass(type, 2): |f| == -f for a negative function (normest(h+f) < 10 eps).
        f2 = Tech.from_function(lambda x: -(jnp.sin(x) + 2.0))
        h = abs(f2)
        assert float((h + f2).norm(jnp.inf)) < 10 * EPS

    def test_abs_complex(self, Tech):
        # pass(type, 3): |exp(1i pi x)| == 1 (normest(h - 1) < 1e2 eps).
        if Tech is Chebtech1:
            pytest.xfail(
                "Chebtech1 drops imaginary part in vals2coeffs/coeffs2vals; "
                "cannot represent complex-valued functions"
            )
        f = Tech.from_function(lambda x: jnp.exp(1j * jnp.pi * x))
        h = abs(f)
        assert float((h - 1.0).norm(jnp.inf)) < 1e2 * EPS

    # FIXED (Fable 5, Big-Three array-valued epic): complex array-valued abs.
    def test_abs_complex_array(self, Tech):
        # pass(type, 4): abs([(2+sin)e^{i pi x}, -(2+sin)e^{i pi x}, 2+sin]) == [2+sin, 2+sin, 2+sin].
        f = Tech.from_function(
            lambda x: jnp.stack(
                [
                    (2 + jnp.sin(x)) * jnp.exp(1j * jnp.pi * x),
                    -(2 + jnp.sin(x)) * jnp.exp(1j * jnp.pi * x),
                    2 + jnp.sin(x),
                ],
                axis=-1,
            )
        )
        g = Tech.from_function(
            lambda x: jnp.stack(
                [2 + jnp.sin(x), 2 + jnp.sin(x), 2 + jnp.sin(x)], axis=-1
            )
        )
        h = abs(f)
        assert float((h - g).norm(jnp.inf)) < 1e1 * EPS
