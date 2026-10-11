"""Eight literal native ABS predicates, Chebfun commit7574c77.

Source tests/chebtech/test_abs.m uses normest for all scalar/array cases.
The original eight L-infinity probes are retained separately below.
Default from_function adapts native make with default tech preferences.
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
        # FIXED: Chebtech1 vals2coeffs/coeffs2vals now split real/imag parts,
        # so complex-valued functions are represented on both tech classes.
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
            lambda x: jnp.stack([2 + jnp.sin(x), 2 + jnp.sin(x), 2 + jnp.sin(x)], axis=-1)
        )
        h = abs(f)
        assert float((h - g).norm(jnp.inf)) < 1e1 * EPS


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
@pytest.mark.parametrize("branch", ["positive", "negative", "complex", "array"])
def test_native_abs_normest(Tech, branch):
    if branch == "positive":
        f = Tech.from_function(lambda x: jnp.sin(x) + 2)
        difference = abs(f) - f
        bound = 10 * EPS
    elif branch == "negative":
        f = Tech.from_function(lambda x: -(jnp.sin(x) + 2))
        difference = abs(f) + f
        bound = 10 * EPS
    elif branch == "complex":
        f = Tech.from_function(lambda x: jnp.exp(1j * jnp.pi * x))
        difference = abs(f) - 1
        bound = 100 * EPS
    else:

        def op(x):
            z = (2 + jnp.sin(x)) * jnp.exp(1j * jnp.pi * x)
            return jnp.stack((z, -z, 2 + jnp.sin(x)), axis=-1)

        f = Tech.from_function(op)
        g = Tech.from_function(lambda x: jnp.stack((2 + jnp.sin(x),) * 3, axis=-1))
        difference = abs(f) - g
        bound = 10 * EPS
    error = float(difference.normest())
    print(
        {"native_tech": Tech.__name__, "branch": branch, "error": error, "bound": bound}, flush=True
    )
    assert error < bound
