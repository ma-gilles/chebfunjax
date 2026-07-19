"""Core unit tests for Ballfunv vector-field methods (Fable 5).

Exercises the audit additions — real/imag/conj, iszero/isequal,
laplacian, scalar-field multiply and scalar divide — without needing any
MATLAB golden reference.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.ballfun.ballfunv import Ballfunv

TOL = 1e4 * float(np.finfo(np.float64).eps)


def _B(op, spherical=False):
    return Ballfun.from_function(op, spherical=spherical)


class TestBallfunvComplexParts:
    def test_real_imag_conj(self):
        f1 = _B(lambda x, y, z: x)
        f2 = _B(lambda x, y, z: 1j * z)
        f3 = _B(lambda x, y, z: jnp.cos(y) + 1j * jnp.sin(x))
        V = Ballfunv(f1, f2, f3)
        exact_real = Ballfunv(_B(lambda x, y, z: x),
                              _B(lambda x, y, z: 0.0 * x),
                              _B(lambda x, y, z: jnp.cos(y)))
        exact_imag = Ballfunv(_B(lambda x, y, z: 0.0 * x),
                              _B(lambda x, y, z: z),
                              _B(lambda x, y, z: jnp.sin(x)))
        assert (V.real() - exact_real).norm() < TOL
        assert (V.imag() - exact_imag).norm() < TOL
        # conj negates the imaginary parts.
        exact_conj = Ballfunv(_B(lambda x, y, z: x),
                              _B(lambda x, y, z: -1j * z),
                              _B(lambda x, y, z: jnp.cos(y) - 1j * jnp.sin(x)))
        assert (V.conj() - exact_conj).norm() < TOL


class TestBallfunvPredicatesCalculus:
    def test_iszero_isequal(self):
        f = _B(lambda x, y, z: 1.0 + 0.0 * x)
        F = Ballfunv(f, f, f)
        assert (F - F).iszero()
        assert F.isequal(F + F - F)

    def test_laplacian(self):
        f = _B(lambda x, y, z: x ** 2 + y ** 2 + z ** 2)
        V = Ballfunv(f, 2 * f, -f)
        exact = Ballfunv(_B(lambda x, y, z: 6.0 + 0.0 * x),
                         _B(lambda x, y, z: 12.0 + 0.0 * x),
                         _B(lambda x, y, z: -6.0 + 0.0 * x))
        assert (V.laplacian() - exact).norm() < 1e7 * float(
            np.finfo(np.float64).eps)

    def test_scalar_field_multiply_and_divide(self):
        V = Ballfunv(_B(lambda x, y, z: x), _B(lambda x, y, z: y),
                     _B(lambda x, y, z: z))
        fc = _B(lambda x, y, z: jnp.cos(y))
        exact = Ballfunv(_B(lambda x, y, z: x * jnp.cos(y)),
                         _B(lambda x, y, z: y * jnp.cos(y)),
                         _B(lambda x, y, z: z * jnp.cos(y)))
        assert (V * fc - exact).norm() < TOL
        assert (fc * V - exact).norm() < TOL
        assert (V / 2 - Ballfunv(_B(lambda x, y, z: x / 2),
                                 _B(lambda x, y, z: y / 2),
                                 _B(lambda x, y, z: z / 2))).norm() < TOL
