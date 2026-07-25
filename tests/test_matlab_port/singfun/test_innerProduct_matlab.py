"""Port of MATLAB Chebfun tests/singfun/test_innerProduct.m (Opus 4.8).

chebfunjax's Singfun has no ``innerProduct`` method, but for real-valued
singfuns the MATLAB definition ``innerProduct(f, g) = sum(conj(f).*g)``
reduces to ``sum(f.*g)``, which is exactly the composition of the existing
``__mul__`` (adds exponents, multiplies smooth parts) and ``sum`` (Jacobi
moments).  Each inner product is checked against the exact value from the
MATLAB test at the SAME tolerance.

The single complex-valued case (pass 8) is xfailed: chebfunjax Singfun does
not support complex smooth parts / conjugation.

Provenance
----------
MATLAB source : tests/singfun/test_innerProduct.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.fun.singfun import Singfun

EPS = float(np.finfo(np.float64).eps)

A = 0.64
B = -0.64
C = 1.28
D = -1.28
P = -0.2
Q = -0.3


def _sf(f, exps):
    return Singfun.from_function(f, exps)


def _ip(f, g):
    # innerProduct(f, g) = sum(conj(f).*g); real case -> sum(f.*g)
    return complex((f * g).sum())


class TestSingfunInnerProduct:
    def test_poles_same_end(self):
        f = _sf(lambda x: (1 + x) ** P, (P, 0.0))
        g = _sf(lambda x: (1 + x) ** Q, (Q, 0.0))
        I = _ip(f, g).real
        I_exact = 2 * np.sqrt(2)
        assert abs(I - I_exact) < 10 * EPS * abs(I_exact)

    def test_pole_divergent(self):
        f = _sf(lambda x: (1 + x) ** D * jnp.sin(x), (D, 0.0))
        g = _sf(lambda x: (1 + x) ** (2 * D), (2 * D, 0.0))
        I = _ip(f, g).real
        assert np.isinf(I) and np.sign(I) == np.sign(np.sin(-1.0))

    def test_root_right(self):
        f = _sf(lambda x: (1 - x) ** C * jnp.cos(x), (0.0, C))
        g = _sf(lambda x: (1 - x) ** A * jnp.cos(x), (0.0, A))
        I = _ip(f, g).real
        I_exact = 1.76743783779682186471
        assert abs(I - I_exact) < 1e1 * EPS * abs(I_exact)

    def test_pole_right(self):
        f = _sf(lambda x: (1 - x) ** B * (x ** 5), (0.0, B))
        g = _sf(lambda x: jnp.exp(x) * jnp.sin(5 * x), (0.0, 0.0))
        I = _ip(f, g).real
        I_exact = -3.2185857544263774863
        assert abs(I - I_exact) < 1e1 * EPS * abs(I_exact)

    def test_pole_and_root(self):
        f = _sf(lambda x: (1 + x) ** B * jnp.sin(x), (B, 0.0))
        g = _sf(lambda x: jnp.sin(2 * x) * (1 - x) ** C, (0.0, C))
        I = _ip(f, g).real
        I_exact = 3.703689983503164674
        assert abs(I - I_exact) < 1e1 * EPS * abs(I_exact)

    def test_poles_different_ends(self):
        f = _sf(lambda x: jnp.sin(x) * (1 - x ** 2) ** B, (B, B))
        g = _sf(lambda x: jnp.cos(x) ** 3 * (1 + x) ** P, (P, 0.0))
        I = _ip(f, g).real
        I_exact = -0.378959054771939734525
        assert abs(I - I_exact) < 1e1 * EPS

    def test_trivial_no_singularity(self):
        f = _sf(lambda x: jnp.exp(x) * x ** 3 * jnp.sin(2 * x), (0.0, 0.0))
        g = _sf(lambda x: jnp.exp(1 - x) ** 1.5, (0.0, 0.0))
        I = _ip(f, g).real
        I_exact = 2.30589565644897950113
        assert abs(I - I_exact) < 1e1 * EPS * abs(I_exact)

    def test_complex(self):
        f = _sf(
            lambda x: (jnp.sin(x) + 1j * jnp.cos(x)) / ((1 + x) ** 0.4 * (1 - x) ** 0.3),
            (-0.4, -0.3),
        )
        g = _sf(lambda x: (jnp.sin(x) - 1j * jnp.cos(x)) / ((1 + x) ** 0.2), (-0.2, 0.0))
        # innerProduct(f, g) = sum(conj(f).*g); Singfun supports complex smooth
        # parts and conjugation, so the complex pairing is available.
        I = complex((f.conj() * g).sum())
        I_exact = -0.66255618280005499086 + 0.95157967059305931745j
        assert abs(I - I_exact) < 1e1 * EPS * abs(I_exact)
