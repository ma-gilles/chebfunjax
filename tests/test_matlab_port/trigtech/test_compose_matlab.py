"""Port of MATLAB Chebfun tests/trigtech/test_compose.m (Fable 5).

FIXED: Trigtech.compose added in the Fable 5 audit; array-valued unary
composition (pass 2, 3) now works via (n, m) Fourier coefficients.

chebfunjax ``Trigtech.compose`` is UNARY only (``compose(op)`` -> op(f)); it has
no binary (pass 5, 6) or trigtech-of-trigtech (pass 7, 8, 9) form, so those and
the two error-identifier cases (pass 10, 11) stay skipped with precise reasons.

Provenance
----------
MATLAB source : tests/trigtech/test_compose.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)
XS = jnp.asarray(np.linspace(-0.97, 0.97, 60))


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestTrigtechCompose:
    def test_sin_of_cos(self):
        f = Trigtech.from_function(
            lambda x: jnp.pi * jnp.cos(jnp.pi * (x - 0.1)))
        g = f.compose(jnp.sin)
        h = Trigtech.from_function(
            lambda x: jnp.sin(jnp.pi * jnp.cos(jnp.pi * (x - 0.1))))
        err = jnp.abs(g(XS) - h(XS))
        assert float(jnp.max(err)) < 100 * h.vscale * EPS

    def test_array_valued(self):
        # pass(2): compose([pi*cos(pi x) pi*cos(2 pi x)], @sin), coeffs match
        # after prolonging to a common length.
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) coeffs.
        f = Trigtech.from_function(
            lambda x: jnp.stack(
                [jnp.pi * jnp.cos(jnp.pi * x), jnp.pi * jnp.cos(2 * jnp.pi * x)], axis=-1
            )
        )
        g = f.compose(jnp.sin)
        h = Trigtech.from_function(
            lambda x: jnp.stack(
                [
                    jnp.sin(jnp.pi * jnp.cos(jnp.pi * x)),
                    jnp.sin(jnp.pi * jnp.cos(2 * jnp.pi * x)),
                ],
                axis=-1,
            )
        )
        n = max(g.n, h.n)
        gp = g.prolong(n)
        hp = h.prolong(n)
        assert _ninf(hp.coeffs - gp.coeffs) < 10 * h.vscale * EPS
