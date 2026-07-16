"""Port of MATLAB Chebfun tests/trigtech/test_assignColumns.m (Opus 4.8).

assignColumns writes columns of an array-valued trigtech.  Array-valued
trigtechs are now supported (FIXED, Fable 5, Big-Three array-valued epic).
chebfunjax ``assign_columns`` uses 0-based column indices (MATLAB is 1-based),
and an empty replacement (``None``) deletes columns.

Deterministic linspace substitutes MATLAB's random x per repo convention.
chebfunjax ``vscale`` is a single scalar (global max), so MATLAB's per-column
``size(vscale(h)) == [1 2]`` shape assertion (pass 3) is replaced by the
equivalent column-count check on the coefficients.

Provenance
----------
MATLAB source : tests/trigtech/test_assignColumns.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)
X = jnp.asarray(np.linspace(-1.0, 1.0, 100, endpoint=False))


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _f3():
    return Trigtech.from_function(
        lambda x: jnp.stack(
            [jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x), jnp.exp(1j * jnp.pi * x)],
            axis=-1,
        )
    )


class TestTrigtechAssignColumns:
    def test_assign_two_columns(self):
        # pass(1): assignColumns(f, [1 3], g) with g = [exp(cos) exp(sin)].
        # FIXED (Fable 5, Big-Three array-valued epic); 0-based cols [0, 2].
        f = _f3()
        g = Trigtech.from_function(
            lambda x: jnp.stack(
                [jnp.exp(jnp.cos(jnp.pi * x)), jnp.exp(jnp.sin(jnp.pi * x))], axis=-1
            )
        )
        h = f.assign_columns([0, 2], g)

        def h_exact(x):
            return jnp.stack(
                [jnp.exp(jnp.cos(jnp.pi * x)), jnp.cos(jnp.pi * x), jnp.exp(jnp.sin(jnp.pi * x))],
                axis=-1,
            )

        assert h.ishappy
        assert _ninf(h(X) - h_exact(X)) < 3e2 * h.vscale * EPS

    def test_assign_unhappy(self):
        # pass(2): assigning a non-resolvable (non-periodic) column -> unhappy.
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _f3()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = Trigtech.from_function(lambda x: x)
            h = f.assign_columns(1, g)
        assert not h.ishappy

    def test_assign_empty_removes(self):
        # pass(3): assignColumns(f, 1, []) deletes column 1 -> [cos exp(1i)].
        # FIXED (Fable 5, Big-Three array-valued epic).  chebfunjax vscale is a
        # scalar, so MATLAB's size(vscale(h))==[1 2] is a 2-column check here.
        f = _f3()
        h = f.assign_columns(0, None)

        def h_exact(x):
            return jnp.stack([jnp.cos(jnp.pi * x), jnp.exp(1j * jnp.pi * x)], axis=-1)

        assert h.coeffs.shape[1] == 2
        assert _ninf(h(X) - h_exact(X)) < 3e2 * h.vscale * EPS

    def test_assign_scalar_column(self):
        # pass(4): assignColumns(sin, 1, cos) == cos.
        # FIXED (Fable 5, Big-Three array-valued epic).  Result carries a (n, 1)
        # coeff column that is bit-identical to the scalar cos representation.
        f = Trigtech.from_function(lambda x: jnp.sin(jnp.pi * x))
        g = Trigtech.from_function(lambda x: jnp.cos(jnp.pi * x))
        h = f.assign_columns(0, g)
        assert _ninf(jnp.ravel(h.coeffs) - g.coeffs) == 0.0
