"""Port of MATLAB Chebfun tests/chebfun/test_any.m (Fable 5).

``any(f)`` is True where the function is nonzero somewhere on its domain.
For array-valued chebfuns it returns a per-column ``(m,)`` boolean array
(``any`` down the continuous dimension).

Includes the source singular and unbounded clauses. Probe points come from a fresh pinned MATLAB seed6178 capture.

Provenance
----------
MATLAB source : tests/chebfun/test_any.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

_REFERENCE = json.loads((Path(__file__).parent / "fixtures/logical_source_matlab.json").read_text())
XR = jnp.asarray(_REFERENCE["x"])
YR = jnp.asarray(_REFERENCE["y"])


class TestChebfunAny:
    def test_empty(self):
        # pass(1): ~any(chebfun()).
        from chebfunjax.chebfun1d.chebfun import chebfun

        assert not chebfun().any()

    def test_columns(self):
        # pass(2): any([sin(x) 0*x exp(x)]) == [1 0 1].
        # FIXED (Fable 5, Big-Three array-valued epic): per-column any() -> (m,).
        f = cj.chebfun(
            lambda x: jnp.stack([jnp.sin(x), 0 * x, jnp.exp(x)], axis=-1),
            domain=(-1, -0.5, 0, 0.5, 1),
            splitting=True,
        )
        assert list(np.asarray(f.any()).astype(int)) == [1, 0, 1]

    def test_columns_with_complex(self):
        # pass(5): any([0*x hvsde(x) exp(2 pi i x)]) == [0 1 1].
        # FIXED (Fable 5, Big-Three array-valued epic).
        hvsde = lambda x: 0.5 * (jnp.sign(x) + 1)
        f = cj.chebfun(
            lambda x: jnp.stack([0 * x, hvsde(x), jnp.exp(2 * np.pi * 1j * x)], axis=-1),
            domain=(-1, 0, 1),
            splitting=True,
        )
        assert list(np.asarray(f.any()).astype(int)) == [0, 1, 1]

    # Sub-second in isolation; the file's shared-fixture JAX warmup plus
    # box contention has tripped 180-300 s per-test timeouts (same
    # pattern as chebfun2's narrow-ridge test) — give it headroom.
    @pytest.mark.timeout(880)
    def test_transpose_rows(self):
        # pass(6): any(f.', 2) on a row chebfun gives the per-component
        # booleans (dim maps 2 -> 1 on the column form); pass(8):
        # any(f.', 1) gives a transposed 0/1 chebfun.
        f = cj.chebfun(
            lambda x: jnp.stack([jnp.sin(x), 0 * x, jnp.exp(x)], axis=-1),
            domain=(-1, -0.5, 0, 0.5, 1),
            splitting=True,
        )
        assert list(np.asarray(f.T.any(2)).astype(int)) == [1, 0, 1]
        g = f.T.any(1)
        assert g.is_transposed
        assert len(g.funs) == 1
        xs = XR
        assert bool(np.all(np.asarray(g(xs)) == 1.0))

        hvsde = lambda x: 0.5 * (jnp.sign(x) + 1)
        g = cj.chebfun(
            lambda x: jnp.stack([0 * x, hvsde(x), jnp.exp(2 * np.pi * 1j * x)], axis=-1),
            domain=(-1, 0, 1),
            splitting=True,
        )
        # dim=2 on the row form maps to the per-component booleans
        # (MATLAB pass(6); dim=1 would build the 0/1 chebfun instead).
        assert list(np.asarray(g.T.any(2)).astype(int)) == [0, 1, 1]

    def test_pointvalues_nan(self):
        # pass(4): any(f, 1) still returns [1 0 1] after setting a pointValues
        # entry to NaN -- any() reduces over the continuous behaviour of each
        # column and ignores an isolated NaN carried in the pointValues
        # metadata (MATLAB semantics).
        f = cj.chebfun(
            lambda x: jnp.stack([jnp.sin(x), 0 * x, jnp.exp(x)], axis=-1),
            domain=(-1, -0.5, 0, 0.5, 1),
            splitting=True,
        )
        pv = np.array(f.point_values, dtype=float, copy=True)
        pv[2, 1] = np.nan
        f = f.set_point_values(jnp.asarray(pv))
        assert list(np.asarray(f.any()).astype(int)) == [1, 0, 1]

    def test_discrete_dimension(self):
        # pass(7): any(f, 2) of [sin, 0, exp] is the constant 1 chebfun.
        f = cj.chebfun(lambda x: jnp.stack([jnp.sin(x), 0 * x, jnp.exp(x)], axis=-1))
        g = f.any(2)
        xs = XR
        assert not g.is_transposed
        assert len(g.funs) == 1
        assert bool(np.all(np.asarray(g(xs)) == 1.0))

        # pass(9): [sin, 0] gets a breakpoint at sin's root with an
        # isolated 0 pointValue.
        f2 = cj.chebfun(lambda x: jnp.stack([jnp.sin(x), 0 * x], axis=-1))
        g2 = f2.any(2)
        assert not g2.is_transposed
        bps = [float(t) for t in g2.domain.breakpoints]
        assert len(bps) == 3 and abs(bps[1]) < 10 * g2.vscale * jnp.finfo(jnp.float64).eps
        assert [int(v) for v in np.asarray(g2.point_values)] == [1, 0, 1]
        assert bool(np.all(np.asarray(g2(xs)) == 1.0))

    def test_dim_error(self):
        # pass(12): any(f, 3) raises CHEBFUN:CHEBFUN:any:dim.
        f = cj.chebfun(lambda x: jnp.sin(x))
        with pytest.raises(ValueError):
            f.any(3)

    def test_unbounded_zero(self):
        # pass(16): ~any(0*x) on the unbounded domain [1, inf).  A zero
        # function on a semi-infinite interval has no nonzero value anywhere.
        g = cj.chebfun(lambda x: 0 * x, domain=(1, jnp.inf))
        assert not bool(g.any())

    def test_singular_and_unbounded_blowup(self):
        # pass(13,14): singular (SingFun) cases; pass(15): any(1/x^2) on
        # [1, inf) with 'exps' [0 -2] -- unbounded-domain blow-up.
        f = cj.chebfun(
            lambda x: jnp.sin(30 * x) / ((x + 2) * (x - 7)),
            domain=(-2, 7),
            exps=(-1, -1),
            splitting=True,
        )
        assert bool(f.any(1))
        g = f.any(2)
        x = YR
        assert bool(jnp.all(g(x) == 1))
        f = cj.chebfun(lambda x: 1 / x**2, domain=(1, jnp.inf), exps=(0, -2))
        assert bool(f.any())


def test_original_empty_dim2_and_transposed_zero_clause():
    assert cj.chebfun().any(2).isempty()
    f = cj.chebfun(lambda x: jnp.stack([jnp.sin(x), 0 * x], axis=-1))
    g = f.T.any(1)
    assert g.is_transposed
    assert len(g.domain.breakpoints) == 3
    assert abs(float(g.domain.breakpoints[1])) < 10 * g.vscale * jnp.finfo(jnp.float64).eps
    assert jnp.array_equal(g.point_values, jnp.asarray([1.0, 0.0, 1.0]))
    x = XR
    assert bool(jnp.all(g(x) == 1))


def test_original_discontinuous_clause11():
    step = lambda x: 0.5 * (jnp.sign(x) + 1)
    f = cj.chebfun(
        lambda x: jnp.stack([step(x), jnp.sin(x) * step(x)], axis=-1),
        domain=(-1, 0, 1),
        splitting=True,
    )
    g = f.any(2)
    assert not g.is_transposed
    assert jnp.array_equal(g.point_values, jnp.asarray([0.0, 1.0, 1.0]))
    x = XR
    expected = (step(x) != 0) | (jnp.sin(x) * step(x) != 0)
    assert bool(jnp.all(g(x) == expected))
