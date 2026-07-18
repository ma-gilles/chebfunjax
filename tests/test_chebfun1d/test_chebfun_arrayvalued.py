"""Array-valued (multi-column) Chebfun-level operations (Fable 5,
Big-Three item 1): constructor, evaluation, calculus, per-column
roots and minandmax, including the piecewise case."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

XS = jnp.asarray(np.linspace(-0.9, 0.9, 9))
XN = np.asarray(XS)


def _mk():
    return cj.chebfun(
        lambda x: jnp.stack([jnp.sin(jnp.pi * x), x ** 2 - 0.25],
                            axis=-1))


class TestChebfunArrayValued:
    def test_construct_and_eval(self):
        f = _mk()
        exact = np.column_stack([np.sin(np.pi * XN), XN ** 2 - 0.25])
        np.testing.assert_allclose(np.asarray(f(XS)), exact, atol=1e-14)

    def test_calculus(self):
        f = _mk()
        np.testing.assert_allclose(
            np.asarray(f.sum()), [0.0, 2.0 / 3.0 - 0.5], atol=1e-14)
        d = np.asarray(f.diff()(XS))
        dex = np.column_stack([np.pi * np.cos(np.pi * XN), 2 * XN])
        np.testing.assert_allclose(d, dex, atol=1e-12)
        c = np.asarray(f.cumsum()(XS))
        cex = np.column_stack(
            [(-np.cos(np.pi * XN) + np.cos(-np.pi)) / np.pi,
             (XN ** 3 + 1.0) / 3.0 - 0.25 * (XN + 1.0)])
        np.testing.assert_allclose(c, cex, atol=1e-14)

    def test_roots_nan_padded(self):
        r = np.asarray(_mk().roots())
        assert r.ndim == 2 and r.shape[1] == 2
        np.testing.assert_allclose(r[:3, 0], [-1.0, 0.0, 1.0],
                                   atol=1e-12)
        np.testing.assert_allclose(r[:2, 1], [-0.5, 0.5], atol=1e-12)
        assert np.isnan(r[2, 1])

    def test_minandmax_per_column(self):
        (mnx, mnv), (mxx, mxv) = _mk().minandmax()
        np.testing.assert_allclose(np.asarray(mnv), [-1.0, -0.25],
                                   atol=1e-12)
        np.testing.assert_allclose(np.asarray(mxv), [1.0, 0.75],
                                   atol=1e-12)
        np.testing.assert_allclose(np.asarray(mnx), [-0.5, 0.0],
                                   atol=1e-7)
        np.testing.assert_allclose(np.asarray(mxx), [0.5, 1.0],
                                   atol=1e-7)

    def test_piecewise_array_roots(self):
        h = cj.chebfun(
            lambda x: jnp.stack(
                [jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x)], axis=-1),
            domain=(-1.0, 0.0, 1.0))
        r = np.asarray(h.roots())
        np.testing.assert_allclose(r[:3, 0], [-1.0, 0.0, 1.0],
                                   atol=1e-12)
        np.testing.assert_allclose(r[:2, 1], [-0.5, 0.5], atol=1e-12)


def _mk3():
    return cj.chebfun(
        lambda x: jnp.stack(
            [jnp.sin(jnp.pi * x), x ** 2 - 0.25, jnp.exp(x)], axis=-1))


EX3 = np.column_stack([np.sin(np.pi * XN), XN ** 2 - 0.25, np.exp(XN)])


class TestColumnManipulation:
    """Chebfun-level column ops added with the array-valued epic."""

    def test_n_columns_extract(self):
        f = _mk3()
        assert f.n_columns == 3
        g = f.extract_columns([0, 2])
        np.testing.assert_allclose(np.asarray(g(XS)), EX3[:, [0, 2]],
                                   atol=1e-14)
        s = f.extract_columns(1)
        assert s.n_columns == 1
        np.testing.assert_allclose(np.asarray(s(XS)), EX3[:, 1],
                                   atol=1e-14)

    def test_assign_mat2cell_repmat(self):
        f = _mk3()
        h = cj.chebfun(lambda x: jnp.stack([x, x ** 3], axis=-1))
        asg = f.assign_columns([0, 2], h)
        exa = np.column_stack([XN, EX3[:, 1], XN ** 3])
        np.testing.assert_allclose(np.asarray(asg(XS)), exa, atol=1e-14)
        assert f.assign_columns(1, None).n_columns == 2
        parts = f.mat2cell([1, 2])
        assert [p.n_columns for p in parts] == [1, 2]
        rep = parts[0].repmat(3)
        np.testing.assert_allclose(
            np.asarray(rep(XS)),
            np.tile(EX3[:, :1], (1, 3)), atol=1e-14)

    def test_fliplr_any_all_pow0(self):
        f = _mk3()
        np.testing.assert_allclose(np.asarray(f.fliplr()(XS)),
                                   EX3[:, ::-1], atol=1e-14)
        assert bool(np.all(np.asarray(f.any())))
        np.testing.assert_array_equal(np.asarray(f.all()),
                                      [False, False, True])
        p0 = f ** 0
        assert p0.n_columns == 3
        np.testing.assert_allclose(np.asarray(p0(XS)), 1.0, atol=0)

    def test_abs_array_valued(self):
        f = _mk3()
        np.testing.assert_allclose(np.asarray(abs(f)(XS)),
                                   np.abs(EX3), atol=1e-13)

    def test_piecewise_eval_and_cumsum(self):
        pc = cj.chebfun(
            lambda x: jnp.stack(
                [jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x)], axis=-1),
            domain=(-1.0, 0.3, 1.0))
        exact = np.column_stack([np.sin(np.pi * XN),
                                 np.cos(np.pi * XN)])
        np.testing.assert_allclose(np.asarray(pc(XS)), exact,
                                   atol=1e-14)
        C = pc.cumsum()
        cex = np.column_stack(
            [(-np.cos(np.pi * XN) + np.cos(-np.pi)) / np.pi,
             (np.sin(np.pi * XN) - np.sin(-np.pi)) / np.pi])
        np.testing.assert_allclose(np.asarray(C(XS)), cex, atol=1e-14)
