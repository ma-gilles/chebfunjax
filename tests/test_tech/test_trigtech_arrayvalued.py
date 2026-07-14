"""Array-valued (multi-column) trigtech foundation (Fable 5,
Big-Three item 1): column-wise transforms, Horner evaluation, and
coefficient ops on (n, m) Fourier coefficient matrices."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.trigtech import (
    Trigtech,
    trig_coeffs2vals,
    trig_vals2coeffs,
)

N = 32
XG = jnp.asarray(-1.0 + 2.0 * np.arange(N) / N)
V = jnp.stack([jnp.sin(jnp.pi * XG), jnp.cos(2 * jnp.pi * XG)], axis=-1)

XS = jnp.asarray(np.linspace(-0.95, 0.95, 9))
XN = np.asarray(XS)
EXACT = np.column_stack([np.sin(np.pi * XN), np.cos(2 * np.pi * XN)])


def _mk() -> Trigtech:
    return Trigtech.from_values(V)


class TestArrayValuedTrigFoundation:
    def test_column_consistent_transforms(self):
        C = trig_vals2coeffs(V)
        assert C.shape == (N, 2)
        for j in range(2):
            np.testing.assert_allclose(
                np.asarray(C[:, j]),
                np.asarray(trig_vals2coeffs(V[:, j])), atol=1e-15)
        np.testing.assert_allclose(
            np.asarray(trig_coeffs2vals(C)).real, np.asarray(V),
            atol=1e-14)

    def test_call_and_simplify(self):
        t = _mk()
        np.testing.assert_allclose(np.asarray(t(XS)), EXACT, atol=1e-14)
        s = t.simplify()
        assert s.coeffs.ndim == 2 and s.n < t.n
        np.testing.assert_allclose(np.asarray(s(XS)), EXACT, atol=1e-14)

    def test_prolong(self):
        p = _mk().prolong(65)
        assert p.coeffs.shape == (65, 2)
        np.testing.assert_allclose(np.asarray(p(XS)), EXACT, atol=1e-14)

    def test_diff_cumsum_sum(self):
        t = _mk()
        dex = np.column_stack([np.pi * np.cos(np.pi * XN),
                               -2 * np.pi * np.sin(2 * np.pi * XN)])
        np.testing.assert_allclose(np.asarray(t.diff()(XS)), dex,
                                   atol=1e-12)
        cex = np.column_stack(
            [(-np.cos(np.pi * XN) + np.cos(-np.pi)) / np.pi,
             (np.sin(2 * np.pi * XN) - np.sin(-2 * np.pi))
             / (2 * np.pi)])
        np.testing.assert_allclose(np.asarray(t.cumsum()(XS)), cex,
                                   atol=1e-14)
        np.testing.assert_allclose(np.asarray(t.sum()), [0.0, 0.0],
                                   atol=1e-14)

    def test_multiply(self):
        t = _mk()
        np.testing.assert_allclose(np.asarray((t * t)(XS)), EXACT ** 2,
                                   atol=1e-13)
        g = Trigtech.from_values(jnp.cos(jnp.pi * XG))
        np.testing.assert_allclose(
            np.asarray((t * g)(XS)),
            EXACT * np.cos(np.pi * XN)[:, None], atol=1e-13)

    def test_roots_nan_padded(self):
        r = np.asarray(_mk().roots())
        assert r.ndim == 2 and r.shape[1] == 2
        # sin(pi x) roots: -1, 0, 1; cos(2 pi x): +-0.25, +-0.75
        assert abs(r[1, 0]) < 1e-12
        np.testing.assert_allclose(
            r[:4, 1], [-0.75, -0.25, 0.25, 0.75], atol=1e-12)

    def test_adaptive_construction(self):
        f = Trigtech.from_function(
            lambda t: jnp.stack(
                [jnp.sin(jnp.pi * t), jnp.cos(2 * jnp.pi * t)],
                axis=-1))
        assert f.ishappy and f.coeffs.ndim == 2
        np.testing.assert_allclose(np.asarray(f(XS)), EXACT, atol=1e-14)
