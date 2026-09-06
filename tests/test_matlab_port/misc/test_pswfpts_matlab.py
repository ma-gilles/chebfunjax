"""Port of MATLAB Chebfun tests/misc/test_pswfpts.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_pswfpts.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.utils.pswf import pswf, pswfpts

jax.config.update("jax_enable_x64", True)


def _wP(w, P, x):
    # w * P(x, :): quadrature applied to every column of P
    return np.asarray(w).ravel() @ np.asarray(P(jnp.asarray(x)))


class TestMiscPswfpts:
    def test_all_matlab_assertions(self):
        P, _ = pswf(np.arange(10), 4, output="chebfun")
        x, w = pswfpts(9, 4)
        x, w = np.asarray(x), np.asarray(w)
        assert np.linalg.norm(x - np.sort(np.asarray(P.extract_columns(9).roots()))) < 1e-10  # pass(1)
        assert np.linalg.norm(np.asarray(P.sum()) - _wP(w, P, x)) < 1e-10  # pass(2)

        k = 51
        f = lambda t: np.cos(k * t)  # noqa: E731
        II = 2 * np.sin(k) / k
        x, w = pswfpts(k, k)
        assert abs(np.asarray(w).ravel() @ f(np.asarray(x)) - II) < 1e-10  # pass(3)

        x, w = pswfpts(5, 4, (-1.0, 1.0), "ggq")
        assert np.linalg.norm(np.asarray(P.sum()) - _wP(w, P, x)) < 1e-10  # pass(4)

        x, w = pswfpts(k, k, (-1.0, 1.0), "ggq")
        assert abs(np.asarray(w).ravel() @ f(np.asarray(x)) - II) < 1e-10  # pass(5)

        x, w = pswfpts(10, np.pi, (-7.0, 7.0))
        x, w = np.asarray(x), np.asarray(w).ravel()
        assert np.linalg.norm(w - w[::-1]) == 0 and np.linalg.norm(x + x[::-1]) == 0  # pass(6)
        x, w = pswfpts(15, np.sqrt(2), (-7.0, 7.0), "ggq")
        x, w = np.asarray(x), np.asarray(w).ravel()
        assert np.linalg.norm(w - w[::-1]) == 0 and np.linalg.norm(x + x[::-1]) == 0  # pass(7)

        x, w = pswfpts(21, 80, (-1.0, 1.0), "ggq")
        P, _ = pswf(40, 80, output="chebfun")
        assert abs(float(P.sum()) - float(np.asarray(w).ravel() @ np.asarray(P(jnp.asarray(x))))) < 1e-10  # pass(8)
