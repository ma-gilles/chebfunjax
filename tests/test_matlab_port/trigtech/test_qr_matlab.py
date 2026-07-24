"""Port of MATLAB Chebfun tests/trigtech/test_qr.m (Opus 4.8[1m]).

qr(f) computes a QR factorization of an (array-valued) trigtech: F = Q R
with Q orthonormal in the continuous L^2 inner product on [-1, 1].

Provenance
----------
MATLAB source : tests/trigtech/test_qr.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)
# Deterministic test points in [-1, 1) (stands in for seedRNG(6178)).
X = jnp.asarray(np.linspace(-1.0, 1.0, 100, endpoint=False)
                + 0.0031415926, dtype=jnp.float64)


def _tt(f):
    return Trigtech.from_function(f)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _eye_err(ip, n):
    return _ninf(jnp.atleast_2d(jnp.asarray(ip)) - jnp.eye(n))


def _check_qr(f):
    n = f.num_columns
    Q, R = f.qr()
    ortho = _eye_err(Q.innerProduct(Q), n)
    err = (Q @ R) - f
    return ortho, _ninf(err(X)), f.vscale


def _check_qr_perm(f):
    n = f.num_columns
    Q, R, E = f.qr(want_e=True)
    ortho = _eye_err(Q.innerProduct(Q), n)
    err = (Q @ R) - (f @ E)
    return ortho, _ninf(err(X)), f.vscale


_F1 = lambda x: jnp.exp(jnp.sin(jnp.pi * x))  # noqa: E731
_F2 = lambda x: jnp.stack(  # noqa: E731
    [jnp.exp(jnp.sin(jnp.pi * x)), 3.0 / (4 - jnp.cos(jnp.pi * x))], axis=-1)
_F3 = lambda x: jnp.stack([  # noqa: E731
    jnp.ones_like(x), jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x),
    jnp.sin(2 * jnp.pi * x), jnp.cos(2 * jnp.pi * x), jnp.sin(3 * jnp.pi * x)],
    axis=-1)
_F4 = lambda x: jnp.stack([  # noqa: E731
    3.0 / (4 - jnp.exp(1j * jnp.pi * x)), jnp.exp(jnp.sin(jnp.pi * x)),
    jnp.cos(3 * jnp.pi * x)], axis=-1)


class TestTrigtechQr:
    # --- scalar-valued f = exp(sin(pi x)) -----------------------------
    def test_factorization_1(self):
        ortho, err, vs = _check_qr(_tt(_F1))
        assert ortho < 10 * vs * EPS

    def test_factorization_2(self):
        _, err, vs = _check_qr(_tt(_F1))
        assert err < 100 * vs * EPS

    def test_factorization_3(self):
        ortho, err, vs = _check_qr_perm(_tt(_F1))
        assert ortho < 10 * vs * EPS

    def test_factorization_4(self):
        _, err, vs = _check_qr_perm(_tt(_F1))
        assert err < 100 * vs * EPS

    # --- 2-column real f ----------------------------------------------
    def test_factorization_5(self):
        ortho, err, vs = _check_qr(_tt(_F2))
        assert ortho < 10 * vs * EPS

    def test_factorization_6(self):
        _, err, vs = _check_qr(_tt(_F2))
        assert err < 100 * vs * EPS

    def test_factorization_7(self):
        ortho, err, vs = _check_qr_perm(_tt(_F2))
        assert ortho < 10 * vs * EPS

    def test_factorization_8(self):
        _, err, vs = _check_qr_perm(_tt(_F2))
        assert err < 100 * vs * EPS

    # --- 6-column trigonometric basis ---------------------------------
    def test_factorization_9(self):
        ortho, err, vs = _check_qr(_tt(_F3))
        assert ortho < 10 * vs * EPS

    def test_factorization_10(self):
        _, err, vs = _check_qr(_tt(_F3))
        assert err < 100 * vs * EPS

    def test_factorization_11(self):
        ortho, err, vs = _check_qr_perm(_tt(_F3))
        assert ortho < 10 * vs * EPS

    def test_factorization_12(self):
        _, err, vs = _check_qr_perm(_tt(_F3))
        assert err < 100 * vs * EPS

    # --- 3-column complex f -------------------------------------------
    def test_factorization_13(self):
        ortho, err, vs = _check_qr(_tt(_F4))
        assert ortho < 10 * vs * EPS

    def test_factorization_14(self):
        _, err, vs = _check_qr(_tt(_F4))
        assert err < 100 * vs * EPS

    def test_factorization_15(self):
        ortho, err, vs = _check_qr_perm(_tt(_F4))
        assert ortho < 10 * vs * EPS

    def test_factorization_16(self):
        _, err, vs = _check_qr_perm(_tt(_F4))
        assert err < 100 * vs * EPS

    # --- 'vector' vs 'matrix' permutation flag ------------------------
    def test_factorization_17(self):
        f = _tt(_F4)
        n = f.num_columns
        _, _, e_mat = f.qr(mode="matrix", want_e=True)
        _, _, e_vec = f.qr(mode="vector", want_e=True)
        # e_mat[:, e_vec] should reproduce the identity (both are identity
        # here, as JAX has no column-pivoted QR).
        err = np.asarray(e_mat)[:, np.asarray(e_vec)] - np.eye(n)
        assert np.all(err == 0)

    # --- rank-deficient problem (MATLAB pass18/19; #1441 forces =1) ----
    def test_factorization_18(self):
        f = _tt(lambda x: jnp.stack(
            [jnp.cos(jnp.pi * x)] * 3, axis=-1))
        Q, R = f.qr()
        assert Q.num_columns == 3 and R.shape == (3, 3)

    def test_factorization_19(self):
        # Orthogonality is not guaranteed for the rank-deficient case
        # (MATLAB disables this check, forcing pass19 = 1).
        assert True

    # --- vscale of Q has one entry per column -------------------------
    def test_factorization_20(self):
        f = _tt(lambda x: jnp.stack(
            [jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x),
             jnp.cos(2 * jnp.pi * x)], axis=-1))
        Q, R = f.qr()
        assert Q.vscale_columns().shape == (3,)
