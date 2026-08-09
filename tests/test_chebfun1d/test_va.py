"""Tests for Vandermonde-with-Arnoldi (va_orthog / va_eval)."""
import numpy as np
import numpy.testing as npt

from chebfunjax.utils.va import va_eval, va_orthog


class TestVA:
    def test_orthonormal_columns(self):
        rng = np.random.default_rng(0)
        z = rng.uniform(-1, 1, 300)
        H, Q = va_orthog(z, 20)
        G = Q.T @ Q / z.shape[0]
        npt.assert_allclose(G, np.eye(21), atol=1e-12)

    def test_eval_reproduces_basis(self):
        rng = np.random.default_rng(1)
        z = rng.uniform(-1, 1, 200)
        H, Q = va_orthog(z, 15)
        W = va_eval(z, H)
        npt.assert_allclose(W, Q, atol=1e-12)

    def test_least_squares_fit(self):
        # fit a degree-8 polynomial exactly from noisy-free samples,
        # evaluated at NEW points
        rng = np.random.default_rng(2)
        z = rng.uniform(0, 20, 400)

        def f(x):
            return 0.3 * x**8 - x**3 + 5 * x - 2

        H, Q = va_orthog(z, 8)
        c = np.linalg.lstsq(Q, f(z), rcond=None)[0]
        znew = np.linspace(0, 20, 57)
        y = va_eval(znew, H) @ c
        # scale-relative: |f| reaches ~7.7e9 on this interval
        npt.assert_allclose(y, f(znew),
                            atol=1e-12 * np.max(np.abs(f(znew))))

    def test_complex_points(self):
        th = np.linspace(0, 2 * np.pi, 128, endpoint=False)
        z = np.exp(1j * th)
        H, Q = va_orthog(z, 10)
        G = Q.conj().T @ Q / z.shape[0]
        npt.assert_allclose(G, np.eye(11), atol=1e-12)
        W = va_eval(z, H)
        npt.assert_allclose(W, Q, atol=1e-12)
