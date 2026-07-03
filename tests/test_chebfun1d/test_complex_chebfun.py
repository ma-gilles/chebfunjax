"""Complex-valued Chebfun support.

MATLAB Chebfun handles complex-valued functions natively (Guide ch.5);
these tests pin the core semantics: construction keeps the imaginary
part, arithmetic/calculus flow in complex128, the inner product is
conjugate-linear in its first argument, and real/imag/conj/angle behave
like their MATLAB counterparts. Real chebfuns must be unaffected
(float64 storage preserved).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

jax.config.update("jax_enable_x64", True)

import chebfunjax as cj  # noqa: E402

PI = float(np.pi)


def _cx(x):
    return complex(np.asarray(x))


class TestComplexConstruction:
    def test_no_complex_warning_and_exact_eval(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            g = cj.chebfun(lambda s: jnp.exp(1j * s), domain=[0.0, PI])
        npt.assert_allclose(_cx(g(jnp.array(1.0))), np.exp(1j), rtol=1e-14)

    def test_coeffs_dtype_complex(self):
        g = cj.chebfun(lambda s: jnp.exp(1j * s), domain=[0.0, PI])
        assert jnp.iscomplexobj(g.funs[0].tech.coeffs)
        assert not g.isreal()

    def test_real_construction_stays_float64(self):
        f = cj.chebfun(jnp.sin)
        assert f.funs[0].tech.coeffs.dtype == jnp.float64
        assert f.isreal()


class TestComplexArithmetic:
    def setup_method(self):
        self.g = cj.chebfun(lambda s: jnp.exp(1j * s), domain=[0.0, PI])
        self.x = jnp.array(1.0)

    def test_product(self):
        npt.assert_allclose(_cx((self.g * self.g)(self.x)), np.exp(2j),
                            rtol=1e-13)

    def test_complex_scalar_ops(self):
        npt.assert_allclose(_cx((1j * self.g)(self.x)), 1j * np.exp(1j),
                            rtol=1e-13)
        npt.assert_allclose(_cx((self.g + 1j)(self.x)), np.exp(1j) + 1j,
                            rtol=1e-13)
        npt.assert_allclose(_cx((self.g / 2j)(self.x)), np.exp(1j) / 2j,
                            rtol=1e-13)

    def test_real_plus_complex_scalar_promotes(self):
        f = cj.chebfun(jnp.sin)
        h = f + 1j
        npt.assert_allclose(_cx(h(jnp.array(0.5))), np.sin(0.5) + 1j,
                            rtol=1e-13)


class TestComplexCalculus:
    def setup_method(self):
        self.g = cj.chebfun(lambda s: jnp.exp(1j * s), domain=[0.0, PI])

    def test_diff(self):
        d = self.g.diff()
        npt.assert_allclose(_cx(d(jnp.array(1.0))), 1j * np.exp(1j),
                            rtol=1e-12)

    def test_cumsum(self):
        F = self.g.cumsum()
        exact = (np.exp(1j) - 1.0) / 1j
        npt.assert_allclose(_cx(F(jnp.array(1.0))), exact, rtol=1e-12)

    def test_sum(self):
        exact = (np.exp(1j * PI) - 1.0) / 1j
        npt.assert_allclose(_cx(self.g.sum()), exact, atol=1e-13)

    def test_norm_is_sesquilinear(self):
        # ||exp(is)||_2^2 = integral of |exp(is)|^2 = pi
        npt.assert_allclose(float(np.real(np.asarray(self.g.norm(2)))),
                            np.sqrt(PI), rtol=1e-12)

    def test_inner_conjugates_first_argument(self):
        # <g, g> = integral conj(g) g = pi (real, positive)
        val = _cx(self.g.inner(self.g))
        npt.assert_allclose(val, PI, atol=1e-12)


class TestRealImagConjAngle:
    def setup_method(self):
        self.g = cj.chebfun(lambda s: jnp.exp(1j * s), domain=[0.0, PI])
        self.x = jnp.array(1.0)

    def test_real(self):
        r = self.g.real()
        assert r.isreal()
        npt.assert_allclose(float(r(self.x)), np.cos(1.0), rtol=1e-13)

    def test_imag(self):
        m = self.g.imag()
        assert m.isreal()
        npt.assert_allclose(float(m(self.x)), np.sin(1.0), rtol=1e-13)

    def test_conj(self):
        npt.assert_allclose(_cx(self.g.conj()(self.x)), np.exp(-1j),
                            rtol=1e-13)

    def test_angle(self):
        npt.assert_allclose(float(self.g.angle()(self.x)), 1.0, atol=1e-12)

    def test_abs_of_complex(self):
        a = abs(self.g)
        npt.assert_allclose(float(a(self.x)), 1.0, rtol=1e-12)


class TestComplexPlot:
    def test_plot_draws_complex_plane_curve(self):
        import matplotlib

        matplotlib.use("Agg")
        g = cj.chebfun(lambda s: jnp.exp(1j * s), domain=[0.0, 2.0 * PI])
        fig, ax = g.plot()
        assert ax.get_aspect() == 1.0
        line = ax.get_lines()[0]
        xdata, ydata = line.get_xdata(), line.get_ydata()
        # points must lie on the unit circle
        npt.assert_allclose(np.hypot(xdata, ydata), 1.0, atol=1e-10)
        import matplotlib.pyplot as plt

        plt.close(fig)
