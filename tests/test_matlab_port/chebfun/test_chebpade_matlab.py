"""All six native Chebyshev-Pade predicates, using public Chebfun norms.

MATLAB source: tests/chebfun/test_chebpade.m; @chebfun/chebpade.m.
Chebfun commit: 7574c77.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = np.finfo(float).eps
DOM = (-1.0, 3.0)
XS = jnp.asarray(np.linspace(-0.99, 2.99, 80))


class TestChebfunChebpade:
    @pytest.fixture(autouse=True)
    def _native_warning_state(self):
        # Native warning('off'), with the previous state restored after tests.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield

    def _setup(self):
        P = cj.chebfun(
            jnp.asarray([0.5045, -1.3813, 2.1122, 0.0558, -0.6817]), domain=DOM, coeffs=True
        )
        Q = cj.chebfun(jnp.asarray([1, 0.1155, -0.8573, -0.2679, 0.5246]), domain=DOM, coeffs=True)
        R = P / Q
        return P, Q, R

    def test_maehly_no_reduction(self):
        P, Q, R = self._setup()
        p, q, _ = cj.chebpade(R, 4, 4, "maehly")
        err = float((P - p).norm()) + float((Q - q).norm())
        bound = 100 * R.vscale * EPS
        print({"native_clause": 1, "continuous_L2_sum": err, "source_bound": bound}, flush=True)
        assert err < bound
        # Retain prior sampled checks as supplements, at their original bounds.
        sampled_error = float(jnp.max(jnp.abs(p(XS) - P(XS)))) + float(
            jnp.max(jnp.abs(q(XS) - Q(XS)))
        )
        assert sampled_error < 100 * float(jnp.max(jnp.abs(R(XS)))) * EPS

    def test_maehly_degree_reduction(self):
        P, Q, R = self._setup()
        p, q, _ = cj.chebpade(R, 6, 5, "maehly")
        err = float((P - p).norm()) + float((Q - q).norm())
        bound = 100 * R.vscale * EPS
        print({"native_clause": 2, "continuous_L2_sum": err, "source_bound": bound}, flush=True)
        assert err < bound
        sampled_error = float(jnp.max(jnp.abs(p(XS) - P(XS)))) + float(
            jnp.max(jnp.abs(q(XS) - Q(XS)))
        )
        assert sampled_error < 100 * float(jnp.max(jnp.abs(R(XS)))) * EPS

    def test_geddes(self):
        a = jnp.asarray(
            [
                -464 / 6375,
                -742 / 6375,
                349 / 12750,
                512 / 6375,
                13 / 3400,
                2129 / 51000,
                1333 / 8160,
                9703 / 34000,
            ]
        )
        b = jnp.asarray([-32 / 85, -28 / 85, 1.0])
        f = cj.chebfun(lambda x: jnp.polyval(a, x) / jnp.polyval(b, x))
        p, q, _ = cj.chebpade(f, 7, 2)
        err = float((f - p / q).norm(jnp.inf))
        print({"native_clause": 3, "continuous_supremum": err, "source_bound": 2e-15}, flush=True)
        assert err < 2e-15
        cp = np.asarray(p.chebcoeffs(len(p)))
        assert abs(cp[0] - 17 / 46) < 1e-13
        # Prior101-site check retained at the original bound as supplemental.
        xs = jnp.asarray(np.linspace(-1, 1, 101))
        assert float(jnp.max(jnp.abs(f(xs) - p(xs) / q(xs)))) < 2e-15

    def test_complex_scaling(self):
        f = cj.chebfun(jnp.exp)
        p1, _, r1 = cj.chebpade(f, 2, 3)
        p2, _, r2 = cj.chebpade(1j * f, 2, 3)
        ratios = np.asarray(
            [
                complex(np.asarray(p2(0.3))) / complex(np.asarray(p1(0.3))),
                complex(np.asarray(r2(-0.4))) / complex(np.asarray(r1(-0.4))),
            ]
        )
        assert np.linalg.norm(ratios - np.asarray([1j, 1j]), np.inf) < 1e-13

    def test_explicit_num_coeffs(self):
        f = cj.chebfun(jnp.exp)
        p, q, r = cj.chebpade(f, 2, 2, 10)
        err = float((f - p / q).norm())
        print({"native_clause": 6, "continuous_L2": err, "source_bound": 2e-4}, flush=True)
        assert err < 2e-4
        # Prior single-site1e-3 probe retained only as supplemental.
        assert abs(float(np.real(np.asarray(r(0.5)))) - np.exp(0.5)) < 1e-3


@pytest.mark.parametrize("m,n,next_degrees", [(2, 1, (1, 1)), (1, 2, (1, 1))])
def test_maehly_source_warning_after_recursion(m, n, next_degrees, monkeypatch):
    import chebfunjax.utils.cfpade as module

    events = []
    sentinel = object()
    f = cj.chebfun(jnp.zeros(m + 2 * n + 1), coeffs=True)
    monkeypatch.setattr(module, "_coeffs", lambda _: np.zeros(m + 2 * n + 1))

    def recursive(_, mm, nn, kind):
        assert (mm, nn) == next_degrees and kind == "maehly"
        events.append("recursive")
        return sentinel

    monkeypatch.setattr(module, "chebpade", recursive)
    monkeypatch.setattr(module.warnings, "warn", lambda *a, **k: events.append("warning"))
    assert module._chebpade_maehly(f, m, n) is sentinel
    assert events == ["recursive", "warning"]


def test_clenshaw_complex_polynomial_zero_denominator_degree():
    # Diagnostic: native beta=1 branch, complex coefficients, no padding.
    coefficients = jnp.asarray([1 + 2j, 3 - 1j, -2 + 0.5j])
    f = cj.chebfun(coefficients, coeffs=True)
    p, q, _ = cj.chebpade(f, 2, 0)
    np.testing.assert_array_equal(np.asarray(p.chebcoeffs(3)), np.asarray(coefficients))
    np.testing.assert_array_equal(np.asarray(q.chebcoeffs(1)), [1])


@pytest.mark.parametrize("coefficients", [[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
def test_exact_singular_hankel_preserves_exception_adapter(coefficients):
    # m=n=1 => Hankel=[c1]=[0]; four supplied coefficients avoid RNG.
    f = cj.chebfun(jnp.asarray(coefficients), coeffs=True)
    with pytest.raises(np.linalg.LinAlgError, match="Singular matrix"):
        cj.chebpade(f, 1, 1)


def test_complex_explicit_M_independent_cosine_oracle():
    coefficients = np.asarray(
        [1 + 2j, 3 - 1j, -2 + 0.5j, 0.25j, 2 - 3j, -1j, 0.7 + 0.3j, -0.2 + 0.9j]
    )
    f = cj.chebfun(jnp.asarray(coefficients), coeffs=True)
    M = 3
    theta = np.arange(M + 1) * np.pi / M
    values = np.polynomial.chebyshev.chebval(np.cos(theta), coefficients)
    # Independent dense cosine interpolation; no package resample or FFT oracle.
    oracle = np.linalg.solve(np.cos(theta[:, None] * np.arange(M + 1)[None, :]), values)
    p, q, _ = cj.chebpade(f, M, 0, M)
    np.testing.assert_allclose(np.asarray(p.chebcoeffs(M + 1)), oracle, rtol=0, atol=3e-15)
    np.testing.assert_array_equal(np.asarray(q.chebcoeffs(1)), [1])


@pytest.mark.parametrize("coefficients", [[1.0, 2.0, 3.0], [1 + 2j, 2 - 1j, 3 + 0.5j]])
def test_explicit_M_zero_native_midpoint(coefficients):
    # Native chebfun(...,1) samples the midpoint; T2(0)=-1.
    f = cj.chebfun(jnp.asarray(coefficients), domain=(-1.0, 3.0), coeffs=True)
    p, q, r = cj.chebpade(f, 0, 0, 0)
    expected = coefficients[0] - coefficients[2]
    np.testing.assert_array_equal(np.asarray(p.chebcoeffs(1)), [expected])
    np.testing.assert_array_equal(np.asarray(q.chebcoeffs(1)), [1])
    np.testing.assert_array_equal(
        np.asarray(r(jnp.asarray([-1.0, 1.0, 3.0]))), np.repeat(expected, 3)
    )
