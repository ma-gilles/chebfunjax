"""Vector selection and source polynomial argument contracts."""

import jax.numpy as jnp
import pytest

import chebfunjax as cj


@pytest.mark.parametrize("fn", [cj.hermpoly, cj.lagpoly])
@pytest.mark.parametrize("degree", [-1, 1.5])
def test_invalid_degree(fn, degree):
    with pytest.raises(IndexError):
        fn(degree)


@pytest.mark.parametrize("fn", [cj.hermpoly, cj.lagpoly])
def test_vector_degree_order(fn):
    x = jnp.array([-0.2, 0.4]) if fn is cj.hermpoly else jnp.array([0.2, 0.4])
    f = fn([2, 0, 2])
    v = f(x)
    want = 4 * x * x - 2 if fn is cj.hermpoly else (x * x - 4 * x + 2) / 2
    assert v.shape == (2, 3)
    assert jnp.max(jnp.abs(v[:, 0] - want)) < 1e-12
    assert jnp.array_equal(v[:, 0], v[:, 2])
    assert jnp.all(v[:, 1] == 1)


@pytest.mark.parametrize("alpha", [[0.0, 1.0], 1j, 0j])
def test_alpha_error(alpha):
    with pytest.raises(ValueError, match="real-valued scalar"):
        cj.lagpoly(2, alpha)


def test_kind_matches_exact_prob_name():
    x = jnp.array([-0.3, 0.4])
    assert jnp.max(jnp.abs(cj.hermpoly(2, "physicists")(x) - (4 * x * x - 2))) < 1e-12
    assert jnp.max(jnp.abs(cj.hermpoly(2, "PrOb")(x) - (x * x - 1))) < 1e-12


@pytest.mark.parametrize("fn", [cj.hermpoly, cj.lagpoly])
def test_explicit_finite_extension(fn):
    f = fn(2, domain=(-2.0, 3.0))
    assert f.domain.a == -2.0 and f.domain.b == 3.0
    x = jnp.array([-0.3, 0.4])
    expected = 4 * x * x - 2 if fn is cj.hermpoly else (x * x - 4 * x + 2) / 2
    assert jnp.max(jnp.abs(f(x) - expected)) < 1e-12


def test_probabilists_degree6_additional_assertion():
    # Native clause14 accidentally reuses degree5 err. Preserve that clause
    # elsewhere and additionally test the degree6 computation it evaluates.
    x = jnp.linspace(-1, 1, 100)
    expected = x**6 - 15 * x**4 + 45 * x * x - 15
    assert jnp.linalg.norm(cj.hermpoly(6, "prob")(x) - expected) < 1e7 * jnp.finfo(float).eps


def test_hermite_domain():
    f = cj.hermpoly(0)
    assert f.domain.a == float("-inf") and f.domain.b == float("inf")


def test_laguerre_domain():
    f = cj.lagpoly(0)
    assert f.domain.a == 0 and f.domain.b == float("inf")
