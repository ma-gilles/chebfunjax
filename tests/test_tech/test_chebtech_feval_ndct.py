"""Native large-feval dispatch and supplementary JAX contracts.

The original chebtech/test_feval.m uses at most1000query points, so these
are additional controls for its untested4000-by4000 source boundary.
"""

import importlib

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

BOTH = [Chebtech1, Chebtech2]


@pytest.mark.parametrize("Tech", BOTH)
@pytest.mark.parametrize(
    "n,nx,large", [(4000, 4001, False), (4001, 4000, False), (4001, 4001, True)]
)
def test_native_strict_size_boundary(Tech, n, nx, large, monkeypatch):
    module = importlib.import_module("chebfunjax.tech.chebtech")
    calls = []

    def small(c, x):
        calls.append("clenshaw")
        return jnp.zeros(x.shape + c.shape[1:])

    def big(c, x):
        calls.append("ndct")
        assert x.ndim == 1
        return jnp.ones((x.size,) + c.shape[1:])

    monkeypatch.setattr(module, "_clenshaw", small)
    monkeypatch.setattr(module, "_large_chebtech_feval", big)
    assert Tech.from_coeffs(jnp.zeros(n))(jnp.zeros(nx)).shape == (nx,)
    assert calls == ["ndct" if large else "clenshaw"]


def coefficients():
    return jnp.zeros(4001).at[:3].set(jnp.asarray([1.0, 0.25, -0.5]))


def polynomial(x):
    return 1.5 + 0.25 * x - x * x


@pytest.mark.parametrize("Tech", BOTH)
def test_large_scalar_matrix_and_tensor_shapes(Tech):
    c = coefficients()
    for shape in [(4002,), (2, 2001), (2, 3, 667)]:
        x = jnp.linspace(-1.0, 1.0, 4002).reshape(shape)
        actual = Tech.from_coeffs(c)(x)
        assert actual.shape == x.shape
        # Authored low-degree analytic control of the source K=16 NDCT,
        # padded to reach the actual high-degree dispatch boundary.
        assert float(jnp.max(jnp.abs(actual - polynomial(x)))) < 1e-12


@pytest.mark.parametrize("Tech", BOTH)
def test_large_complex_and_exterior_native_angle_projection(Tech):
    x = jnp.linspace(-1.2, 1.2, 4002).astype(jnp.complex128)
    x = x.at[100:200].add(0.3j)
    projected = jnp.cos(jnp.real(jnp.arccos(x)))
    c = coefficients()
    expected = polynomial(projected)
    assert float(jnp.max(jnp.abs(Tech.from_coeffs(c)(x) - expected))) < 1e-12
    # Native NDCT takes imag(vals) for a wholly imaginary coefficient array.
    actual = Tech.from_coeffs(1j * c)(x)
    assert float(jnp.max(jnp.abs(actual - expected))) < 1e-12


@pytest.mark.parametrize("Tech", BOTH)
def test_multicolumn_large_layout_and_rank_error(Tech):
    c = coefficients()
    f = Tech.from_coeffs(jnp.stack((c, 2 * c), axis=-1))
    x = jnp.linspace(-1.0, 1.0, 4002).reshape(2, 2001)
    actual = f(x)
    assert actual.shape == (2, 2001, 2)
    assert (
        float(jnp.max(jnp.abs(actual - jnp.stack((polynomial(x), 2 * polynomial(x)), axis=-1))))
        < 2e-12
    )
    with pytest.raises(ValueError, match="more than two dimensions"):
        f(x.reshape(2, 3, 667))


@pytest.mark.parametrize("Tech", BOTH)
def test_large_jit_jvp_vjp_and_vmap(Tech):
    c = coefficients()
    x = jnp.linspace(-1.0, 1.0, 4002)

    def fn(cc, xx):
        return Tech.from_coeffs(cc)(xx)

    expected = polynomial(x)
    actual = jax.jit(fn)(c, x)
    assert float(jnp.max(jnp.abs(actual - expected))) < 1e-12
    dc = jnp.zeros_like(c).at[0].set(0.3).at[1].set(0.2)
    dx = jnp.full_like(x, 0.4)
    _, tangent = jax.jit(lambda cc, xx: jax.jvp(fn, (cc, xx), (dc, dx)))(c, x)
    expected_tangent = 0.3 + 0.2 * x + 0.4 * (0.25 - 2 * x)
    assert bool(jnp.all(jnp.isfinite(tangent)))
    assert float(jnp.max(jnp.abs(tangent - expected_tangent))) < 2e-12
    gx = jax.jit(jax.grad(lambda xx: jnp.sum(fn(c, xx))))(x)
    assert bool(jnp.all(jnp.isfinite(gx)))
    assert float(jnp.max(jnp.abs(gx - (0.25 - 2 * x)))) < 2e-12
    # Each vmapped invocation retains4002points and therefore uses NDCT.
    batched = jax.jit(jax.vmap(lambda xx: fn(c, xx)))(jnp.stack((x, x * 0.9)))
    assert float(jnp.max(jnp.abs(batched - jnp.stack((expected, polynomial(x * 0.9)))))) < 1e-12


@pytest.mark.parametrize("Tech", BOTH)
def test_small_multicolumn_rank_error_and_empty_adaptation(Tech):
    with pytest.raises(ValueError, match="more than two dimensions"):
        Tech.from_coeffs(jnp.ones((2, 2)))(jnp.zeros((2, 2, 2)))
    # Preserve the preexisting Python empty-tech vectorized zero adaptation.
    assert Tech.from_coeffs(jnp.empty((0, 2)))(jnp.zeros((2, 2, 2))).shape == (2, 2, 2, 2)


@pytest.mark.parametrize("Tech", BOTH)
def test_large_mixed_coefficients_and_coefficient_ad(Tech):
    c = coefficients().astype(jnp.complex128) * 1j
    c = c.at[0].add(2.0)
    x = jnp.linspace(-1.1, 1.1, 4002).astype(jnp.complex128).at[300].add(0.2j)
    u = jnp.cos(jnp.real(jnp.arccos(x)))
    assert float(jnp.max(jnp.abs(Tech.from_coeffs(c)(x) - (2 + 1j * polynomial(u))))) < 1e-12
    realx = jnp.linspace(-1.0, 1.0, 4002)

    def fn(cc):
        return Tech.from_coeffs(cc)(realx)

    for cc, dc, expected in [
        (coefficients(), coefficients(), polynomial(realx)),
        (1j * coefficients(), 1j * coefficients(), polynomial(realx)),
        (c, 1j * coefficients(), 1j * polynomial(realx)),
    ]:
        _, tangent = jax.jit(lambda cc, dc: jax.jvp(fn, (cc,), (dc,)))(cc, dc)
        assert bool(jnp.all(jnp.isfinite(tangent)))
        assert float(jnp.max(jnp.abs(tangent - expected))) < 1e-12


@pytest.mark.parametrize("Tech", BOTH)
def test_large_nested_mixed_endpoint_ad(Tech):
    x = jnp.linspace(-1.0, 1.0, 4002)
    c = jnp.zeros(4001).at[2].set(1.0)

    # Differentiate an endpoint query derivative with respect to the T2
    # coefficient, including the endpoint continuation.
    def right_derivative(scale):
        return jax.grad(lambda xx: jnp.sum(Tech.from_coeffs(scale * c)(xx)))(x)[-1]

    actual = jax.jit(jax.grad(right_derivative))(jnp.asarray(0.7))
    assert bool(jnp.isfinite(actual))
    assert abs(float(actual) - 4.0) < 1e-12
    # Reverse coefficient derivative remains finite even with endpoint nodes.
    value = jax.jit(jax.grad(lambda scale: jnp.sum(Tech.from_coeffs(scale * c)(x))))(
        jnp.asarray(0.7)
    )
    assert bool(jnp.isfinite(value))
    assert abs(float(value) - float(jnp.sum(2 * x * x - 1))) < 1e-9


def test_first_large_call_inside_jit_in_fresh_process():
    # A first import under tracing used to save a Bessel-table tracer in the
    # module globals; a second independently compiled call then failed.
    import os
    import subprocess
    import sys

    code = """
import sys
import jax
import jax.numpy as jnp
from chebfunjax.tech.chebtech import Chebtech2
assert 'chebfunjax.utils.ndct_fast' not in sys.modules
c = jnp.zeros(4001).at[0].set(1.).at[1].set(.25)
x = jnp.linspace(-1., 1., 4002)
f = jax.jit(lambda c,x: Chebtech2.from_coeffs(c)(x))
y = f(c,x)
g = jax.jit(lambda c,x: 2*Chebtech2.from_coeffs(c)(x))
z = g(c,x)
assert float(jnp.max(jnp.abs(y-(1+.25*x)))) < 1e-12
assert float(jnp.max(jnp.abs(z-2*y))) < 1e-12
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("Tech", BOTH)
def test_large_second_query_derivative_and_exterior(Tech):
    x = jnp.linspace(-1.0, 1.0, 4002)
    c = coefficients()

    def first(xx):
        return jax.grad(lambda z: jnp.sum(Tech.from_coeffs(c)(z)))(xx)

    second = jax.jit(jax.grad(lambda xx: jnp.sum(first(xx))))(x)
    assert bool(jnp.all(jnp.isfinite(second)))
    assert float(jnp.max(jnp.abs(second + 2))) < 2e-12
    outside = x.at[0].set(-1.1).at[-1].set(1.1)
    exterior = jax.jit(first)(outside)
    assert bool(jnp.all(exterior[jnp.asarray([0, -1])] == 0))
