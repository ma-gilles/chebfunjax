"""Source return type, orientation and technology controls for sum."""
import jax.numpy as jnp

from chebfunjax import chebfun, chebfun2


def test_return_orientation_and_complex_nonconjugating_sum():
    from chebfunjax.chebfun1d.chebfun import Chebfun
    f = chebfun2(lambda x, y: (1+2j)*x + (3-1j)*y, domain=(1.,3.,-2.,1.))
    a, b = f.sum(), f.sum(2)
    assert isinstance(a, Chebfun) and a.is_transposed
    assert isinstance(b, Chebfun) and not b.is_transposed
    expected_a = chebfun(lambda x: 3*(1+2j)*x - 1.5*(3-1j), domain=(1.,3.)).T
    expected_b = chebfun(lambda y: 4*(1+2j)+2*(3-1j)*y, domain=(-2.,1.))
    assert float((a-expected_a).norm()) < 2e-13
    assert float((b-expected_b).norm()) < 2e-13
    assert float((f.approx.sum()-a).norm()) < 1e-14


def test_empty_sum_native_numeric_empty():
    from chebfunjax.chebfun2d.chebfun2 import Chebfun2
    f = Chebfun2.empty()
    assert jnp.asarray(f.sum()).size == 0
    assert f.mean().isempty()
    assert f.std().isempty()


def test_trig_marginal_preserves_technology():
    f = chebfun2(lambda x,y: (2+jnp.cos(jnp.pi*x))*(3+jnp.sin(jnp.pi*y)), trig=True)
    a, b = f.sum(), f.sum(2)
    x = jnp.linspace(-1.,1.,17)
    assert jnp.max(jnp.abs(a(x)-6*(2+jnp.cos(jnp.pi*x)))) < 1e-12
    assert jnp.max(jnp.abs(b(x)-4*(3+jnp.sin(jnp.pi*x)))) < 1e-12
    from chebfunjax.tech.trigtech import Trigtech
    assert isinstance(a.funs[0].tech, Trigtech)
    assert isinstance(b.funs[0].tech, Trigtech)
