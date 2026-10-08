"""Source real/imag values and even Fourier/Nyquist regression controls."""
import jax.numpy as jnp
import pytest

from chebfunjax.tech.trigtech import Trigtech


@pytest.mark.parametrize("n",[4,5,8,9])
@pytest.mark.parametrize("part",["real","imag"])
def test_even_odd_analytic_parts(n,part):
    x=-1+2*jnp.arange(n)/n
    values=2+jnp.cos(jnp.pi*x)+1j*(3+jnp.sin(jnp.pi*x))
    f=Trigtech.from_values(values)
    g=getattr(f,part)()
    z=jnp.linspace(-.973,.981,47)
    expected=2+jnp.cos(jnp.pi*z) if part=="real" else 3+jnp.sin(jnp.pi*z)
    assert jnp.max(jnp.abs(g(z)-expected)) < 2e-14
    assert g.is_real
    if part=="imag":
        assert len(g)==n  # Source imag has no simplify.

@pytest.mark.parametrize("n",[4,5,8,9])
def test_tiny_real_component_preserved(n):
    x=-1+2*jnp.arange(n)/n
    g=Trigtech.from_values(1e-16*(2+jnp.cos(jnp.pi*x))+1j*jnp.ones(n)).real()
    assert jnp.max(jnp.abs(g(x)-1e-16*(2+jnp.cos(jnp.pi*x)))) < 1e-30

@pytest.mark.parametrize("part",["real","imag"])
def test_empty_parts(part):
    assert getattr(Trigtech.empty(),part)().isempty()


def test_values_are_exact_and_coefficient_transforms_invalidate():
    values=jnp.asarray([1e-16+1j,2e-16+1j,3e-16+1j,2e-16+1j])
    f=Trigtech.from_values(values)
    assert jnp.array_equal(f.values,values)
    assert jnp.array_equal((f*1j).values,values*1j)
    g=f.prolong(7)
    assert g._values is None
    assert len(g.values)==7
    assert g.diff()._values is None


def test_dynamic_values_leaf_jvp():
    import jax
    values=jnp.asarray([1.,2.,3.,2.])
    tangent=jnp.asarray([.2,.3,.4,.3])
    def extract(v):
        return (Trigtech.from_values(v)*1j).imag().values
    primal, derivative=jax.jit(lambda v,d:jax.jvp(extract,(v,),(d,)))(values,tangent)
    assert jnp.max(jnp.abs(primal-values)) < 1e-14
    assert jnp.max(jnp.abs(derivative-tangent)) < 1e-14
