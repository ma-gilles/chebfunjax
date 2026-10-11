"""Native ultrapts.m rescale skips [-1,1], Chebfun7574c77."""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.utils.quadrature import jacpts, ultrapts


@pytest.mark.parametrize('n,lam', [(2,.6), (42,.3), (251,.7)])
def test_canonical_interval_is_exact_identity(n, lam):
    ref = ultrapts(n,lam,bary=True)
    actual = ultrapts(n,lam,[-1.,1.],bary=True)
    assert all(jnp.array_equal(x,y) for x,y in zip(ref,actual))
    assert jnp.array_equal(actual[0],-actual[0][::-1])
    if n == 251:
        public = jacpts(n,.2,.2,bary=True)
        assert all(jnp.array_equal(x,y) for x,y in zip(ref,public))


def test_traced_canonical_interval():
    compiled = jax.jit(lambda a,b: ultrapts(42,.3,(a,b),bary=True))
    actual = compiled(jnp.float64(-1),jnp.float64(1))
    ref = ultrapts(42,.3,bary=True)
    assert all(jnp.array_equal(x,y) for x,y in zip(ref,actual))
    mapped = compiled(jnp.float64(0),jnp.float64(10))
    eager = ultrapts(42,.3,(0.,10.),bary=True)
    for x,y in zip(mapped,eager):
        assert jnp.max(jnp.abs(x-y)) < 1e-14
