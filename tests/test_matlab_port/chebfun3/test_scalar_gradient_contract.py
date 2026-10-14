"""Scalar gradient API and compatibility controls beyond original four clauses."""
import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v


def test_gradient_vector_unpack_domain_and_jit():
    domain = (0., 2., -3., 1., 2., 5.)
    f = Chebfun3.from_function(lambda x, y, z: x*x+y*z, domain=domain)
    vector = f.gradient()
    assert isinstance(vector, Chebfun3v)
    gx, gy, gz = vector
    assert gx is vector[0] and gy is vector[1] and gz is vector[2]
    assert all(tuple(g.domain) == domain for g in vector)
    assert not vector.is_transposed
    point = (.7, -.2, 3.)
    np.testing.assert_allclose([g(*point) for g in vector], [1.4, 3., -.2], rtol=0, atol=100*jnp.finfo(float).eps)
    value = jax.jit(lambda x: f.grad()[0](x, point[1], point[2]))(point[0])
    assert abs(float(value)-1.4) < 100*jnp.finfo(float).eps


def test_gradient_empty_components():
    empty = Chebfun3.empty()
    for vector in (empty.grad(), empty.gradient()):
        assert isinstance(vector, Chebfun3v)
        assert vector.n_components == 3 and vector.isempty()
        assert all(component is empty for component in vector)
