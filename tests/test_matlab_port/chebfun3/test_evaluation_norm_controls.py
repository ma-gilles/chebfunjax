"""Independent analytic controls for factor norms and evaluation dispatch."""
import jax
import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3


def test_complex_constant_norm():
    f = chebfun3(lambda x, y, z: 2+3j, domain=(0, 1, 0, 2, 0, 3))
    assert abs(f.norm()-jnp.sqrt(78)) < 2e-13


def test_separable_polynomial_norm():
    f = chebfun3(lambda x, y, z: x*y*z)
    assert abs(f.norm()-jnp.sqrt(8/27)) < 2e-13


def test_norm_core_gradient():
    f = chebfun3(lambda x, y, z: 1)
    def norm_of_scale(scale):
        return Chebfun3(cols=f.cols, rows=f.rows, tubes=f.tubes,
                        core=f.core*scale, domain=f.domain).norm()
    assert abs(jax.grad(norm_of_scale)(2.)-jnp.sqrt(8)) < 2e-13


def test_empty_norm_and_evaluation():
    f = Chebfun3.empty()
    assert f.norm().size == 0
    assert f(0, 0, 0).size == 0


def test_periodic_sine_norm():
    f = chebfun3(lambda x, y, z: jnp.sin(jnp.pi*(x+y+z)), trig=True)
    assert abs(f.norm()-2) < 2e-13
