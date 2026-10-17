"""Public trigtech source layouts and separate Chebfun internal packing."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech.trigtech import Trigtech


def _tech():
    return Trigtech.from_coeffs(jnp.array([[.5, .5j], [1., 2.], [.5, -.5j]]), is_real=True)


@pytest.mark.parametrize("shape,outshape", [
    ((2, 3), (2, 6)), ((2, 3, 2), (2, 12)),
    ((2, 1, 3), (6, 2)), ((2, 3, 1), (2, 6)),
])
def test_trigtech_column_major_source_layout(shape, outshape):
    x = jnp.linspace(-.8, .8, int(np.prod(shape))).reshape(shape)
    # Source x(:), followed by full component columns; NumPy oracle only.
    xf = np.asarray(x).reshape(-1, order="F")
    expected = np.column_stack((1+np.cos(np.pi*xf), 2+np.sin(np.pi*xf)))
    expected = expected.reshape(outshape, order="F")
    f = _tech()
    for actual in (f(x), jax.jit(lambda t: f(t))(x)):
        assert actual.shape == outshape
        np.testing.assert_allclose(actual, expected, rtol=0, atol=16*np.finfo(float).eps)


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 2)])
def test_chebfun_owns_component_packing_after_raw_trig_evaluation(shape):
    x = jnp.linspace(-.8, .8, int(np.prod(shape))).reshape(shape)
    f = Chebfun([_Piece(tech=_tech(), interval=(-1., 1.))], Domain((-1., 1.)))
    # Chebfun keeps higher query axes; this differs from Trigtech for rank3.
    expected = np.concatenate((1+np.cos(np.pi*np.asarray(x)),
                               2+np.sin(np.pi*np.asarray(x))), axis=1)
    for actual in (f(x), jax.jit(lambda t: f(t))(x)):
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=0, atol=16*np.finfo(float).eps)
