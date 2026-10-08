"""Independent controls for diffmat source options and traced interval scaling."""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.utils.diffmat import diffmat
from chebfunjax.utils.quadrature import chebpts


def test_square_boundary_replacement():
    x = chebpts(8, kind=1)
    d = diffmat(8, 1, 'chebkind1', 'dirichlet')
    rhs = (3*x*x).at[0].set(-1.)
    assert jnp.max(jnp.abs(jnp.linalg.solve(d, rhs)-x**3)) < 2e-13


def test_rectangular_jit_and_domain_gradient():
    def matrix(width):
        return diffmat((6, 8), 2, 'chebkind2', 'chebkind1', domain=(0., width))
    d = matrix(3.)
    assert jnp.max(jnp.abs(jax.jit(matrix)(3.)-d)) < 2e-12
    assert jnp.max(jnp.abs(jax.jacfwd(matrix)(3.)+2*d/3)) < 2e-12


@pytest.mark.parametrize('args', [(8, 2, 'dirichlet'),
                                  ((6, 8), 2, 'periodic'),
                                  (8, 'chebkind1', 'chebkind2', 'leg')])
def test_source_invalid_options(args):
    with pytest.raises(ValueError):
        diffmat(*args)
