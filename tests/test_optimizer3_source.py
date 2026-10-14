"""Three-variable source mechanics; native trace comparisons are separate."""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.utils._active_set_box_qp import box_qp
from chebfunjax.utils._active_set_qp import choltrap, compdir


@pytest.mark.parametrize('compiled',[False,True])
def test_three_variable_cholesky_and_projected_direction(compiled):
    h=jnp.diag(jnp.asarray([4.,16.,64.]))
    factor= jax.jit(choltrap) if compiled else choltrap
    ell,direction,present=factor(h)
    assert not present
    assert jnp.array_equal(ell,jnp.diag(jnp.asarray([2.,4.,8.])))
    assert jnp.array_equal(direction,jnp.zeros(3))
    run=jax.jit(compdir) if compiled else compdir
    result=run(jnp.eye(3),h,jnp.asarray([4.,-16.,32.]))
    assert jnp.array_equal(result.step,jnp.asarray([-1.,1.,-.5]))

@pytest.mark.parametrize('pivot',[0,1,2])
def test_three_variable_negative_pivot(pivot):
    diagonal=jnp.asarray([4.,16.,64.]).at[pivot].set(-1.)
    _,direction,present=choltrap(jnp.diag(diagonal))
    assert present
    assert direction@jnp.diag(diagonal)@direction<0

@pytest.mark.parametrize('gradient,expected',[
    ([-4.,16.,-32.],[1.,-1.,.5]),
    ([-16.,64.,-256.],[2.,-2.,2.]),
])
def test_three_variable_box_quadratic(gradient,expected):
    result=box_qp(jnp.diag(jnp.asarray([4.,16.,64.])),jnp.asarray(gradient),
                  jnp.full(3,-2.),jnp.full(3,2.),max_iterations=60)
    assert jnp.array_equal(result.x,jnp.asarray(expected))
