"""Independent source vectorCheck controls for disk construction."""
import jax.numpy as jnp
import pytest

from chebfunjax.diskfun.diskfun import _disk_vector_check


def test_vector_check_detects_matrix_product():
    def op(theta, r):
        if theta.ndim == 0:
            return r*jnp.sin(theta)
        return r@jnp.sin(theta)
    with pytest.warns(UserWarning, match="CHEBFUN:DISKFUN:constructor:vectorize"):
        assert _disk_vector_check(op, "polar")


def test_vector_check_cartesian_probe_coordinates():
    calls = []
    def op(x, y):
        calls.append((x, y))
        return x*y
    assert not _disk_vector_check(op, "cart")
    assert len(calls) == 5
    theta, r = jnp.meshgrid(jnp.asarray([-jnp.pi, jnp.pi])/3+(2*jnp.pi)/3,
                            jnp.asarray([0., 1.])/2+1/3)
    assert jnp.array_equal(calls[0][0], r*jnp.cos(theta))
    assert jnp.array_equal(calls[0][1], r*jnp.sin(theta))


def test_vector_check_scalar_output():
    assert not _disk_vector_check(lambda x, y: 1., "cart")


def test_even_zero_projection_preserves_storage():
    from chebfunjax.diskfun.diskfun import diskfun
    f = diskfun(jnp.zeros((5, 4)))
    assert f.rows[0].n == 4
    assert f.cols[0].n == 5
    assert f.projectOntoBMCII().length() == f.length()


@pytest.mark.parametrize("count", [4, 5])
def test_projection_retains_nonzero_even_modes(count):
    from chebfunjax.diskfun.diskfun import Diskfun
    from chebfunjax.tech.chebtech import Chebtech2
    from chebfunjax.tech.trigtech import Trigtech
    from chebfunjax.utils.quadrature import trigpts
    theta, _ = trigpts(count)
    row = Trigtech.from_values(jnp.cos(2*jnp.pi*theta))
    col = Chebtech2.from_coeffs(jnp.asarray([.5, 0., .5]))
    f = Diskfun(cols=[col], rows=[row], pivots=jnp.ones(1),
                idx_plus=(0,), idx_minus=())
    g = f.projectOntoBMCII()
    assert g.rows[0].n == count
    query = jnp.linspace(-jnp.pi, jnp.pi, 29)
    expected = .36*jnp.cos(2*query)
    assert jnp.max(jnp.abs(g(query, jnp.full_like(query, .6))-expected)) < 3e-15


@pytest.mark.parametrize("op", ["sin(x)", lambda x: x, lambda: 1.])
def test_single_argument_rejected_like_source(op):
    from chebfunjax.diskfun import diskfun
    with pytest.raises(ValueError) as caught:
        diskfun(op)
    assert caught.value.identifier == "CHEBFUN:DISKFUN:CONSTRUCTOR:toFewInputArgs"
