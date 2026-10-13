"""Dtype-stable TreeVar unary RHS JIT and analytic tangent controls.

Provenance
----------
MATLAB source: @treeVar/treeVar.m and tests/treeVar/test_univariate.m.
Chebfun commit: 7574c77.
JAX differentiation is supplemental; complex checks use real-linear JVPs.
"""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.operators.treevar import to_first_order

NAMES = ("abs conj real imag sin cos tan sinh cosh tanh exp expm1 "
         "asinh atan uminus uplus").split()


def _value(name, z):
    if name == "uminus":
        return -z
    if name == "uplus":
        return z
    return getattr(jnp, {"asinh": "arcsinh", "atan": "arctan"}.get(name, name))(z)


def _tangent(name, z, dz):
    if name == "abs":
        return jnp.real(jnp.conj(z)*dz)/jnp.abs(z)
    if name == "conj":
        return jnp.conj(dz)
    if name == "real":
        return jnp.real(dz)
    if name == "imag":
        return jnp.imag(dz)
    derivatives = {
        "sin": lambda: jnp.cos(z), "cos": lambda: -jnp.sin(z),
        "tan": lambda: 1/jnp.cos(z)**2,
        "sinh": lambda: jnp.cosh(z), "cosh": lambda: jnp.sinh(z),
        "tanh": lambda: 1/jnp.cosh(z)**2,
        "exp": lambda: jnp.exp(z), "expm1": lambda: jnp.exp(z),
        "asinh": lambda: 1/jnp.sqrt(1+z*z),
        "atan": lambda: 1/(1+z*z), "uminus": lambda: -1.,
        "uplus": lambda: 1.,
    }
    return derivatives[name]()*dz


def _rhs(name):
    def op(u):
        value = -u if name == "uminus" else +u if name == "uplus" else getattr(u, name)()
        return u.diff(2)+u.diff()+2*value
    return to_first_order(op, 3.2, (0., 2.))[0]


@pytest.mark.parametrize("name", NAMES)
def test_real_jit_and_analytic_derivatives(name):
    rhs = _rhs(name)
    state = jnp.asarray([.4, .2])
    expected = jnp.asarray([[state[1]], [3.2-state[1]-2*_value(name, state[0])]])
    got = jax.jit(rhs)(1., state)
    bound = 10*jnp.finfo(jnp.float64).eps
    assert got.dtype == jnp.float64
    assert jnp.linalg.norm(got-expected) < bound
    slope = _tangent(name, state[0], 1.)
    expected_jac = jnp.asarray([[0., 1.], [-2*slope, -1.]])
    assert jnp.linalg.norm(jax.jacfwd(lambda u: rhs(1., u).ravel())(state)
                           - expected_jac) < bound
    assert jnp.linalg.norm(jax.grad(lambda u: jnp.sum(rhs(1., u)))(state)
                           - jnp.sum(expected_jac, axis=0)) < bound


@pytest.mark.parametrize("name", NAMES)
def test_complex_real_linear_jvp(name):
    rhs = _rhs(name)
    state = jnp.asarray([.4+.3j, .2-.1j])
    tangent = jnp.asarray([.3-.2j, .1+.7j])
    got, dgot = jax.jvp(jax.jit(lambda u: rhs(1., u)), (state,), (tangent,))
    expected = jnp.asarray([[state[1]], [3.2-state[1]-2*_value(name, state[0])]])
    dexpected = jnp.asarray([[tangent[1]], [-tangent[1]-2*_tangent(name, state[0], tangent[0])]])
    bound = 10*jnp.finfo(jnp.float64).eps
    assert got.dtype == jnp.complex128
    assert jnp.linalg.norm(got-expected) < bound
    assert jnp.linalg.norm(dgot-dexpected) < bound
