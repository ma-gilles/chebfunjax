"""Independent numerical controls for the JAX builtin Airy replacement."""
# uses-numpy: independent SciPy reference generation and assertion arrays.
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import airy

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.utils.airy_general import airy_all


@pytest.mark.parametrize('radius', [0., .5, 2., 6., 12., 20., 23.999999,
                                   24.000001, 30., 60., 100.])
def test_complex_plane_oracle(radius):
    z = radius*np.exp(1j*np.linspace(-np.pi, np.pi, 49))
    actual = np.asarray(airy_all(jnp.asarray(z)))
    reference = np.asarray(airy(z))
    # Numerical replacement gate, not a source assertion tolerance. The
    # absolute floor avoids demanding relative accuracy at Airy zeros.
    assert np.max(np.abs(actual-reference)/np.maximum(1., np.abs(reference))) < 1e-12


def test_decaying_solution_retains_relative_accuracy():
    x = np.array([.55, 1., 4., 12., 20., 24., 24.01, 32., 100.])
    actual = np.asarray(airy_all(jnp.asarray(x)))[:2]
    reference = np.asarray(airy(x))[:2]
    assert np.max(np.abs((actual-reference)/reference)) < 5e-13


@pytest.mark.parametrize('value', [0., .55, -24., 32., 2+3j, -10+17j])
def test_jit_tangent_against_independent_airy_ode(value):
    x = jnp.asarray(value)
    actual, tangent = jax.jit(lambda z: jax.jvp(airy_all, (z,), (jnp.ones_like(z),)))(x)
    reference = airy(value)
    expected = np.array([reference[1], value*reference[0], reference[3], value*reference[2]])
    assert np.max(np.abs(np.asarray(tangent)-expected)/np.maximum(1., np.abs(expected))) < 1e-12
    assert np.max(np.abs(np.asarray(actual)-reference)/np.maximum(1., np.abs(reference))) < 1e-12


@pytest.mark.parametrize('kind', [0, 1, 2, 3])
def test_complex_scaling_and_domain(kind):
    domain = (1., 2.)
    f = chebfun(lambda x: (1+.2j)*x, domain=domain)
    g = f.airy(kind, 1)
    xx = np.linspace(1., 2., 17)
    z = (1+.2j)*xx
    factor = np.exp((2/3)*z**1.5) if kind < 2 else np.exp(-np.abs((2/3)*(z**1.5).real))
    expected = airy(z)[kind]*factor
    assert np.max(np.abs(np.asarray(g(jnp.asarray(xx)))-expected)) < 1e-12
    assert tuple(g.domain.breakpoints) == domain


def test_empty_nonfinite_and_parameter_errors():
    assert all(a.shape == (0,) for a in airy_all(jnp.empty((0,))))
    assert np.isnan(np.asarray(airy_all(jnp.array([np.inf, -np.inf, np.nan])))).all()
    f = chebfun(lambda x: x)
    for kind, scale in [(4, 0), (0, 2)]:
        with pytest.raises(ValueError, match='airy:params'):
            f.airy(kind, scale)
