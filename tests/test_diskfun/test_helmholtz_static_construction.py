"""Source word controls for grouped static disk construction, Chebfun7574c77."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.discretization import ultras
from chebfunjax.diskfun._helmholtz import _static_operators, assemble_operator


def _words_equal(a, b):
    np.testing.assert_array_equal(np.asarray(a).view(np.uint64), np.asarray(b).view(np.uint64))


@pytest.mark.parametrize('m', [3, 4, 5, 9, 17, 33])
def test_grouped_static_construction_retains_eager_source_words(m):
    a = jnp.asarray([0., 1.])
    b = jnp.asarray([.5, 0., .5])
    first, second = ultras.diffmat(m, 1), ultras.diffmat(m, 2)
    radial = ultras.multmat(m, a, 1)
    squared = ultras.multmat(m, b, 2)
    cheb_squared = ultras.multmat(m, b, 0)
    conversion = ultras.convertmat(m, 0, 1)
    adjacent = ultras.convertmat(m, 1, 1)
    differential = squared @ second + adjacent @ radial @ first
    actual = _static_operators(m, a, b)
    for x, y in zip(actual, (differential, conversion, cheb_squared)):
        _words_equal(x, y)
    converted = ultras.convertmat(m, 0, 1) @ ultras._prolong_multmat(m, b)
    assert float(converted[0]) == .16666666666666669
    # Nonzero a2 requires the first recurrence; only then is the tail empty
    # (m3) or strictly below source eps (m>=4). No degree-based truncation.
    assert float(converted[2]) != 0
    if m > 3:
        assert bool(jnp.all(jnp.abs(converted[3:]) < jnp.finfo(jnp.float64).eps))


@pytest.mark.parametrize('m', [1, 2])
def test_degenerate_small_shapes_preserve_source_dispatch(m):
    # Existing empty/interior shapes remain valid without entering the fixed
    # nonconstant degree2 branch. Operator has no interior rows at these sizes.
    operator, conversion, squared = assemble_operator(m, .7+.2j)
    assert operator.shape == (0, m)
    _words_equal(conversion, ultras.convertmat(m, 0, 1))
    _words_equal(squared, ultras.multmat(m, jnp.asarray([.5, 0., .5]), 0))


@pytest.mark.parametrize('m,lam,coefficients', [
    (3, 2, [.5, 0., .5]),
    (8, 2, [.5, 0., .125, 0., 0., .25]),
    (8, 3, [.5, 0., 0., .125]),
])
def test_traced_tail_uses_entire_remaining_vector(m, lam, coefficients):
    coefficients = jnp.asarray(coefficients)
    padded = ultras._prolong_multmat(m, coefficients)
    actual = jax.jit(lambda a: ultras._multmat_nonconstant(
        m, a, lam, finite=True, traced_tail=True))(padded)
    if m == 3:
        _words_equal(actual, ultras.multmat(m, coefficients, lam))
        return
    # Beyond fixed disk multipliers, traced scalar arithmetic can change words.
    # An independent physical product checks recurrence/tail correctness without
    # imposing an unsupported generic compiler word-identity contract.
    for x in [-.8, -.2, .4, .9]:
        cheb = [1., x]
        for _ in range(1, len(coefficients)-1):
            cheb.append(2*x*cheb[-1]-cheb[-2])
        f = sum(float(c)*v for c, v in zip(coefficients, cheb))
        basis = [1., 2*lam*x]
        for k in range(1, m-1):
            basis.append(2*(k+lam)/(k+1)*x*basis[-1]
                         -(k+2*lam-1)/(k+1)*basis[-2])
        basis = jnp.asarray(basis)
        for column in range(m-len(coefficients)+1):
            product = basis*actual[:, column]
            expected = f*basis[column]
            scale = jnp.sum(jnp.abs(product))+jnp.abs(expected)
            # Same source operation-count/scale allowance as the independent
            # polynomial controls in test_ultras_multmat_source_tail.py.
            bound = 64*m*jnp.finfo(jnp.float64).eps*scale
            assert float(jnp.abs(jnp.sum(product)-expected)) <= float(bound)
