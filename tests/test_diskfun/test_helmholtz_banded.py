"""Independent coefficient controls for disk Helmholtz band elimination.

Source: @diskfun/helmholtz.m, Chebfun commit7574c77. Original bounds unchanged.
"""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.diskfun._helmholtz import _border_correction, _solve_parity, assemble_operator


@pytest.mark.parametrize('m,K', [(9, 0.), (17, 5.5), (33, .7+.2j), (17, 2.4048255577)])
@pytest.mark.parametrize('parity', [0, 1])
def test_banded_matches_independent_dense_source_system(m, K, parity):
    operator, conversion, _ = assemble_operator(m, K)
    modes = jnp.array([parity, parity+2, parity+4])
    indices = jnp.arange(3)
    forcing = jnp.sin(jnp.arange(3*m).reshape(3,m)) * .125
    boundary = jnp.array([.25, -.125, .5])
    interior = operator[parity::2, parity::2][None] - modes[:,None,None]**2*conversion[:-2][parity::2,parity::2][None]
    q = interior.shape[-1]
    first = jnp.zeros((3,1,q)).at[:,0,0].set(1)
    systems = jnp.concatenate((first, interior), axis=1)
    rows, cols = jnp.indices((q,q))
    assert bool(jnp.all(jnp.where(jnp.abs(cols-rows)>2, systems == 0, True)))
    rhs = jnp.concatenate((boundary[:,None], forcing[:,parity:-2:2]),axis=1)
    weights = jnp.zeros_like(rhs).at[:,0].set(1)
    dense = jnp.linalg.solve(systems,jnp.stack((rhs,weights),axis=-1))
    expected = _border_correction(dense[...,0],dense[...,1])
    actual = _solve_parity(operator,conversion[:-2],forcing,boundary,modes,indices,parity=parity)
    # Independent full border residual avoids reusing the rank-one correction.
    # Both solves retain the same residual bound, including near the first J0
    # root, where coefficients reach 1e11 and absolute forward error is ill-scaled.
    bordered = systems.at[:,0,:].set(1)
    matrix_norm = jnp.linalg.norm(bordered,ord=jnp.inf,axis=(1,2))
    rhs_norm = jnp.linalg.norm(rhs,ord=jnp.inf,axis=1)
    residual_norms, solution_norms = [], []
    for solution in (actual, expected):
        residual = jnp.einsum('bij,bj->bi',bordered,solution)-rhs
        residual_norm = jnp.linalg.norm(residual,ord=jnp.inf,axis=1)
        solution_norm = jnp.linalg.norm(solution,ord=jnp.inf,axis=1)
        scale = matrix_norm*solution_norm+rhs_norm
        assert float(jnp.max(residual_norm/scale)) < 3e-14
        residual_norms.append(residual_norm)
        solution_norms.append(solution_norm)
    # A posteriori forward bound from A(x-y)=r_x-r_y, allowing for the
    # binary64 q-term dot products used to measure those residuals. gamma_q
    # is the standard q*eps/(1-q*eps) accumulation bound; factor4 covers
    # complex multiplication/addition. No observed forward delta sets this.
    eps = jnp.finfo(jnp.float64).eps
    gamma = q*eps/(1-q*eps)
    inverse_norm = jnp.linalg.norm(jnp.linalg.inv(bordered),ord=jnp.inf,axis=(1,2))
    rounding = 4*gamma*(matrix_norm*(solution_norms[0]+solution_norms[1])+2*rhs_norm)
    bound = inverse_norm*(residual_norms[0]+residual_norms[1]+rounding)
    difference = jnp.linalg.norm(actual-expected,ord=jnp.inf,axis=1)
    assert bool(jnp.all(difference <= bound))
    compiled = jax.jit(lambda f,b:_solve_parity(operator,conversion[:-2],f,b,modes,indices,parity=parity))(forcing,boundary)
    assert bool(jnp.array_equal(actual, compiled))
    if m == 17 and K == 5.5 and parity == 0:
        # Exact source first column: [1,25/24,121/48,...]. The largest
        # candidate is two rows below the diagonal, beyond tridiagonal LU.
        # The entries use rounded conversion products and a mode-square
        # subtraction. Eight binary64 eps covers those few operations here
        # (observed rational-vs-assembly difference was 1.1102230246251565e-15).
        assert float(jnp.max(jnp.abs(systems[1,:3,0]-jnp.array([1,25/24,121/48])))) < 8*jnp.finfo(jnp.float64).eps
        assert int(jnp.argmax(jnp.abs(systems[1,:3,0]))) == 2


@pytest.mark.parametrize('value', [0., jnp.nan, jnp.inf])
def test_singular_and_nonfinite_system_retains_nonfinite_status(value):
    operator = jnp.full((3,5), value)
    conversion = jnp.zeros((3,5))
    actual = _solve_parity(operator,conversion,jnp.ones((1,5)),jnp.ones(1),
                           jnp.zeros(1),jnp.zeros(1,dtype=int),parity=0)
    assert bool(jnp.all(jnp.isnan(actual)))
