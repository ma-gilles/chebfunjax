"""Coefficient controls for @diskfun/helmholtz.m, Chebfun commit7574c77.

Bounds below are new coefficient/residual controls. The original disk L2
predicates live unchanged in test_matlab_port/diskfun/test_helmholtz_matlab.py.
"""
import ast
import inspect
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.diskfun._helmholtz import _border_correction, assemble_operator, solve_modes
from chebfunjax.diskfun.diskfun import _diskfun_helmholtz_ultras

FIXTURE = Path(__file__).parents[1] / 'fixtures' / 'disk_helmholtz_native_20261018.json'


def test_border_correction_source_arithmetic_classes():
    # These inverse columns are realized by the finite triangular system
    # [[1,0],[-1e308,1]], with b=[0,1e308] and w=[1,0]. Source helmholtz.m
    # multiplies sum(invBb(2:end))*invBw before dividing by its denominator.
    b, w = [0., 1e308], [1., 1e308]
    expected = [entry-(sum(b[1:])*weight)/(1+sum(w[1:])) for entry,weight in zip(b,w)]
    assert expected[0] == -1 and math.isinf(expected[1]) and expected[1] < 0
    actual = _border_correction(jnp.array([b]),jnp.array([w]))[0]
    assert float(actual[0]) == expected[0]
    assert math.isinf(float(actual[1])) and float(actual[1]) < 0
    # An exact zero Sherman-Morrison denominator retains IEEE zero/zero NaNs;
    # the source has no regularization or finite-result fallback.
    zero_denominator = _border_correction(jnp.array([[0.,0.]]),jnp.array([[1.,-1.]]))
    assert bool(jnp.all(jnp.isnan(zero_denominator)))
    nonfinite = _border_correction(jnp.array([[0.,jnp.nan]]),jnp.array([[1.,1.]]))
    assert bool(jnp.all(jnp.isnan(nonfinite)))


def test_wavenumber_promotes_before_squaring():
    # The pre-port solver converted K to Python complex, hence binary64
    # components before squaring. Integer overflow and lower-precision
    # squaring must not change the operator for the same represented K.
    for value in [10**10, 10**20, jnp.float32(.7), jnp.complex64(.7+.2j)]:
        promoted = complex(value) if jnp.iscomplexobj(value) else float(value)
        actual, _, _ = assemble_operator(9,value)
        expected, _, _ = assemble_operator(9,promoted)
        assert bool(jnp.array_equal(actual,expected))
        traced = jax.jit(lambda k: assemble_operator(9,k)[0])(jnp.asarray(promoted))
        assert bool(jnp.array_equal(traced,expected))


@pytest.mark.parametrize('case_index', range(4))
def test_native_source_coefficients_and_independent_bordered_residual(case_index):
    case = json.loads(FIXTURE.read_text())['cases'][case_index]
    K = complex(*case['K'])
    operator, conversion, _ = assemble_operator(17, K)
    source_operator = jnp.array(case['operator_re']) + 1j*jnp.array(case['operator_im'])
    assert float(jnp.max(jnp.abs(operator-source_operator))) < 2e-13
    forcing, boundary = jnp.array(case['forcing']), jnp.array(case['boundary'])
    actual = solve_modes(operator, conversion[:-2], forcing, boundary, False)
    expected = jnp.array(case['coefficients_re']) + 1j*jnp.array(case['coefficients_im'])
    assert float(jnp.max(jnp.abs(actual-expected))) < 2e-12
    # Direct dense border, independent of the Sherman-Morrison correction.
    for column, mode in enumerate(range(-6, 6)):
        parity = mode % 2
        interior = (source_operator-mode**2*conversion[:-2])[parity::2, parity::2]
        system = jnp.concatenate((jnp.ones((1, interior.shape[1])), interior), axis=0)
        rhs = jnp.concatenate((boundary[column:column+1], forcing[column, parity:-2:2]))
        coeffs = actual[parity::2, column]
        scale = jnp.linalg.norm(system, jnp.inf)*jnp.linalg.norm(coeffs, jnp.inf)+jnp.linalg.norm(rhs,jnp.inf)
        assert float(jnp.linalg.norm(system@coeffs-rhs,jnp.inf)/scale) < 3e-14
    if case_index == 3:
        # The source never discards an imaginary component of K².
        assert float(jnp.max(jnp.abs(jnp.imag(operator)))) > 0
        assert float(jnp.max(jnp.abs(jnp.imag(actual)))) > 0


@pytest.mark.parametrize('n', [10, 12])
def test_real_fourier_completion_matches_full_solve(n):
    operator, conversion, _ = assemble_operator(13, .7)
    forcing = jnp.zeros((n, 13), dtype=complex).at[n//2, 0].set(1)
    boundary = jnp.zeros(n, dtype=complex)
    # A conjugate pair in both parities exercises negative/positive modes.
    for mode in [1, 2]:
        forcing = forcing.at[n//2-mode, mode].set(.25+.125j)
        forcing = forcing.at[n//2+mode, mode].set(.25-.125j)
        boundary = boundary.at[n//2-mode].set(.125+.0625j)
        boundary = boundary.at[n//2+mode].set(.125-.0625j)
    reduced = solve_modes(operator, conversion[:-2], forcing, boundary, True)
    full = solve_modes(operator, conversion[:-2], forcing, boundary, False)
    assert float(jnp.max(jnp.abs(reduced-full))) < 1e-13


def test_solver_boundary_contains_no_numpy_provider():
    tree = ast.parse(inspect.getsource(_diskfun_helmholtz_ultras))
    assert not any(isinstance(node,ast.Name) and node.id == 'np' for node in ast.walk(tree))
    # The coefficient kernel also traces without callbacks into a host provider.
    op, conv, _ = assemble_operator(9, .7+2e-15j)
    traced = str(jax.make_jaxpr(lambda f,b: solve_modes(op,conv[:-2],f,b,False))(
        jnp.zeros((8,9)),jnp.zeros(8)))
    assert 'pure_callback' not in traced and 'io_callback' not in traced
