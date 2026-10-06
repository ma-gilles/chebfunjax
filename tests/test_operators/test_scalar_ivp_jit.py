"""CPU compilation of the extracted scalar IVP callback.

Provenance
----------
MATLAB source : @chebop/solveivp.m (pointwise operator extraction)
Chebfun commit: 7574c77
Independent oracle : analytic solution of u' - u = sin(t).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.operators import chebop as chebop_module
from chebfunjax.operators.chebop import Chebop, _compile_ivp_rhs


def test_compiled_rhs_preserves_dynamic_state_and_time():
    def rhs(t, y):
        return jnp.stack((y[1], -jnp.sin(t)*y[0]+y[1]**2))

    selected = _compile_ivp_rhs(rhs, 0.0, [1.0, 0.0])
    assert selected is not rhs
    for t, y in [(0.1, [0.5, -0.3]), (0.9, [-2.0, 1.2])]:
        values = jnp.asarray(y)
        npt.assert_allclose(selected(t, values), rhs(t, values),
                            atol=4*np.finfo(float).eps, rtol=0.0)


def test_compilation_failure_is_resolved_before_marching():
    attempts = []

    def rhs(t, y):
        attempts.append(isinstance(t, jax.core.Tracer))
        return jnp.array([math.sin(float(t))+y[0]])

    selected = _compile_ivp_rhs(rhs, 0.0, [1.0])
    assert selected is rhs
    assert attempts == [False, True]
    npt.assert_allclose(selected(0.3, jnp.array([2.0])),
                        [2.0+math.sin(0.3)], atol=0.0, rtol=0.0)
    assert attempts == [False, True, False]


@pytest.mark.parametrize("forcing", [1.0+2.0j, jnp.array([1.0, 2.0])])
def test_real_scalar_contract_rejects_forcing_before_scipy_cast(forcing):
    problem = Chebop(lambda t, u: u.diff()-u, domain=(0.0, 1.0))
    problem.lbc = 1.0
    with pytest.raises(TypeError, match="forcing must be a real scalar"):
        problem.solve_ivp(forcing)


@pytest.mark.parametrize("forcing_kind", ["jax", "math", "python_branch"])
def test_direct_marcher_qualifies_compiled_and_eager_forcing(
        monkeypatch, forcing_kind):
    if forcing_kind == "jax":
        forcing = jnp.sin
    elif forcing_kind == "math":
        def forcing(t):
            return math.sin(float(t))
    else:
        def forcing(t):
            # Both branches represent sin(t); this tests trace rejection,
            # without introducing a separate discontinuous ODE contract.
            if t > 0.5:
                return jnp.sin(t)
            return -jnp.sin(-t)

    choices = []
    original = _compile_ivp_rhs

    def record(rhs, t, y):
        selected = original(rhs, t, y)
        choices.append(selected is not rhs)
        return selected

    monkeypatch.setattr(chebop_module, "_compile_ivp_rhs", record)
    problem = Chebop(lambda t, u: u.diff()-u, domain=(0.0, 1.0))
    problem.ivp_method = "LSODA"  # This test observes the explicit SciPy adapter.
    problem.lbc = 1.0
    # The direct method prevents the public collocation fallback masking
    # either a trace failure or an incorrect compiled callback.
    solution = problem.solve_ivp(forcing)
    assert choices == [forcing_kind == "jax"]
    points = np.linspace(0.0, 1.0, 101)
    expected = 1.5*np.exp(points)-(np.sin(points)+np.cos(points))/2
    npt.assert_allclose(solution(jnp.asarray(points)), expected,
                        atol=5e-10, rtol=0.0)
