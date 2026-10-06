"""Independent controls for native ode113 tolerance shape and complex-state contracts.

Provenance: MATLAB R2025b private/odearguments.m, SHA-256
453659846f9a461918f6a217bfb08abe2c3ecd0d07e17259a7019158184443f3.
The complex-state control preserves existing behavior. Tolerance controls cover
source ``length(atol)==neq`` validation and ``atol=atol(:)`` shape handling. These are supplemental analytic
controls, not numbered Chebfun MATLAB assertions. Bounds match the existing
native Adams polynomial controls (100 binary64 eps, absolute, zero rtol).
"""

import jax.numpy as jnp

# uses-numpy: host-side assertions compare JAX outputs with independent exact polynomials.
import numpy as np
import pytest

from chebfunjax.utils.native_ode113 import native_ode113

_EPS = np.finfo(np.float64).eps
_BOUND = 100.0 * _EPS


def test_complex_initial_state_with_real_rhs_keeps_promoted_state_dtype():
    """A real callback must not erase the imaginary initial-state component."""
    solved = native_ode113(
        lambda t, y: jnp.ones_like(jnp.real(y)),
        (0.0, 1.0),
        jnp.asarray([[1.0 + 2.0j]]),
        {"RelTol": 1e-8, "AbsTol": 1e-10},
    )
    x = jnp.linspace(0.0, 1.0, 101)
    exact = (1.0 + 2.0j + x)[None, :]
    values, derivatives = solved["sol"](x, return_derivative=True)

    assert jnp.iscomplexobj(solved["y"])
    np.testing.assert_allclose(values, exact, atol=_BOUND, rtol=0.0)
    np.testing.assert_allclose(derivatives, np.ones((1, 101)), atol=_BOUND, rtol=0.0)
    np.testing.assert_array_equal(solved["sol"](solved["x"]), solved["y"])


@pytest.mark.parametrize("abs_tol", [[[1e-10], [1e-11]], [[1e-10, 1e-11]]], ids=["column", "row"])
def test_abs_tol_vector_shapes_use_source_component_order(abs_tol):
    """MATLAB ``atol(:)`` accepts row/column vectors for a two-state system."""
    solved = native_ode113(
        lambda t, y: jnp.asarray([1.0, 2.0 * t]),
        (0.0, 1.0),
        jnp.asarray([[0.0], [0.0]]),
        {"RelTol": 1e-8, "AbsTol": abs_tol},
    )
    x = jnp.linspace(0.0, 1.0, 101)
    exact = jnp.stack((x, x * x))
    exact_derivative = jnp.stack((jnp.ones_like(x), 2.0 * x))
    values, derivatives = solved["sol"](x, return_derivative=True)

    np.testing.assert_allclose(values, exact, atol=_BOUND, rtol=0.0)
    np.testing.assert_allclose(derivatives, exact_derivative, atol=_BOUND, rtol=0.0)
    np.testing.assert_array_equal(solved["sol"](solved["x"]), solved["y"])



@pytest.mark.parametrize(
    "initial,options",
    [
        ([0.0] * 4, {"AbsTol": [[1e-10, 1e-11], [1e-12, 1e-13]]}),
        ([0.0] * 2, {"AbsTol": [[1e-10], [1e-11]], "NormControl": "on"}),
        ([0.0] * 2, {"AbsTol": [[1e-10], [-1e-11]]}),
    ],
    ids=["matrix-source-length-mismatch", "norm-control-requires-scalar", "negative-component"],
)
def test_abs_tol_shape_adapter_preserves_source_validation(initial, options):
    """Source length, scalar NormControl and positivity checks still reject."""
    with pytest.raises(ValueError, match="AbsTol"):
        native_ode113(lambda t, y: jnp.ones_like(y), (0.0, 1.0), initial, options)
