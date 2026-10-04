"""Direct controls for the source-shaped scalar projected assembly helper.

Provenance
----------
MATLAB source: ``@chebcolloc/reduce.m``, ``@opDiscretization/{matrix,
getConstraints,partition}.m``, ``@linop/deriveContinuity.m``,
``@valsDiscretization/rhs.m``,
``@valsDiscretization/points.m`` (SHA-256
``f879c65682d77ddb84f1246f14855350432bbd308d571a29a8c644110e438491``),
``@chebcolloc2/{functionPoints,equationPoints,toFunctionOut}.m``, and
``@linop/linsolve.m`` at Chebfun commit ``7574c77``. The independent cubic
oracle below is analytic (``u=x**3-x``), not copied from the assembly helper.

The checks qualify only this bounded assembly primitive. Derivative matrices
are supplied independently by ``diffmat``; adaptive dimensions, coefficient
extraction and public solver dispatch are not tested here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from chebfunjax.operators.chebop import _piecewise_row_scaled_solve
from chebfunjax.operators.piecewise_linear import (
    _build_scalar_projected_system,
    _panel_projection,
    _project_scalar_raw_solution,
)
from chebfunjax.tech.chebtech import Chebtech1
from chebfunjax.utils.diffmat import diffmat
from chebfunjax.utils.quadrature import chebpts

EPS = np.finfo(np.float64).eps
DOMAINS = (-1.0, -0.3, 1.0)
BASE = (8, 13)


def _raw_derivatives(base=BASE):
    return tuple(
        tuple(jnp.asarray(diffmat(n + 2, order, kind=2)) for order in range(3))
        for n in base
    )


def _raw_cubic_values(domains=DOMAINS, base=BASE):
    values = []
    for a, b, d in zip(domains[:-1], domains[1:], base):
        xi = chebpts(d + 2, kind=2)
        x = b * (xi + 1.0) / 2.0 + a * (1.0 - xi) / 2.0
        values.append(x**3 - x)
    return jnp.concatenate(values)


def _closed_form_cubic_coefficients(a, b):
    """Chebyshev coefficients from x=center+halfwidth*t algebra."""
    center = 0.5 * (a + b)
    h = 0.5 * (b - a)
    # t**2=(T2+T0)/2 and t**3=(T3+3*T1)/4.
    return np.array([
        center**3 - center + 1.5 * center * h**2,
        3.0 * center**2 * h - h + 0.75 * h**3,
        1.5 * center * h**2,
        0.25 * h**3,
    ])


def _matrix_action_case(variable_coefficients):
    if variable_coefficients:
        coefficient_fields = tuple(
            (lambda x: 2.0 - x, lambda x: x**2, lambda x: 1.0 + 0.25 * x)
            for _ in BASE
        )

        def rhs(x):
            u = x**3 - x
            return ((1.0 + 0.25 * x) * (6.0 * x)
                    + x**2 * (3.0 * x**2 - 1.0) + (2.0 - x) * u)
    else:
        coefficient_fields = tuple(
            (lambda x: 0.0 * x, lambda x: 0.0 * x, lambda x: 1.0 + 0.0 * x)
            for _ in BASE
        )

        def rhs(x):
            return 6.0 * x

    matrix, rhs_vector = _build_scalar_projected_system(
        jnp.asarray(DOMAINS),
        BASE,
        coefficient_fields,
        _raw_derivatives(),
        tuple(rhs for _ in BASE),
        endpoint_values=(0.0, 0.0),
    )
    raw = _raw_cubic_values()
    expected = jnp.concatenate((
        jnp.zeros((2 * len(BASE),)),
        *(rhs(b * (chebpts(d, kind=1) + 1.0) / 2.0
              + a * (1.0 - chebpts(d, kind=1)) / 2.0)
          for a, b, d in zip(DOMAINS[:-1], DOMAINS[1:], BASE)),
    ))
    action = matrix @ raw
    scale = max(1.0, float(jnp.linalg.norm(matrix, ord=jnp.inf))
                * float(jnp.linalg.norm(raw, ord=jnp.inf)),
                float(jnp.linalg.norm(expected, ord=jnp.inf)))
    assert float(jnp.linalg.norm(action - expected, ord=jnp.inf)) <= 100 * EPS * scale
    rhs_scale = max(1.0, float(jnp.linalg.norm(expected, ord=jnp.inf)))
    npt.assert_allclose(rhs_vector, expected, rtol=0, atol=100 * EPS * rhs_scale)
    return matrix, rhs_vector, raw


def test_unequal_panel_d2_system_action_and_solve_match_cubic_oracle():
    """Direct A@u, solve, constraints and projected output match u=x³−x."""
    matrix, rhs, raw = _matrix_action_case(variable_coefficients=False)
    solution = _piecewise_row_scaled_solve(matrix, rhs)
    npt.assert_allclose(solution, raw, rtol=0, atol=5e-10)
    projected = _project_scalar_raw_solution(jnp.asarray(DOMAINS), BASE, solution)

    for p, (a, b, d, got_values) in enumerate(
        zip(DOMAINS[:-1], DOMAINS[1:], BASE, projected)
    ):
        xi = chebpts(d, kind=1)
        x = b * (xi + 1.0) / 2.0 + a * (1.0 - xi) / 2.0
        expected_values = x**3 - x
        npt.assert_allclose(got_values, expected_values, rtol=0, atol=100 * EPS)

        got_coefficients = np.asarray(Chebtech1.vals2coeffs(got_values))
        expected_coefficients = _closed_form_cubic_coefficients(a, b)
        npt.assert_allclose(
            got_coefficients[:4], expected_coefficients, rtol=0, atol=100 * EPS
        )
        assert np.max(np.abs(got_coefficients[4:]), initial=0.0) <= 100 * EPS


def test_complex_amplitude_preserves_matrix_solution_and_projection_dtype():
    """A complex RHS remains complex through solve, projection and transform."""
    amplitude = 1.0 + 0.5j
    coefficient_fields = tuple(
        (lambda x: 0.0 * x, lambda x: 0.0 * x,
         lambda x: (1.0 + 0.0j) + 0.0 * x)
        for _ in BASE
    )
    rhs_fields = tuple(lambda x: amplitude * 6.0 * x for _ in BASE)
    matrix, rhs = _build_scalar_projected_system(
        jnp.asarray(DOMAINS), BASE, coefficient_fields, _raw_derivatives(),
        rhs_fields, endpoint_values=(0.0j, 0.0j),
    )
    raw = amplitude * _raw_cubic_values()
    solution = _piecewise_row_scaled_solve(matrix, rhs)
    assert jnp.issubdtype(solution.dtype, jnp.complexfloating)
    npt.assert_allclose(solution, raw, rtol=0, atol=5e-10)
    projected = _project_scalar_raw_solution(jnp.asarray(DOMAINS), BASE, solution)
    for a, b, d, got_values in zip(DOMAINS[:-1], DOMAINS[1:], BASE, projected):
        xi = chebpts(d, kind=1)
        x = b * (xi + 1.0) / 2.0 + a * (1.0 - xi) / 2.0
        npt.assert_allclose(got_values, amplitude * (x**3 - x), rtol=0, atol=100 * EPS)
        got_coefficients = np.asarray(Chebtech1.vals2coeffs(got_values))
        expected_coefficients = amplitude * _closed_form_cubic_coefficients(a, b)
        npt.assert_allclose(
            got_coefficients[:4], expected_coefficients, rtol=0, atol=100 * EPS
        )
        assert np.max(np.abs(got_coefficients[4:]), initial=0.0) <= 100 * EPS


def test_unequal_panel_variable_coefficients_and_jit_builder():
    """Variable coefficients are sampled on raw nodes and project correctly."""
    eager_matrix, eager_rhs, raw = _matrix_action_case(variable_coefficients=True)

    build = jax.jit(lambda domains: _build_scalar_projected_system(
        domains,
        BASE,
        tuple((lambda x: 2.0 - x, lambda x: x**2, lambda x: 1.0 + 0.25 * x)
              for _ in BASE),
        _raw_derivatives(),
        tuple(lambda x: (1.0 + 0.25 * x) * 6.0 * x + x**2 * (3.0 * x**2 - 1.0)
              + (2.0 - x) * (x**3 - x) for _ in BASE),
        endpoint_values=(0.0, 0.0),
    ))
    jit_matrix, jit_rhs = build(jnp.asarray(DOMAINS))
    matrix_scale = max(1.0, float(jnp.linalg.norm(eager_matrix, ord=jnp.inf)))
    assert float(jnp.linalg.norm(jit_matrix - eager_matrix, ord=jnp.inf)) <= 1e-9 * matrix_scale
    rhs_scale = max(1.0, float(jnp.linalg.norm(eager_rhs, ord=jnp.inf)))
    npt.assert_allclose(jit_rhs, eager_rhs, rtol=0, atol=100 * EPS * rhs_scale)
    npt.assert_equal(raw.shape, (sum(n + 2 for n in BASE),))
    expected_action = eager_rhs
    action_scale = max(
        1.0,
        float(jnp.linalg.norm(jit_matrix, ord=jnp.inf))
        * float(jnp.linalg.norm(raw, ord=jnp.inf)),
    )
    assert float(jnp.linalg.norm(jit_matrix @ raw - expected_action, ord=jnp.inf)) <= 100 * EPS * action_scale

    solution = _piecewise_row_scaled_solve(jit_matrix, jit_rhs)
    projected = _project_scalar_raw_solution(jnp.asarray(DOMAINS), BASE, solution)
    for a, b, d, values in zip(DOMAINS[:-1], DOMAINS[1:], BASE, projected):
        xi = chebpts(d, kind=1)
        x = b * (xi + 1.0) / 2.0 + a * (1.0 - xi) / 2.0
        npt.assert_allclose(values, x**3 - x, rtol=0, atol=100 * EPS)


def test_projection_jit_and_shape_guards():
    raw = _raw_cubic_values()
    project = jax.jit(lambda x: _project_scalar_raw_solution(
        jnp.asarray(DOMAINS), BASE, x
    ))
    projected = project(raw)
    assert tuple(v.shape for v in projected) == ((8,), (13,))
    with np.testing.assert_raises_regex(ValueError, "raw_solution"):
        _project_scalar_raw_solution(jnp.asarray(DOMAINS), BASE, jnp.r_[raw, 1.0])
    with np.testing.assert_raises_regex(ValueError, "physical_domains"):
        _project_scalar_raw_solution(jnp.asarray((-1.0, 1.0)), BASE, raw)
    with np.testing.assert_raises_regex(ValueError, "at least one panel"):
        _project_scalar_raw_solution(jnp.asarray((0.0,)), (), jnp.asarray(()))
    with np.testing.assert_raises_regex(ValueError, "positive static integers"):
        _project_scalar_raw_solution(jnp.asarray((0.0, 1.0)), (0,), jnp.zeros((2,)))


def test_panel_projection_uses_literal_source_map_at_tiny_interval_endpoints():
    """MATLAB's endpoint operation order avoids midpoint endpoint drift."""
    a, b = 0.2, 0.2000004
    # The midpoint rearrangement rounds the right endpoint one ULP upward here.
    assert 0.5 * ((b - a) * 1.0 + (a + b)) != b
    projection, equation_nodes, raw_nodes = _panel_projection(
        a, b, base_n=7, raw_n=9
    )
    del projection
    raw_reference = np.asarray(chebpts(9, kind=2))
    equation_reference = np.asarray(chebpts(7, kind=1))
    # Independent literal operation order from @valsDiscretization/points.m.
    expected_raw = b * (raw_reference + 1.0) / 2.0 + a * (1.0 - raw_reference) / 2.0
    expected_equation = (
        b * (equation_reference + 1.0) / 2.0
        + a * (1.0 - equation_reference) / 2.0
    )
    npt.assert_array_equal(np.asarray(raw_nodes), expected_raw)
    npt.assert_array_equal(np.asarray(equation_nodes), expected_equation)
    assert float(raw_nodes[0]) == a
    assert float(raw_nodes[-1]) == b
    assert expected_equation.shape == (7,)


def test_three_panel_constraint_grouping_matches_source_order():
    """Source rows group all value jumps before all derivative jumps."""
    domains = jnp.asarray((-1.0, -0.5, 0.5, 1.0))
    base = (5, 7, 6)
    slopes = (1.0, 2.0, 4.0)
    intercepts = (0.0, 3.0, 8.0)
    fields = tuple(
        (lambda x: 0.0 * x, lambda x: 0.0 * x, lambda x: 0.0 * x)
        for _ in base
    )
    matrices = tuple(
        tuple(jnp.asarray(diffmat(n + 2, order, kind=2))
              for order in range(3)) for n in base
    )
    matrix, rhs = _build_scalar_projected_system(
        domains,
        base,
        fields,
        matrices,
        continuity_values=jnp.asarray((-2.5, -6.0, -1.0, -2.0)),
        endpoint_values=(-1.0, 12.0),
    )

    raw_parts = []
    for a, b, n, slope, intercept in zip(
        domains[:-1], domains[1:], base, slopes, intercepts
    ):
        xi = chebpts(n + 2, kind=2)
        x = b * (xi + 1.0) / 2.0 + a * (1.0 - xi) / 2.0
        raw_parts.append(slope * x + intercept)
    raw = jnp.concatenate(raw_parts)
    expected_value_traces = jnp.asarray((-2.5, -6.0))
    expected_derivative_traces = jnp.asarray((-1.0, -2.0))
    expected_endpoint_traces = jnp.asarray((-1.0, 12.0))
    npt.assert_allclose(
        matrix[:2] @ raw, expected_value_traces, rtol=0, atol=100 * EPS
    )
    npt.assert_allclose(
        matrix[4:6] @ raw, expected_endpoint_traces, rtol=0, atol=100 * EPS
    )
    derivative_roundoff_scale = jnp.maximum(
        1.0, jnp.abs(matrix[2:4]) @ jnp.abs(raw)
    )
    # The source discretization test uses an aggregate 1e-9 norm bound. This
    # independent trace control initially used absolute 100*EPS (2.22e-14),
    # but v1 measured a 3.41e-13 derivative-trace difference when large-offset
    # samples cancel. Scale only these derivative rows by their dot-product
    # magnitude, the standard componentwise backward-error bound; value and
    # endpoint traces retain their original absolute 100*EPS checks.
    assert bool(jnp.all(
        jnp.abs(matrix[2:4] @ raw - expected_derivative_traces)
        <= 100 * EPS * derivative_roundoff_scale
    ))
    npt.assert_allclose(rhs[:6], jnp.concatenate((
        expected_value_traces, expected_derivative_traces, expected_endpoint_traces,
    )), rtol=0, atol=0)


def test_coefficient_gradient_matches_analytic_second_derivative_action():
    """JAX differentiation through projected coefficient assembly is valid."""
    domains = jnp.asarray(DOMAINS)
    raw = _raw_cubic_values()
    raw_derivatives = _raw_derivatives()

    def action_sum(alpha):
        fields = tuple(
            (lambda x: 0.0 * x, lambda x: 0.0 * x,
             lambda x, alpha=alpha: alpha + 0.0 * x)
            for _ in BASE
        )
        matrix, _rhs = _build_scalar_projected_system(
            domains, BASE, fields, raw_derivatives,
            endpoint_values=(0.0, 0.0),
        )
        return jnp.sum(matrix @ raw)

    got = jax.grad(action_sum)(jnp.asarray(1.0))
    expected = sum(
        jnp.sum(6.0 * (b * (chebpts(n, kind=1) + 1.0) / 2.0
                       + a * (1.0 - chebpts(n, kind=1)) / 2.0))
        for a, b, n in zip(DOMAINS[:-1], DOMAINS[1:], BASE)
    )
    unit_c2_fields = tuple(
        (lambda x: 0.0 * x, lambda x: 0.0 * x, lambda x: 1.0 + 0.0 * x)
        for _ in BASE
    )
    unit_c2_matrix, _rhs = _build_scalar_projected_system(
        domains, BASE, unit_c2_fields, raw_derivatives,
        endpoint_values=(0.0, 0.0),
    )
    operator_start = 2 * len(BASE)
    action_roundoff_scale = float(jnp.sum(
        jnp.abs(unit_c2_matrix[operator_start:]) @ jnp.abs(raw)
    ))
    npt.assert_allclose(
        got, expected, rtol=0,
        atol=100 * EPS * max(1.0, action_roundoff_scale),
    )
