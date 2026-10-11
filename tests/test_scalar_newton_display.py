"""Pinned source event/format contracts; fixtures are structural, not MATLAB runs."""

import math
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.operators._scalar_display import boundary_error, format_display_info
from chebfunjax.operators.scalar_newton import _conditions, _iteration_info


@pytest.mark.parametrize("lam", [1.0, 0.125])
def test_literal_source_iteration_format(lam):
    row = dict(
        iteration=1, normDelta=0.5, contraction=0.25, **{"lambda": lam}, lenDelta=17, lenU=33
    )
    # Literal format strings from displayInfoIter.m, with the same fields.
    spec = (
        " %02d %12.2e %12.2e %11.4f %7i %9i \n"
        if lam == 1
        else " %02d %12.2e %12.2e %12.2e %6i %9i \n"
    )
    expected = spec % (1, 0.5, 0.25, lam, 17, 33)
    assert format_display_info("iter", "iter", step=row) == expected


def test_header_exact_initial_off_and_case_contracts():
    dash = "-" * 62
    assert (
        format_display_info("init", "iter")
        == "Iter.   || du ||   Contraction    stepsize   len(du)   len(u)\n" + dash + "\n"
    )
    assert (
        format_display_info("exactInitial", "iter")
        == "\n"
        + dash
        + "\nInitial guess appears to be a solution of the BVP. \nInitial guess returned as the solution.\n"
        + dash
        + "\n"
    )
    for mode in ["init", "iter", "exactInitial", "final"]:
        assert format_display_info(mode, "off") == ""
    assert format_display_info("init", "ITER") == format_display_info("init", "iter")
    assert format_display_info("exactInitial", "ITER") == format_display_info(
        "exactInitial", "iter"
    )
    assert format_display_info("final", "ITER") == ""  # native final strcmp


@pytest.mark.parametrize(
    "backend,basis",
    [("chebcolloc1", "Values"), ("chebcolloc2", "Values"), ("ultraS", "Coefficients")],
)
def test_final_actual_error_and_basis_contract(backend, basis):
    expected = "-" * 62 + "\nNewton's method converged in 1 iteration.\n"
    expected += f"Discretization basis used: {basis}. \n"
    expected += "Final error estimate: 1.25e-05 (differential equation) \n"
    expected += "%30.2e (boundary conditions). \n\n" % 0.125
    assert (
        format_display_info(
            "final", "final", iterations=1, backend=backend, error=1.25e-5, boundary_error=0.125
        )
        == expected
    )
    assert "2 iterations." in format_display_info("final", "iter", iterations=2)
    assert "NaN (differential equation)" in format_display_info("final", "iter", error=float("nan"))


def test_nan_contraction_has_literal_matlab_capitalization():
    row = dict(
        iteration=2, normDelta=0.25, contraction=math.nan, **{"lambda": 1.0}, lenDelta=3, lenU=7
    )
    text = format_display_info("iter", "iter", step=row)
    assert text.split()[2] == "NaN"
    assert "nan" not in text


def test_actual_lengths_not_discretization_size():
    u = Chebfun.from_coeffs(jnp.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), domain=(0.0, 1.0))
    delta = Chebfun.from_coeffs(jnp.asarray([0.0, 0.0, 1.0]), domain=(0.0, 1.0))
    row = _iteration_info([u], [delta], 2, 0.25, 0.5, 1.0)
    assert row["lenDelta"] == len(delta) == 3
    assert row["lenU"] == len(u) == 7
    assert row["normDelta"] == 0.25 and row["contraction"] == 0.5 and row["lambda"] == 1.0


def test_boundary_groups_preserve_single_evaluation_and_source_squares():
    calls = []

    def left(u):
        calls.append("left")
        return lambda x: 3.0

    def right(u):
        calls.append("right")
        return lambda x: 4.0

    def general(u):
        calls.append("general")
        return [1j, 2j]

    op = SimpleNamespace(_lbc_raw=left, _rbc_raw=right, _bc_general=general, domain=(0.0, 1.0))
    groups = []
    values = _conditions(op, object(), group_lengths=groups)
    assert calls == ["left", "right", "general"]
    assert values == [3.0, 4.0, 1j, 2j]
    assert groups == [("endpoint", 0, 1), ("endpoint", 1, 2), ("general", 2, 4)]
    assert boundary_error(values, groups) == math.sqrt(30)
    # Native endpoint squares have no conjugation; general 2-norm does.
    assert boundary_error([1j], [("endpoint", 0, 1)]) == 1j
    assert boundary_error([1j], [("general", 0, 1)]) == 1
    assert isinstance(boundary_error(values, groups), __import__("jax").Array)
    assert boundary_error([], []) == 0
    assert boundary_error([jnp.asarray([3.0, 4.0])], [("general", 0, 1)]) == 5
