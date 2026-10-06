"""Pinned MATLAB ODESOL empty-option and domain-sort controls.

Provenance
----------
MATLAB source : @chebfun/odesol.m
Chebfun commit: 7574c77
These are focused source-contract controls, not original MATLAB test-file assertions.
"""
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from chebfunjax.utils import ode_solution
from chebfunjax.utils.ode_solution import _odesol_from_dense


@pytest.mark.parametrize("options", [None, {}, [], ()])
def test_empty_options_fall_back_to_first_solver_extdata(options, monkeypatch):
    seen = {}

    def fake_construct(interpolants, domain, states, relative_tolerance,
                       absolute_tolerance, *, check):
        seen.update(relative=relative_tolerance, absolute=absolute_tolerance,
                    check=check, domain=domain, states=states)
        return "converted"

    monkeypatch.setattr(ode_solution, "_odesol_from_dense", fake_construct)
    solution = SimpleNamespace(
        y=jnp.array([[1.0, 2.0]]),
        sol=lambda x: jnp.asarray(x)[None, :],
        extdata={"options": {"RelTol": 2e-5, "AbsTol": 3e-7,
                             "happinessCheck": "classicCheck"}},
    )
    assert ode_solution.odesol(solution, (0.0, 1.0), options) == "converted"
    assert seen["relative"] == 2e-5
    assert seen["absolute"] == 3e-7
    assert seen["check"] == "classic"


def test_single_interpolant_sorts_nonmonotone_source_domain():
    def dense(x):
        # MATLAB deval(sol,x).' returns samples-by-components, even for
        # a one-component ODE solution.
        return (jnp.asarray(x)**2)[:, None]

    source_domain = (0.0, 2.0, 1.0)
    states = dense(jnp.asarray(source_domain)).T
    solution = _odesol_from_dense(dense, source_domain, states, 1e-11, 1e-11)
    assert solution.domain.breakpoints == (0.0, 1.0, 2.0)
    points = jnp.array([0.25, 0.75, 1.25, 1.75])
    actual = solution(points)
    assert actual.shape == (points.size, 1)
    assert jnp.allclose(actual[:, 0], dense(points)[:, 0], atol=5e-11, rtol=0)


def test_restarted_handles_follow_matlab_first_two_sort_indices_without_flip():
    def first(x):
        return jnp.asarray(x)[:, None]

    def second(x):
        return (10.0 + jnp.asarray(x))[:, None]

    domain = (0.0, 2.0, 1.0)  # sort indices begin [0, 2]: no source flip
    solution = _odesol_from_dense([first, second], domain,
                                  jnp.array([[0.0, 2.0, 1.0]]))
    assert solution.domain.breakpoints == (0.0, 1.0, 2.0)
    actual = solution(jnp.array([0.5, 1.5]))
    assert actual.shape == (2, 1)
    assert jnp.allclose(actual[:, 0], jnp.array([0.5, 11.5]), atol=5e-14, rtol=0)


def test_restarted_handles_reverse_when_matlab_first_two_sort_indices_descend():
    def first(x):
        return (10.0 + jnp.asarray(x))[:, None]

    def second(x):
        return (20.0 + jnp.asarray(x))[:, None]

    domain = (1.0, 0.0, 2.0)  # sort indices begin [1, 0]: reverse handles
    solution = _odesol_from_dense([first, second], domain,
                                  jnp.array([[1.0, 0.0, 2.0]]))
    assert solution.domain.breakpoints == (0.0, 1.0, 2.0)
    actual = solution(jnp.array([0.5, 1.5]))
    assert actual.shape == (2, 1)
    assert jnp.allclose(actual[:, 0], jnp.array([20.5, 11.5]), atol=5e-14, rtol=0)
