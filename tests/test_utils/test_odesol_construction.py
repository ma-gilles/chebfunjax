"""Independent trajectories and source ODESOL construction policy.

Provenance
----------
MATLAB source : @chebfun/odesol.m
Chebfun commit: 7574c77
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.chebfun1d.chebfun import _Piece
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.utils.ode_solution import _odesol_from_dense


def _trajectory(x):
    return jnp.stack((jnp.exp(x), jnp.exp(-2*x), jnp.zeros_like(x)), axis=-1)


@pytest.mark.parametrize("domain", [(0., 1.), (1., 0.), (0., .4, 1.), (1., .4, 0.)])
def test_source_global_array_fit_on_declared_domain(domain):
    states = _trajectory(jnp.linspace(0, 1, 25)).T
    solution = _odesol_from_dense(_trajectory, domain, states, 1e-11, 1e-11)
    assert solution.n_columns == 3
    assert solution.domain.breakpoints == tuple(sorted(domain))
    assert solution.ishappy
    x = jnp.linspace(0, 1, 47)
    npt.assert_allclose(jax.jit(lambda f, x: f(x))(solution, x), _trajectory(x),
                        atol=5e-11, rtol=0)


@pytest.mark.parametrize("reverse", [False, True])
def test_restart_interpolants_keep_order_after_final_value_domain_sort(reverse):
    # Continuous value with an actual derivative jump at the restart.
    def first(x):
        return jnp.stack((x, jnp.zeros_like(x)), axis=-1)

    def second(x):
        return jnp.stack((2*x-.4, jnp.zeros_like(x)), axis=-1)

    domain = (0., .4, 1.)
    operators = [first, second]
    if reverse:
        domain = domain[::-1]
        operators.reverse()
    solution = _odesol_from_dense(operators, domain, jnp.array([[0., .4, 1.6], [0., 0., 0.]]))
    x = jnp.array([0., .2, .4, .6, 1.])
    npt.assert_allclose(solution(x)[:, 0], [0., .2, .4, .8, 1.6],
                        atol=10*np.finfo(float).eps, rtol=0)
    assert solution.domain.breakpoints == (0., .4, 1.)


def test_dense_fit_preserves_complex_components_and_zero_state():
    def operator(x):
        return jnp.stack((jnp.exp(1j*x), jnp.zeros_like(x)), axis=-1)

    solution = _odesol_from_dense(operator, (-1., 1.), operator(jnp.linspace(-1, 1, 17)).T,
                                  1e-12, 1e-12)
    x = jnp.linspace(-1, 1, 53)
    npt.assert_allclose(solution(x), operator(x), atol=2e-12, rtol=0)


def test_odesol_disables_offgrid_sample_evaluation():
    observed = []

    def operator(x):
        observed.append(np.asarray(x))
        return jnp.stack((jnp.ones_like(x), jnp.zeros_like(x)), axis=-1)

    _odesol_from_dense(operator, (-1., 1.), jnp.array([[1., 1.], [0., 0.]]))
    assert observed
    assert not any(x.shape == (2,) and np.array_equal(
        x, [-.357998918959666, .036785641195074]) for x in observed)


def test_unhappy_global_fit_retries_source_splitting_policy(monkeypatch):
    import importlib
    module = importlib.import_module("chebfunjax.chebfun1d.chebfun")
    original = _Piece.from_function.__func__
    calls = []

    def force_first_unhappy(cls, *args, **kwargs):
        piece = original(cls, *args, **kwargs)
        if not calls:
            calls.append("global")
            return piece.with_tech(Chebtech2(coeffs=piece.coeffs, ishappy=False))
        return piece

    split_original = module._construct_with_splitting

    def record_retry(*args, **kwargs):
        calls.append(kwargs)
        return split_original(*args, **kwargs)

    monkeypatch.setattr(_Piece, "from_function", classmethod(force_first_unhappy))
    monkeypatch.setattr(module, "_construct_with_splitting", record_retry)
    solution = _odesol_from_dense(_trajectory, (0., 1.), _trajectory(jnp.array([0., 1.])).T)
    assert solution.ishappy
    assert calls[0] == "global"
    assert calls[1]["split_max_length"] == 20000
    assert calls[1]["sample_test"] is False
    npt.assert_allclose(solution(jnp.array([.1, .7])), _trajectory(jnp.array([.1, .7])),
                        atol=2e-6, rtol=0)


def test_dense_evaluator_error_propagates_without_alternate_fit():
    def invalid(x):
        raise RuntimeError("dense output unavailable")

    with pytest.raises(RuntimeError, match="dense output unavailable"):
        _odesol_from_dense(invalid, (0., 1.), jnp.ones((1, 2)))


@pytest.mark.parametrize("entry", ["top", "class", "factory"])
def test_public_odesol_reads_options_and_returns_time(entry):
    import chebfunjax as cj
    choices = {"top": cj.odesol, "class": cj.Chebfun.odesol,
               "factory": cj.chebfun.odesol}
    source_solution = SimpleNamespace(
        y=_trajectory(jnp.array([0., 1.])).T,
        sol=lambda x: _trajectory(x).T,
        extdata={"options": {"RelTol": 1e-11, "AbsTol": [1e-11]*3,
                             "happinessCheck": "standardCheck"}},
    )
    t, y = choices[entry](source_solution, (0., 1.), return_time=True)
    x = jnp.linspace(0, 1, 41)
    npt.assert_allclose(t(x), x, atol=2*np.finfo(float).eps, rtol=0)
    npt.assert_allclose(y(x), _trajectory(x), atol=5e-11, rtol=0)
    assert t.domain == y.domain


@pytest.mark.parametrize("check", ["standard", "strict", "classic", "plateau"])
def test_dense_fit_accepts_source_builtin_happiness_variants(check):
    def operator(x):
        return jnp.stack((jnp.exp(x), jnp.exp(-2*x)), axis=-1)

    solution = _odesol_from_dense(operator, (0., 1.), operator(jnp.array([0., 1.])).T,
                                  1e-11, 1e-11, check=check)
    x = jnp.linspace(0, 1, 43)
    npt.assert_allclose(solution(x), operator(x), atol=5e-11, rtol=0)


@pytest.mark.parametrize("restarted", [False, True])
def test_dense_breakpoint_values_follow_single_handle_or_cell_contract(restarted):
    # getValuesAtBreakpoints.m evaluates a single handle at all breaks;
    # a cell of restarted handles uses FUN limits after chopping instead.
    def operator(x):
        return jnp.exp(x)[:, None]

    bounds = (0., .4, 1.)
    dense = [operator, operator] if restarted else operator
    solution = _odesol_from_dense(dense, bounds, operator(jnp.asarray(bounds)).T,
                                  1e-4, 1e-4)
    limits = jnp.stack([solution.funs[0](0.),
                        (solution.funs[0](.4)+solution.funs[1](.4))/2,
                        solution.funs[1](1.)])
    exact = operator(jnp.asarray(bounds))
    assert float(jnp.max(jnp.abs(limits-exact))) > 1e-10
    expected = limits if restarted else exact
    npt.assert_array_equal(solution(jnp.asarray(bounds)), expected)
    npt.assert_array_equal(jax.jit(lambda f, x: f(x))(solution, jnp.asarray(bounds)),
                           expected)
