"""Literal native cumsum clauses10–19/30; captured source inputs, original bounds."""

import warnings
from contextlib import contextmanager
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf

import chebfunjax as cj

EPS = np.finfo(float).eps


@contextmanager
def source_exponent_warning_policy():
    # Native clauses11/12 temporarily suppress this identifier, then restore.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"^CHEBFUN:SINGFUN:plus:exponentDiff:")
        yield


def draw(k):
    with np.load(Path(__file__).parent / "fixtures/cumsum_native_inputs.npz") as data:
        return jnp.asarray(data[f"draw{k}"].reshape(-1))


def maxabs(x):
    return float(jnp.max(jnp.abs(x)))


def test_source_10():
    x = draw(1)
    power = -1.64
    f = cj.chebfun(lambda x: (x + 2) ** power, domain=[-2, 7], exps=[power, 0])
    g = f.cumsum()
    exact = (x + 2) ** (power + 1) / (power + 1)
    assert maxabs(g(x) - exact) < 1e4 * EPS * maxabs(exact), "source10"


def test_source_11():
    f = cj.chebfun(
        [jnp.sin, lambda x: 1 / jnp.sqrt(1 + x), lambda x: x + 1],
        domain=[-2, -1, 0, 1],
        exps=[0, 0, -0.5, 0, 0, 0],
    )
    with source_exponent_warning_policy():
        g = f.cumsum()
    exacts = [
        lambda x: jnp.cos(-2.0) - jnp.cos(x),
        lambda x: 2 * jnp.sqrt(1 + x),
        lambda x: x * x / 2 + x + 2,
    ]
    for k, op in enumerate(exacts, 2):
        x = draw(k)
        exact = op(x)
        err = g(x) - exact
        assert maxabs(err - jnp.mean(err)) < 1e7 * EPS * maxabs(exact), "source11"


def test_source_12():
    op = lambda x: jnp.sin(100 * x) / (jnp.sqrt(x + 1) * jnp.sqrt((x - 1) + 0j))  # noqa: E731 - native anonymous function
    f = cj.chebfun(op, domain=[-1, 1], exps=[-0.5, -0.5])
    with source_exponent_warning_policy():
        g = f.cumsum()
    x = draw(5)
    check = cj.chebfun(op, domain=[-0.9, 0.9], splitting=True).cumsum()
    exact = check(x)
    err = g(x) - exact
    assert maxabs(err - jnp.mean(err)) < 1e2 * (5e5 * EPS * maxabs(exact)), "source12"


def test_source_13():
    x = draw(6)
    g = cj.chebfun(lambda x: jnp.exp(-x * x), domain=[-np.inf, np.inf]).cumsum()
    exact = jnp.sqrt(jnp.pi) * erf(x) / 2 + jnp.sqrt(jnp.pi) / 2
    assert maxabs(g(x) - exact) < 5e4 * EPS * g.vscale, "source13"


def test_source_14():
    x = draw(7)
    g = cj.chebfun(lambda x: 5 * x, domain=[1, np.inf], exps=[0, 1]).cumsum()
    exact = 5 * x * x / 2 - 5 / 2 + g(1.0)
    assert maxabs(g(x) - exact) < 200 * EPS * g.vscale, "source14"


def test_source_15():
    g = cj.chebfun([jnp.exp, lambda x: jnp.sin(3 * x)], domain=[-np.inf, -1, 3 * np.pi]).cumsum()
    x1, x2 = draw(8), draw(9)
    errors = jnp.concatenate(
        [
            jnp.ravel(g(x1) - jnp.exp(x1)),
            jnp.ravel(g(x2) - (-jnp.cos(3 * x2) / 3 + jnp.cos(-3.0) / 3 + jnp.exp(-1.0))),
        ]
    )
    assert maxabs(errors) < 1e6 * EPS * g.vscale, "source15"


def test_source_16_to_17():
    x = cj.chebfun("x")
    f = cj.heaviside(x + 0.5) + cj.heaviside(x) + cj.heaviside(x - 0.5)
    fp = f.diff().diff().cumsum()
    assert float((f - fp.cumsum()).norm()) < 100 * EPS, "source16"
    f = x.sign() - x
    f2 = f.diff().cumsum()
    assert float((f - f2).norm()) < 100 * EPS, "source17"


def test_source_18_to_19():
    x = cj.chebfun("x")
    f, g = cj.heaviside(x), cj.heaviside(x - 0.5)
    s = cj.Chebfun.horzcat(f.diff(), g.diff())
    integral = s.cumsum()
    assert float((integral[0] - f).norm()) < 100 * EPS, "source18"
    assert float((integral[1] - g).norm()) < 100 * EPS, "source19"


def test_source_30():
    x = cj.chebfun("x")
    n = 2
    f1 = (1 + x) ** 0.5 * (1 - x) ** 0 * cj.jacpoly(n, 0, 0.5)
    f2 = (1 + x) ** 0.5 * (1 - x) ** EPS * cj.jacpoly(n, EPS, 0.5)
    assert float((f1.cumsum() - f2.cumsum()).norm(np.inf)) < 10 * EPS, "source30"


def test_source_warning_filter_restoration():
    before = list(warnings.filters)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with source_exponent_warning_policy():
            warnings.warn("CHEBFUN:SINGFUN:plus:exponentDiff: source scoped diagnostic")
            warnings.warn("unrelated retained diagnostic")
        assert [str(item.message) for item in caught] == ["unrelated retained diagnostic"]
        warnings.warn("CHEBFUN:SINGFUN:plus:exponentDiff: restored diagnostic")
        assert len(caught) == 2
    assert warnings.filters == before


def test_native_fixture_matches_exact_source_stream_order():
    with np.load(Path(__file__).parent / "fixtures/cumsum_native_inputs.npz") as data:
        assert np.array_equal(data["xr"], 2 * np.random.RandomState(7681).rand(1000) - 1)
        rng = np.random.RandomState(6178)
        bounds = [
            (-2, 7),
            (-2, -1),
            (-1, 0),
            (0, 1),
            (-0.9, 0.9),
            (-100, 100),
            (1, 100),
            (-1e6, -1),
            (-1, 3 * np.pi),
        ]
        for k, (left, right) in enumerate(bounds, 1):
            assert np.array_equal(data[f"draw{k}"], (right - left) * rng.rand(100) + left)
