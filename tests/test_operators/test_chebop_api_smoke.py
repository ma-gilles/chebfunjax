"""Coverage smokes for the chebop API surface added in the Fable 5
parity campaign (feval/arithmetic, string constructor, bc keywords,
linearize, multi-output helpers, domain validation).  Functional
correctness is pinned by the tests/test_matlab_port/chebop ports; the
CI coverage job excludes those, so these fast mirrors keep the
coverage gate honest.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.operators.chebop import (
    Chebop,
    SystemSolution,
    _op_from_string,
    _validate_chebop_domain,
)

jax.config.update("jax_enable_x64", True)


def _n(f, d=(-1.0, 1.0)):
    xs = jnp.linspace(d[0] + 1e-9, d[1] - 1e-9, 17)
    return float(jnp.max(jnp.abs(jnp.asarray(f(xs)))))


def test_feval_and_arithmetic():
    u = cj.chebfun(jnp.sin)
    A = Chebop(lambda u: u.diff() + u)
    B = Chebop(lambda u: u)
    assert _n(A(u) - (u.diff() + u)) < 1e-10
    assert _n(A.feval(u) - A(u)) == 0.0
    assert _n((A + B)(u) - (A(u) + B(u))) < 1e-12
    assert _n((A - B)(u) - (A(u) - B(u))) < 1e-12
    assert _n((-A)(u) + A(u)) < 1e-12
    assert _n((2.0 * A)(u) - 2.0 * A(u)) < 1e-12
    assert _n((A * 2.0)(u) - 2.0 * A(u)) < 1e-12
    assert _n(A.eye()(u) - u) == 0.0
    with pytest.raises(TypeError):
        A * B
    N2 = Chebop(lambda x, u, v: [u.diff() + v, v.diff() - u])
    out = N2(u, u.cos())
    assert isinstance(out, (list, tuple, SystemSolution))
    out2 = N2([u, u.cos()])
    assert _n(out[0] - out2[0]) < 1e-12


def test_string_ctor_and_bc_keywords():
    f = _op_from_string("u`+sin(u)+x")
    u = cj.chebfun(jnp.cos)
    x = cj.chebfun(lambda t: t)
    assert _n(f(x, u) - (u.diff() + u.sin() + x)) < 1e-10
    L = Chebop("u``+u", domain=(0.0, 1.0))
    L.lbc = "dirichlet"
    L.rbc = "neumann"
    v = L.solve(1.0)
    assert abs(float(v(jnp.asarray(0.0)))) < 1e-8
    L2 = Chebop(lambda u: u.diff(2) + u, domain=(0.0, 1.0))
    L2.rbc = [1.0, "neumann"]
    assert L2.rbc == [1.0, 0.0]
    with pytest.raises(ValueError):
        L2.lbc = "robin"


def test_linearize_and_info():
    D = Chebop(lambda u: u.diff(), domain=(0.0, 1.0))
    L, res, is_linear = D.linearize()
    assert all(is_linear)
    N = Chebop(lambda u: u.diff(2) - u ** 3, domain=(0.0, 1.0))
    N.lbc = 1.0
    N.rbc = 0.0
    u0 = cj.chebfun(lambda t: 1.0 - t, domain=(0.0, 1.0))
    J = N.linearize(u0)
    du = J.solve(N * u0)
    assert _n(du, (0.0, 1.0)) < 10.0
    N2 = Chebop(lambda u: u.diff(2) + u, domain=(0.0, 1.0))
    N2.lbc = 0.0
    N2.rbc = 1.0
    sol, info = N2.solve_with_info(0.0)
    assert isinstance(info, dict) and info["isLinear"]


def test_system_deal_and_domain_validation():
    L = Chebop(lambda t, u, v: [u.diff() - v, v.diff() + u],
               domain=(0.0, 1.0))
    L.lbc = lambda u, v: [u, v - 1.0]
    uv = L.solve([0.0, 0.0])
    a, b = uv.deal()
    xs = jnp.linspace(0.0, 1.0, 9)
    assert float(jnp.max(jnp.abs(jnp.asarray(a(xs)) - jnp.sin(xs)))) < 1e-6
    assert np.allclose(_validate_chebop_domain((0, 1, 2)), (0.0, 1.0, 2.0))
    for bad in ((1.0,), (1.0, 0.0), "xy"):
        with pytest.raises((ValueError, TypeError)):
            _validate_chebop_domain(bad)


def test_krylov_smoke():
    """pcg/gmres on the identity-coefficient operator (fast case)."""
    N = Chebop(lambda x, u: (-1.0) * u.diff(2) + u)
    N.bc = 0.0
    f = cj.chebfun(lambda x: 1 - 3 * x ** 2)
    u = N.solve(f)
    assert _n(u - N.pcg(f)) < 1e-6
    assert _n(u - N.gmres(f)) < 1e-6


def test_svds_smoke():
    """svds of d/dx on [0, pi]: singular values 2, 1, 0."""
    N = Chebop(lambda x, u: u.diff(), domain=(0.0, float(np.pi)))
    U, S, V = N.svds(3, n=48)
    s = np.diag(np.asarray(S))
    assert np.max(np.abs(s - np.array([2.0, 1.0, 0.0]))) < 1e-7
    assert _n(N(V[0]) - float(s[0]) * U[0], (0.0, float(np.pi))) < 1e-6


def test_maxnorm_blowup_and_determine_discretization_smoke():
    """maxnorm event halts the march (u' = u^2 blows up); NaN padding
    and NaN-aware inf-norm; 'values' keyword dispatch."""
    from chebfunjax.chebpref import ChebopPref
    N = Chebop(lambda t, y: y.diff() - y * y, domain=(0.0, 3.0))
    N.lbc = 1.0
    N.maxnorm = 5.0
    y = N.solve(0.0)
    assert y.isnan()
    assert abs(float(y.norm(jnp.inf)) - 5.0) < 0.05
    N2 = Chebop(lambda u: u.diff(2), domain=(0.0, 1.0))
    out = N2.determine_discretization(2, ChebopPref())
    assert out.discretization == "chebcolloc2"
