"""Core-suite smokes for the ultraS/chebcolloc1 backends, treeVar and
the chebop altdisc solver.

The full behavior is pinned by the MATLAB port tree
(tests/test_matlab_port/{linop,treeVar,chebop}); the coverage job
ignores that tree, so these fast smokes keep the new modules counted
(same pattern as test_plotting_smoke2.py — see pyproject coverage
note).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.operators.blocklinop import linop
from chebfunjax.operators.blocks import D, I, eval_at, mult
from chebfunjax.operators.chebmatrix import ChebMatrix

jax.config.update("jax_enable_x64", True)

DOM = (-1.0, 1.0)


@pytest.mark.parametrize("disc", ["ultraS", "chebcolloc1"])
def test_altdisc_scalar_solve_and_eigs(disc):
    L = linop(D(DOM) ** 2)
    L = L.addbc(eval_at(-1.0, DOM), 0.0).addbc(eval_at(1.0, DOM), 0.0)
    u = L.linsolve(lambda t: jnp.exp(t), n=48, discretization=disc)[0]
    resid = float((u.diff(2) - cj.chebfun(lambda t: jnp.exp(t),
                                          domain=DOM)).norm())
    assert resid < 1e-9
    lam, vecs = L.eigs(k=3, n=48, discretization=disc)
    exact = -(np.pi * np.arange(1, 4) / 2) ** 2
    got = np.sort(np.asarray(lam).real)[::-1]
    assert np.max(np.abs(got - exact) / np.abs(exact)) < 1e-8


@pytest.mark.parametrize("disc", ["ultraS", "chebcolloc1"])
def test_altdisc_system_solve(disc):
    x = cj.chebfun(lambda t: t, domain=DOM)
    L = linop(ChebMatrix([[D(DOM), -I(DOM)], [I(DOM), D(DOM)]]))
    L = L.addbc([eval_at(-1.0, DOM),
                 -eval_at(1.0, DOM)], 0.0)
    L = L.addbc([eval_at(1.0, DOM), eval_at(-1.0, DOM)], 1.0)
    u = L.linsolve([x, 0 * x], n=48, discretization=disc)
    r1 = float((u[0].diff() - u[1] - x).norm())
    r2 = float((u[0] + u[1].diff()).norm())
    assert max(r1, r2) < 1e-8


@pytest.mark.parametrize("disc", ["ultraS", "chebcolloc1"])
def test_altdisc_expm(disc):
    L = linop(D(DOM) ** 2)
    L = L.add_constraint(eval_at(-1.0, DOM), 0.0)
    L = L.add_constraint(eval_at(1.0, DOM), 0.0)
    u0 = cj.chebfun(lambda t: (1 - t * t) * jnp.sin(t), domain=DOM)
    u = L.expm(0.01, u0, n=48, discretization=disc)
    ref = L.expm(0.01, u0, n=48)
    assert float((u[0] - ref[0]).norm()) < 1e-8


def test_altdisc_piecewise_coefficient():
    x = cj.chebfun(lambda t: t, domain=DOM)
    c = 1.0 + (x.abs() < 0.5)
    L = linop(-0.1 * D(DOM) ** 2 + mult(c))
    L = L.addbc(eval_at(-1.0, DOM), 0.0).addbc(eval_at(1.0, DOM), 0.0)
    lam_u, _ = L.eigs(k=2, n=33, discretization="ultraS")
    lam_c, _ = L.eigs(k=2, n=33, discretization="chebcolloc1")
    assert np.max(np.abs(np.sort(np.asarray(lam_u).real)
                         - np.sort(np.asarray(lam_c).real))) < 1e-8


def test_treevar_core():
    from chebfunjax.operators.treevar import (
        TreeVar,
        TreeVarError,
        print_tree,
        sort_conditions,
        to_first_order,
    )
    f, idx, dom_out, coeffs, orders = to_first_order(
        lambda u: 5 * (u.diff(2) + 3 * u), 0, DOM)
    assert np.allclose(np.ravel(f(1, [2, 1])), [1, -6])
    assert list(idx) == [1] and list(orders) == [2]
    f2, *_ = to_first_order(
        lambda t, u, v: [u.diff() + 3 * v, v.diff() - u.sin()],
        [1, 2], DOM)
    got = np.ravel(f2(0.3, [0.4, 0.2]))
    want = [1 - 3 * 0.2, 2 + np.sin(0.4)]
    assert np.allclose(got, want)
    assert list(sort_conditions(lambda u: [u.diff(), u], DOM, 2)) \
        == [2, 1]
    with pytest.raises(TreeVarError):
        to_first_order(lambda u: (-u).diff(), 0, DOM)
    u = TreeVar()
    s = print_tree((u.cos() + u.diff(2)).tree)
    assert "plus" in s and "diff" in s
    import matplotlib
    matplotlib.use("Agg")
    assert (u.exp() * 2).plot() is not None


def test_chebop_altdisc_linear():
    from chebfunjax.operators.chebop import Chebop
    from chebfunjax.operators.chebop_altdisc import solve_bvp_altdisc
    N = Chebop(lambda x, u: u.diff(2) + u, domain=(0.0, 1.0))
    N.lbc = 0.0
    N.rbc = 1.0
    for disc in ("ultraS", "chebcolloc1"):
        u = solve_bvp_altdisc(N, 0.0, disc, n=48)[0]
        exact = np.sin(0.4) / np.sin(1.0)
        assert abs(float(u(jnp.asarray(0.4))) - exact) < 1e-8


def test_chebfun_get_and_fold():
    from chebfunjax.chebfun3d.chebfun3 import Chebfun3
    f = cj.chebfun(lambda x: jnp.sin(x), domain=(-1.0, 0.0, 1.0))
    c = f.get("coeffs")
    assert isinstance(c, list) and len(c) == 2
    assert f.get("domain") == (-1.0, 0.0, 1.0)
    assert abs(f.get("vscale") - np.sin(1.0)) < 1e-12
    T = jnp.arange(24.0).reshape(2, 3, 4)
    M = Chebfun3.unfold(T, 2)
    assert M.shape == (3, 8)
    back = Chebfun3.fold(M, (2, 3, 4), 2, (1, 3))
    assert float(jnp.max(jnp.abs(back - T))) == 0.0


def test_chebpref_and_matlab_expr_smoke():
    from chebfunjax.chebpref import ChebfunPref, ChebopPref
    from chebfunjax.utils.matlab_expr import matlab_expression
    p = ChebfunPref({"splitting": True, "testPref": "t"})
    assert p.splitting and p.testPref == "t"
    assert ChebfunPref.mergeTechPrefs(p, {"testPref": "q"}) \
        .testPref == "q"
    q = ChebopPref()
    q.plotting = "on"
    assert q.plotting == "on" and q.discretization == "chebcolloc2"
    fn = matlab_expression("cos(x) + sin(x.*y)", ("x", "y"))
    assert abs(float(fn(jnp.asarray(0.3), jnp.asarray(0.4)))
               - (np.cos(0.3) + np.sin(0.12))) < 1e-15


def test_slices_and_chebmatrix_plots_smoke():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from chebfunjax.diskfun.diskfun import Diskfun
    from chebfunjax.spherefun.spherefun import Spherefun
    f = Spherefun.from_function(
        lambda lam, th: jnp.cos(th) + jnp.sin(lam) * jnp.sin(th))
    sl = f.slice_theta(0.7)
    assert abs(float(sl(jnp.asarray(0.2)))
               - (np.cos(0.7) + np.sin(0.2) * np.sin(0.7))) < 1e-10
    _ = f.slice_lambda(0.3)
    _ = f.slice_z(0.5)
    g = Diskfun.from_function(lambda t, r: r * jnp.cos(t))
    assert abs(float(g.slice_r(0.5)(jnp.asarray(0.1)))
               - 0.5 * np.cos(0.1)) < 1e-10
    _ = g.slice_theta(0.1)
    assert g.nonzero_poles is not None and len(g) >= 1
    _ = g.pivot_values, g.pivot_locations

    x = cj.chebfun(lambda t: t)
    M = ChebMatrix([[x], [x ** 2], [1.0]])
    assert M.plot() is not None
    assert M.plotcoeffs() is not None
    Q = ChebMatrix([[x.sin(), (2 * x).sin()]])
    assert Q.waterfall() is not None
    assert M.change_tech("chebtech2")[0, 0] is x
    assert M.is_not_diff_or_int == [[True], [True], [True]]
    plt.close("all")


def test_chebop_altdisc_nonlinear_and_systems():
    """Nonlinear chord-Newton path + system solve under each
    discretization (coverage for the Frechet/bc-collection code)."""
    from chebfunjax.operators.chebop import Chebop
    from chebfunjax.operators.chebop_altdisc import solve_bvp_altdisc

    # Nonlinear scalar BVP: u'' + sin(u) = 0, u(0)=0, u(1)=0.5.
    N = Chebop(lambda x, u: u.diff(2) + u.sin(), domain=(0.0, 1.0))
    N.lbc = 0.0
    N.rbc = 0.5
    for disc in ("ultraS", "chebcolloc1"):
        u = solve_bvp_altdisc(N, 0.0, disc, n=48)[0]
        # residual check at interior points
        res = u.diff(2) + u.sin()
        xs = jnp.linspace(0.05, 0.95, 9)
        assert float(jnp.max(jnp.abs(jnp.asarray(res(xs))))) < 1e-6
        assert abs(float(u(jnp.asarray(0.0)))) < 1e-8
        assert abs(float(u(jnp.asarray(1.0))) - 0.5) < 1e-8

    # 2x2 first-order linear system: u' = v, v' = -u with u(0)=0, v(0)=1.
    N = Chebop(lambda x, u, v: [u.diff() - v, v.diff() + u],
               domain=(0.0, 1.0))
    N.lbc = lambda u, v: [u, v - 1.0]
    for disc in ("ultraS", "chebcolloc1"):
        U = solve_bvp_altdisc(N, [0.0, 0.0], disc, n=48)
        xs = jnp.linspace(0.0, 1.0, 7)
        assert float(jnp.max(jnp.abs(
            jnp.asarray(U[0](xs)) - jnp.sin(xs)))) < 1e-7
        assert float(jnp.max(jnp.abs(
            jnp.asarray(U[1](xs)) - jnp.cos(xs)))) < 1e-7


def test_chebop_altdisc_generalized_eigs():
    """Generalized eigenproblem u'' = lambda*u under ultraS."""
    from chebfunjax.operators.chebop import Chebop
    from chebfunjax.operators.chebop_altdisc import eigs_generalized_altdisc

    N = Chebop(lambda x, u: u.diff(2), domain=(0.0, np.pi))
    N.lbc = 0.0
    N.rbc = 0.0
    B = Chebop(lambda x, u: u, domain=(0.0, np.pi))
    for disc in ("ultraS", "chebcolloc1"):
        V, lam = eigs_generalized_altdisc(N, B, 3, 48, disc, sort="SM")
        lam = np.sort(np.abs(np.asarray(lam)))[:3]
        # eigenvalues of u'' = lam*u with Dirichlet: -k^2
        assert np.allclose(lam, [1.0, 4.0, 9.0], atol=1e-5)
