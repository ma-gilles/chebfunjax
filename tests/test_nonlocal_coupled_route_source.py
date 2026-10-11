"""Exact-block and routing diagnostics, not native test predicates."""

from unittest.mock import patch

import numpy as np

from chebfunjax import chebfun
from chebfunjax.operators._coupled_ivp import classify_local_operator
from chebfunjax.operators._coupled_newton import fit_bcs, linearize
from chebfunjax.operators.chebop import Chebop


def source_operator():
    N = Chebop(
        lambda t, u, v: [u.diff() - v, v.diff() + v((1 - v).exp()) * v**2 * (1 - v).exp()],
        domain=(1.0, 5.0),
    )
    N.lbc = [0.0, 1.0]
    return N


def test_actual_ad_blocks_and_fit_bcs():
    N = source_operator()
    L, _, flags = linearize(N)
    assert len(L.A.blocks) == 2 and len(L.A.blocks[0]) == 2
    initial = fit_bcs(L)
    pts = np.array([1.0, 2.0, 3.0, 5.0])
    np.testing.assert_allclose(initial[0](pts), 0.0, atol=32 * np.finfo(float).eps, rtol=0)
    np.testing.assert_allclose(initial[1](pts), 1.0, atol=32 * np.finfo(float).eps, rtol=0)
    L, _, flags = linearize(N, initial)
    h = chebfun(lambda t: t * t, domain=(1.0, 5.0))
    # At v=1 the literal composition chain rule yields h'+h(t)+h(1).
    actual = L.A.blocks[1][1].apply(h)
    np.testing.assert_allclose(
        actual(pts), 2 * pts + pts**2 + 1, atol=256 * np.finfo(float).eps, rtol=0
    )


def test_positive_nonlocal_routes_existing_solver():
    N = source_operator()
    sentinel = object()
    with patch(
        "chebfunjax.operators._coupled_newton.solve_coupled", return_value=sentinel
    ) as solve:
        assert N._solve_impl(0.0) is sentinel
        assert solve.call_count == 1
        assert solve.call_args.kwargs == dict(
            n=None, max_iter=25, bvp_tol=5e-13, n_min=32, n_max=4096
        )
        assert N.init is None


def test_local_and_unknown_do_not_select_nonlocal_route():
    local = Chebop(lambda t, u, v: [u.diff() - v, v.diff() + v], domain=(0.0, 1.0))
    local.lbc = [0.0, 1.0]
    unknown = Chebop(lambda t, u, v: [u.diff() - v, v.diff() + v.sqrt()], domain=(1.0, 5.0))
    unknown.lbc = [0.0, 1.0]
    assert classify_local_operator(local) is True
    assert classify_local_operator(unknown) is None
    with patch(
        "chebfunjax.operators._coupled_newton.solve_coupled",
        side_effect=AssertionError("wrong route"),
    ):
        u, v = local._solve_impl(0.0)
        np.testing.assert_allclose(
            v(np.array([0.0, 0.5, 1.0])), np.exp(-np.array([0.0, 0.5, 1.0])), atol=1e-10, rtol=0
        )
