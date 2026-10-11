"""Portable nonlinear-finalization contracts; saved DDE science is external."""

from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators.chebop import Chebop, SystemSolution, _SourceNonlinearSystemSolution


def polynomial_solution():
    # Exact dyadic coefficients and explicit point values exercise storage
    # independently of any adaptive solve or machine-specific error receipt.
    f = Chebfun.from_coeffs(jnp.array([1.0, 2.0, 3.0]), domain=(1.0, 5.0)).set_point_values(
        jnp.array([7.0, 11.0])
    )
    g = Chebfun.from_coeffs(
        jnp.array([0.5, 0.25, 0.125, 0.0625, 0.03125]), domain=(1.0, 5.0)
    ).set_point_values(jnp.array([13.0, 17.0]))
    return _SourceNonlinearSystemSolution(v.simplify(tol=ChebopPref().bvpTol) for v in (f, g))


def test_independent_finalization_and_metadata():
    out = polynomial_solution()
    assert [f.funs[0].n for f in out] == [3, 5]
    assert Chebop._simplify_solution(out) is out
    assert isinstance(out, SystemSolution)
    assert out.blocks == list(out) and len(out.deal()) == 2
    np.testing.assert_array_equal(out[0].point_values, jnp.array([7.0, 11.0]))
    np.testing.assert_array_equal(out[1].point_values, jnp.array([13.0, 17.0]))


def test_public_dispatch_preserves_finalized_tag():
    out = polynomial_solution()
    N = Chebop(
        lambda t, u, v: [u.diff() - v, v.diff() + v((1 - v).exp()) * v**2 * (1 - v).exp()],
        domain=(1.0, 5.0),
    )
    N.lbc = [0.0, 1.0]
    with patch("chebfunjax.operators._coupled_newton.solve_coupled", return_value=out) as solve:
        result = N.solve(0.0)
        assert result is out and isinstance(result, SystemSolution)
        assert [f.funs[0].n for f in result] == [3, 5]
        assert solve.call_count == 1 and N.init is None
        assert solve.call_args.kwargs["bvp_tol"] == ChebopPref().bvpTol


def test_linear_and_legacy_finalization_unchanged():
    out = polynomial_solution()
    points = jnp.array([1.0, 2.0, 3.0, 5.0])
    for container in (list, tuple, SystemSolution):
        result = Chebop._simplify_solution(container(out))
        assert type(result) is container
        assert [f.funs[0].n for f in result] == [5, 5]
        for f, g in zip(result, out):
            np.testing.assert_array_equal(f.point_values, g.point_values)
            np.testing.assert_array_equal(f(points), g(points))
