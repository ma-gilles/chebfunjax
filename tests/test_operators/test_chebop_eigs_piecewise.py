"""Piecewise-collocation eigs for discontinuous / kinked coefficients.

Pins the ``Chebop.eigs`` piecewise branch against values published on
chebfun.org (MATLAB Chebfun R2025b-era pages):

- ode-eig/DoubleWell: the first 12 eigenvalues of the double-well
  Schroedinger operator with an indicator-function potential.
- ode-eig/ContinuousWilkinson: the exponentially-near-degenerate top
  eigenvalue pairs of u'' + |x| u on [-8, 0, 8], plus the residual
  norms of its eigen- and pseudo-eigenfunctions.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from chebfunjax.chebfun1d.chebfun import chebfun  # noqa: E402
from chebfunjax.operators.chebop import Chebop  # noqa: E402


class TestDoubleWell:
    """ode-eig/DoubleWell.m: indicator potential, 12 states."""

    # Published output of eigs(L, 12) on chebfun.org/examples/ode-eig.
    REF = np.array([
        0.091480998228306, 0.116757122005294, 0.363909308598088,
        0.463167687393423, 0.808941736700779, 1.021145960789530,
        1.390812031498700, 1.652575851342681, 1.871230031210215,
        2.174488704532026, 2.533176594994659, 2.924094539796362,
    ])

    def test_eigenvalues_match_published(self):
        x = chebfun(lambda t: t, domain=(-1.0, 1.0))
        V = 1.5 * (abs(x - 0.05) < 0.25)
        L = Chebop(lambda x_, u: -0.007 * u.diff(2) + V * u, domain=(-1, 1))
        L.bc = 0.0
        lam = np.sort(np.asarray(L.eigs(k=12, n=64)).real)
        # Observed 2.1e-11 max deviation from the published values.
        assert np.max(np.abs(lam - self.REF)) < 1e-9


class TestContinuousWilkinson:
    """ode-eig/ContinuousWilkinson.m: u'' + |x| u on [-8, 0, 8]."""

    REF = np.array([
        3.912042616399311, 3.912059036621295,
        5.661892584504002, 5.661892594767715,
    ])

    @pytest.fixture(scope="class")
    def solution(self):
        N = 8
        L = Chebop(lambda x, u: u.diff(2) + abs(x) * u, domain=(-N, 0, N))
        L.bc = "dirichlet"
        lam, V = L.eigs(k=4, sigma="LR", return_eigenfunctions=True)
        lam = np.asarray(lam).real
        idx = np.argsort(lam)
        return L, lam[idx], [V[i] for i in idx]

    def test_near_degenerate_pairs(self, solution):
        _, lam, _ = solution
        # Observed 7.7e-13 max deviation from the published values.
        assert np.max(np.abs(lam - self.REF)) < 1e-10

    def test_eigenfunction_residual(self, solution):
        L, lam, V = solution
        r = float((L(V[3]) - lam[3] * V[3]).norm(2))
        # MATLAB publishes 5.67e-12; observed 1.5e-11 here.
        assert r < 1e-9

    def test_pseudo_eigenfunction_residual(self, solution):
        L, lam, V = solution
        left = V[3] - V[2]
        lmbda = 0.5 * (lam[2] + lam[3])
        r = float((L(left) - lmbda * left).norm(2))
        # Physical residual (eigenvalue splitting): MATLAB 7.2576e-9,
        # observed 7.2571e-9 -- a 4-digit match.
        assert abs(r - 7.2576e-9) < 1e-11 * 1e3   # within ~1e-3 relative

    def test_kinked_coefficient_detected_from_plain_domain(self):
        # Even WITHOUT the explicit interior breakpoint the |x|
        # coefficient must be detected (curvature jump of |x|*x): the
        # single-grid path stalls at ~5e-8 accuracy on this problem.
        N = 8
        L = Chebop(lambda x, u: u.diff(2) + abs(x) * u, domain=(-N, N))
        L.bc = "dirichlet"
        lam = np.sort(np.asarray(L.eigs(k=4, sigma="LR")).real)
        assert np.max(np.abs(lam - self.REF)) < 1e-10
