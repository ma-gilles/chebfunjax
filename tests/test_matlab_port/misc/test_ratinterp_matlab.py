"""Port of MATLAB Chebfun tests/misc/test_ratinterp.m (Fable 5).

Ports the type0 (roots-of-unity) case together with the type1 (1st-kind
Chebyshev) and type2 (2nd-kind Chebyshev) grid variants; all recover the
same type-(4, 2) approximant with poles at -0.2 and 2.2.

Provenance
----------
MATLAB source : tests/misc/test_ratinterp.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.utils.quadrature import chebpts
from chebfunjax.utils.ratapprox import ratinterp

TOL = 1e-10


def f(x):
    return (x ** 4 - 3) / ((x + 0.2) * (x - 2.2))


def _mapab(x, a, b):
    return a + (b - a) * (x + 1) / 2


class TestRatinterp:
    def test_degrees_and_poles(self):
        p, q, r, mu, nu, poles, res = ratinterp(f, 10, 10,
                                                domain=(0.0, 2.0))
        assert mu == 4 and nu == 2
        pl = np.sort_complex(np.asarray(poles))
        assert float(np.max(np.abs(pl - np.array([-0.2, 2.2])))) < TOL

    def test_grid_type_variants(self):
        # MATLAB pass(2)/pass(3): the type1 (1st-kind Chebyshev) and type2
        # (2nd-kind Chebyshev) grids both yield the type-(4, 2) approximant
        # with poles at -0.2 and 2.2.
        for grid in ("type0", "type1", "type2", "equi"):
            p, q, r, mu, nu, poles, res = ratinterp(f, 10, 10,
                                                    domain=(0.0, 2.0), xi=grid)
            assert mu == 4 and nu == 2
            pl = np.sort_complex(np.asarray(poles))
            assert float(np.max(np.abs(pl - np.array([-0.2, 2.2])))) < TOL

    def test_data_vector_reduction(self):
        # MATLAB pass(9)/(10)/(11): a length-N=100 vector of samples on
        # 1st-/2nd-kind Chebyshev and equispaced grids still reduces the
        # requested type-(10, 10) to the exact type-(4, 2).
        N = 100
        x0 = 1 + np.exp(2j * np.pi * np.arange(N) / N)
        _, _, _, mu, nu, poles, _ = ratinterp(f(x0), 10, 10, N, "type0",
                                              domain=(0.0, 2.0))
        assert mu == 4 and nu == 2
        np.testing.assert_allclose(np.sort_complex(poles), [-0.2, 2.2],
                                   atol=TOL, rtol=0)
        x1 = _mapab(np.asarray(chebpts(N, kind=1)), 0, 2)
        _, _, _, mu, nu, _, _ = ratinterp(f(x1), 10, 10, N, "type1",
                                          domain=(0.0, 2.0))
        assert mu == 4 and nu == 2
        x2 = _mapab(np.asarray(chebpts(N, kind=2)), 0, 2)
        _, _, _, mu, nu, _, _ = ratinterp(f(x2), 10, 10, N, "type2",
                                          domain=(0.0, 2.0))
        assert mu == 4 and nu == 2
        xe = np.linspace(0, 2, N)
        _, _, _, mu, nu, _, _ = ratinterp(f(xe), 10, 10, N, "equi",
                                          domain=(0.0, 2.0))
        assert mu == 4 and nu == 2

    def test_simple_pole(self):
        # MATLAB pass(12): 1/(x - 0.2) requested (10, 10) reduces to type
        # (0, 1) with the single pole at 0.2.
        _, _, _, mu, nu, poles, _ = ratinterp(lambda x: 1.0 / (x - 0.2),
                                              10, 10, xi="type2")
        assert mu == 0 and nu == 1
        assert abs(float(np.real(poles[0])) - 0.2) < 1e-10

    def test_arbitrary_complex_nodes(self):
        # MATLAB pass(19): complex nodes supplied without TYPE0 metadata.
        nodes = np.exp(2j * np.pi * np.arange(32) / 32)
        _, _, _, _, _, poles, _ = ratinterp(
            lambda x: np.sin(x) / (x - 0.1), 30, 1, xi=nodes)
        np.testing.assert_allclose(poles, [0.1], atol=TOL, rtol=0)

    def test_polynomial_robustification(self):
        # MATLAB pass(18): the remote pole reduces to a polynomial.
        _, _, _, _, nu, _, _ = ratinterp(
            lambda x: 1 / (x - 1.7), 128, 1, xi="type0", tol=1e-14)
        assert nu == 0

    def test_constant_at_arbitrary_nodes(self):
        # MATLAB pass(23): reduced constant denominator on arbitrary points.
        r, _, _, _, _, _, _ = ratinterp(
            np.ones(5), 1, 1, xi=np.asarray(chebpts(5)))
        assert abs(r(1) - 1) < TOL

    def test_no_robustification(self):
        # MATLAB pass(21)-(22), retained without widening its tolerance.
        r, _, _, _, _, _, _ = ratinterp(
            lambda x: 1 / (x - 0.2), 10, 10, xi="type2", tol=0)
        x = np.linspace(-1, 1, 20)
        assert np.linalg.norm(r(x) - 1 / (x - 0.2)) < TOL

    @pytest.mark.parametrize('grid', ['type1', 'type2', 'equi'])
    def test_chebfun_input(self, grid):
        # MATLAB pass(5)-(7): use a chebfun instead of an anonymous function.
        cf = chebfun(f, domain=(0.0, 2.0))
        _, _, _, mu, nu, poles, _ = ratinterp(cf, 10, 10, xi=grid, domain=(0.0, 2.0))
        assert (mu, nu) == (4, 2)
        np.testing.assert_allclose(np.sort_complex(poles), [-0.2, 2.2], atol=TOL, rtol=0)

    @pytest.mark.parametrize('arbitrary', [False, True])
    def test_nonsmooth_approximation(self, arbitrary):
        # MATLAB pass(13)-(17): polynomial sizes, full norm and 300-point check.
        cf = abs(chebfun(np.exp, domain=(1.0, 3.0)) - 5)
        nodes = _mapab(np.asarray(chebpts(6)), 1.0, 3.0) if arbitrary else 'type2'
        r, a, b, _, _, _, _ = ratinterp(cf, 2, 3, xi=nodes, domain=(1.0, 3.0))
        if arbitrary:
            assert len(a) == 3 and len(b) == 4
        approx = chebfun(r, domain=(1.0, 3.0))
        assert float((cf - approx).norm(np.inf)) < 0.6
        x = np.linspace(1.0, 3.0, 300)
        assert np.max(np.abs(cf(x) - r(x))) < 0.6

    @pytest.mark.parametrize("grid", ["type0", "type1", "type2", "equi"])
    def test_complex_poles_and_evaluation(self, grid):
        # MATLAB ratinterp.m returns complex poles and permits complex
        # evaluation of its rational function handle.
        z, w = 0.2 + 0.3j, 1.5 - 0.4j

        def f(x):
            return (1 + 2j) / ((x - z) * (x - w))

        r, _, _, mu, nu, poles, _ = ratinterp(f, 2, 2, xi=grid)
        assert (mu, nu) == (0, 2)
        np.testing.assert_allclose(np.sort_complex(poles),
                                   np.sort_complex([z, w]), atol=1e-10, rtol=0)
        x = np.array([0.1 + 0.2j, -0.5 + 0.1j])
        np.testing.assert_allclose(r(x), f(x), atol=1e-11, rtol=0)

    @pytest.mark.parametrize("grid", ["type0", "type1", "type2", "equi"])
    @pytest.mark.parametrize("domain", [(-1.0, 1.0), (2.0, 6.0)])
    def test_complex_simple_residues(self, grid, domain):
        # ratinterp.m delegates to residue(p, q); these independent analytic
        # residues check complex data and the reference-to-physical scaling.
        mid, half = sum(domain) / 2, (domain[1] - domain[0]) / 2
        z, w = mid + half * (0.2 + 0.3j), mid + half * (1.5 - 0.4j)
        amplitude = 1 + 2j
        _, _, _, _, _, poles, residues = ratinterp(
            lambda x: amplitude / ((x - z) * (x - w)), 2, 2,
            xi=grid, domain=domain)
        expected = amplitude / (z - w)
        for pole, residue in zip(poles, residues, strict=True):
            target = expected if abs(pole - z) < abs(pole - w) else -expected
            np.testing.assert_allclose(residue, target, atol=1e-10, rtol=0)
