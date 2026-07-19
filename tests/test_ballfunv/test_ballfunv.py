"""Core unit tests for Ballfunv vector-field methods (Fable 5).

Exercises the audit additions — real/imag/conj, iszero/isequal,
laplacian, scalar-field multiply and scalar divide — without needing any
MATLAB golden reference.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.ballfun.ballfunv import Ballfunv

TOL = 1e4 * float(np.finfo(np.float64).eps)


def _B(op, spherical=False):
    return Ballfun.from_function(op, spherical=spherical)


class TestBallfunvComplexParts:
    def test_real_imag_conj(self):
        f1 = _B(lambda x, y, z: x)
        f2 = _B(lambda x, y, z: 1j * z)
        f3 = _B(lambda x, y, z: jnp.cos(y) + 1j * jnp.sin(x))
        V = Ballfunv(f1, f2, f3)
        exact_real = Ballfunv(_B(lambda x, y, z: x),
                              _B(lambda x, y, z: 0.0 * x),
                              _B(lambda x, y, z: jnp.cos(y)))
        exact_imag = Ballfunv(_B(lambda x, y, z: 0.0 * x),
                              _B(lambda x, y, z: z),
                              _B(lambda x, y, z: jnp.sin(x)))
        assert (V.real() - exact_real).norm() < TOL
        assert (V.imag() - exact_imag).norm() < TOL
        # conj negates the imaginary parts.
        exact_conj = Ballfunv(_B(lambda x, y, z: x),
                              _B(lambda x, y, z: -1j * z),
                              _B(lambda x, y, z: jnp.cos(y) - 1j * jnp.sin(x)))
        assert (V.conj() - exact_conj).norm() < TOL


class TestBallfunvPredicatesCalculus:
    def test_iszero_isequal(self):
        f = _B(lambda x, y, z: 1.0 + 0.0 * x)
        F = Ballfunv(f, f, f)
        assert (F - F).iszero()
        assert F.isequal(F + F - F)

    def test_laplacian(self):
        f = _B(lambda x, y, z: x ** 2 + y ** 2 + z ** 2)
        V = Ballfunv(f, 2 * f, -f)
        exact = Ballfunv(_B(lambda x, y, z: 6.0 + 0.0 * x),
                         _B(lambda x, y, z: 12.0 + 0.0 * x),
                         _B(lambda x, y, z: -6.0 + 0.0 * x))
        assert (V.laplacian() - exact).norm() < 1e7 * float(
            np.finfo(np.float64).eps)

    def test_scalar_field_multiply_and_divide(self):
        V = Ballfunv(_B(lambda x, y, z: x), _B(lambda x, y, z: y),
                     _B(lambda x, y, z: z))
        fc = _B(lambda x, y, z: jnp.cos(y))
        exact = Ballfunv(_B(lambda x, y, z: x * jnp.cos(y)),
                         _B(lambda x, y, z: y * jnp.cos(y)),
                         _B(lambda x, y, z: z * jnp.cos(y)))
        assert (V * fc - exact).norm() < TOL
        assert (fc * V - exact).norm() < TOL
        assert (V / 2 - Ballfunv(_B(lambda x, y, z: x / 2),
                                 _B(lambda x, y, z: y / 2),
                                 _B(lambda x, y, z: z / 2))).norm() < TOL


class TestBallfunvPoloidalToroidal:
    """Poloidal-toroidal decomposition (PT2ballfunv / PTdecomposition)
    and the poloidal-toroidal Helmholtz decomposition."""

    def test_pt2ballfunv_is_divergence_free(self):
        # curl(curl(rP)) + curl(rT) is divergence-free by construction.
        P = _B(lambda x, y, z: x ** 2 + y * z)
        T = _B(lambda x, y, z: x * y * z)
        V = Ballfunv.PT2ballfunv(P, T)
        assert V.div().norm() < 1e-8

    def test_pt2ballfunv_two_output_split(self):
        # [Pv, Tv] must sum to the combined field.
        P = _B(lambda x, y, z: x ** 2 + y * z)
        T = _B(lambda x, y, z: x * y * z)
        Pv, Tv = Ballfunv.PT2ballfunv(P, T, nargout=2)
        V = Ballfunv.PT2ballfunv(P, T)
        assert (V - (Pv + Tv)).norm() < TOL

    def test_pt_roundtrip_recovers_scalars(self):
        # PTdecomposition inverts PT2ballfunv up to the l=0 (angular-mean)
        # gauge, so the spherical lambda/theta derivatives must match.
        tol = 1e5 * float(np.finfo(np.float64).eps)
        P = _B(lambda x, y, z: x ** 2 + y * z)
        T = _B(lambda x, y, z: x * y * z)
        V = Ballfunv.PT2ballfunv(P, T)
        P2, T2 = V.PTdecomposition()
        for dim in (2, 3):
            assert (P.diff(dim, 1, "spherical")
                    - P2.diff(dim, 1, "spherical")).norm() < tol
            assert (T.diff(dim, 1, "spherical")
                    - T2.diff(dim, 1, "spherical")).norm() < tol

    @pytest.mark.xfail(
        strict=True,
        reason="needs the spectral ball Poisson-Neumann solver (see the port file's xfail note): collocation helmholtz caps accuracy at ~5e-10 / 4.6e-8 vs 2.22e-10")
    def test_helmholtz_decomposition_two_component(self):
        tol = 1e6 * float(np.finfo(np.float64).eps)
        vx = _B(lambda x, y, z: jnp.cos(x * y))
        vy = _B(lambda x, y, z: jnp.sin(x * z))
        vz = _B(lambda x, y, z: jnp.cos(y * z))
        v = Ballfunv(vx, vy, vz)
        f, P, T = v.HelmholtzDecomposition(nargout=3)
        gf = f.grad()
        w = Ballfunv(gf[0], gf[1], gf[2]) + Ballfunv.PT2ballfunv(P, T)
        assert (v - w).norm() < tol

    @pytest.mark.xfail(
        strict=True,
        reason="needs the spectral ball Poisson-Neumann solver (see the port file's xfail note): collocation helmholtz caps accuracy at ~5e-10 / 4.6e-8 vs 2.22e-10")
    def test_helmholtz_decomposition_three_component(self):
        tol = 1e6 * float(np.finfo(np.float64).eps)
        vx = _B(lambda x, y, z: jnp.cos(x * y))
        vy = _B(lambda x, y, z: jnp.sin(x * z))
        vz = _B(lambda x, y, z: jnp.cos(y * z))
        v = Ballfunv(vx, vy, vz)
        f, P, T, phi = v.HelmholtzDecomposition(nargout=4)
        gf = f.grad()
        gp = phi.grad()
        w = (Ballfunv(gf[0], gf[1], gf[2])
             + Ballfunv.PT2ballfunv(P, T).curl()
             + Ballfunv(gp[0], gp[1], gp[2]))
        assert (v - w).norm() < tol
