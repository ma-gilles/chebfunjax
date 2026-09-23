"""Helmholtz-Hodge decomposition of a vector field.

Faithful replica of sphere/HelmholtzDecomposition.m by Alex Townsend
and Grady Wright (May 2016): any tangent vector field on the sphere
splits uniquely as f = grad(phi) + curl(psi), where phi and psi solve
Poisson equations with div(f) and vorticity(f) as right-hand sides.

The surface calculus here runs on the double-Fourier-sphere grid with
exact FFT spectral differentiation in numpy: the library's
spherefun vorticity/divergence path cannot compile the rank-~100
tangent field (XLA JIT kernel-size defect, recorded in the audit
ledger), so this replica performs the same mathematics -- DFS
sampling, spectral derivatives, spherical-harmonic Poisson solves --
at machine precision without JIT.

Original: https://www.chebfun.org/examples/sphere/HelmholtzDecomposition.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from numpy.polynomial.legendre import leggauss
from scipy.special import gammaln, lpmv

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'sphere')

N_LAM, N_TH = 256, 128        # DFS grid (theta doubled to 2*N_TH)
LMAX, NQ = 90, 160


def ylm_np(l, m, lam, th):
    ma = abs(m)
    a = np.sqrt((2 * l + 1) / (4 * np.pi)
                * np.exp(gammaln(l - ma + 1) - gammaln(l + ma + 1)))
    P = lpmv(ma, l, np.cos(th))     # lpmv carries the CS phase
    if m > 0:
        return np.sqrt(2) * a * P * np.cos(ma * lam)
    if m < 0:
        return np.sqrt(2) * a * P * np.sin(ma * lam)
    return a * P


def field(x, y, z):
    """The example's Cartesian field F(x, y, z)."""
    return (y * z * np.cos(x * y * z),
            x * z * np.sin(4 * x + .1 * y + 5 * z**2),
            1 + x * y * z)


def tangent_sph(lam, th):
    """Spherical components (f_lam, f_th) of tangent(F) at (lam, th)."""
    x = np.cos(lam) * np.sin(th)
    y = np.sin(lam) * np.sin(th)
    z = np.cos(th)
    Fx, Fy, Fz = field(x, y, z)
    # unit vectors
    lx, ly = -np.sin(lam), np.cos(lam)
    tx, ty, tz = (np.cos(lam) * np.cos(th), np.sin(lam) * np.cos(th),
                  -np.sin(th))
    f_lam = Fx * lx + Fy * ly
    f_th = Fx * tx + Fy * ty + Fz * tz
    return f_lam, f_th


def dfs_derivs(comp_fn):
    """Exact spectral d/dlam and d/dth of a scalar DFS-extendable
    sample of ``comp_fn(lam, th)`` with the VECTOR-component parity
    (both spherical components flip sign under the glide reflection
    (lam, th) -> (lam + pi, 2 pi - th))."""
    lam = -np.pi + 2 * np.pi * np.arange(N_LAM) / N_LAM
    th = np.pi * (np.arange(N_TH) + 0.5) / N_TH      # interior nodes
    L, T = np.meshgrid(lam, th, indexing="ij")
    V = comp_fn(L, T)
    # glide-reflected half: theta in (pi, 2 pi)
    V2 = np.empty((N_LAM, 2 * N_TH))
    V2[:, :N_TH] = V
    V2[:, N_TH:] = -np.roll(V[:, ::-1], N_LAM // 2, axis=0)
    kl = np.fft.fftfreq(N_LAM, d=1.0 / N_LAM)
    kt = np.fft.fftfreq(2 * N_TH, d=1.0 / (2 * N_TH))
    C = np.fft.fft2(V2)
    dV_dlam = np.real(np.fft.ifft2(1j * kl[:, None] * C))[:, :N_TH]
    dV_dth = np.real(np.fft.ifft2(1j * kt[None, :] * C))[:, :N_TH]
    return lam, th, V, dV_dlam, dV_dth


def dfs_eval_grid(Vhalf, sign, LQ, TQ):
    """Evaluate the DFS trig interpolant of a half-grid sample at
    arbitrary (LQ, TQ) points.  ``sign`` is the glide parity."""
    V2 = np.empty((N_LAM, 2 * N_TH))
    V2[:, :N_TH] = Vhalf
    V2[:, N_TH:] = sign * np.roll(Vhalf[:, ::-1], N_LAM // 2, axis=0)
    C = np.fft.fft2(V2) / (N_LAM * 2 * N_TH)
    kl = np.fft.fftfreq(N_LAM, d=1.0 / N_LAM)
    kt = np.fft.fftfreq(2 * N_TH, d=1.0 / (2 * N_TH))
    th0 = np.pi * 0.5 / N_TH
    El = np.exp(1j * np.outer(LQ.ravel() + np.pi, kl))
    Et = np.exp(1j * np.outer(TQ.ravel() - th0, kt))
    return np.real(np.einsum("pk,kl,pl->p", El, C, Et)).reshape(LQ.shape)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # Surface divergence and vorticity by exact DFS differentiation.
    lam, th, FL, dFL_dl, dFL_dt = dfs_derivs(
        lambda L, T: tangent_sph(L, T)[0])
    _, _, FT, dFT_dl, dFT_dt = dfs_derivs(
        lambda L, T: tangent_sph(L, T)[1])
    T = np.meshgrid(lam, th, indexing="ij")[1]
    sinT, cosT = np.sin(T), np.cos(T)
    DIV = (dFL_dl + cosT * FT + sinT * dFT_dt) / sinT
    VORT = (cosT * FL + sinT * dFL_dt - dFT_dl) / sinT

    # Project onto spherical harmonics on the GL grid; solve Poisson.
    xg, wg = leggauss(NQ)
    thq = np.arccos(xg)
    lamq = -np.pi + 2 * np.pi * np.arange(2 * NQ) / (2 * NQ)
    LQ, TQ = np.meshgrid(lamq, thq)
    wl = 2 * np.pi / (2 * NQ)
    DVQ = dfs_eval_grid(DIV * sinT, -1, LQ, TQ) / np.sin(TQ)
    VVQ = dfs_eval_grid(VORT * sinT, -1, LQ, TQ) / np.sin(TQ)

    def solve_poisson(F):
        co = {}
        for l in range(1, LMAX + 1):
            for m in range(-l, l + 1):
                Y = ylm_np(l, m, LQ, TQ)
                c = float(np.sum(F * Y * wg[:, None]) * wl)
                if abs(c) > 1e-14:
                    co[(l, m)] = c / (-l * (l + 1))
        return co

    cphi = solve_poisson(DVQ)
    cpsi = solve_poisson(VVQ)

    def pot_eval(co, L, T):
        out = np.zeros(np.shape(L))
        for (l, m), c in co.items():
            out = out + c * ylm_np(l, m, L, T)
        return out

    # grad(phi) and curl(psi) in spherical components via DFS derivs.
    _, _, PH, dPH_dl, dPH_dt = (lambda r: r)(dfs_derivs_scalar(cphi))
    _, _, PS, dPS_dl, dPS_dt = (lambda r: r)(dfs_derivs_scalar(cpsi))
    GP_L, GP_T = dPH_dl / sinT, dPH_dt                 # grad phi
    CP_L, CP_T = dPS_dt, -dPS_dl / sinT                # n x grad psi

    # Published identity 1: vorticity of the curl-free part.
    _, _, _, dGPL_dl, dGPL_dt = dfs_derivs_grid(GP_L)
    _, _, _, dGPT_dl, _ = dfs_derivs_grid(GP_T)
    V1 = (cosT * GP_L + sinT * dGPL_dt - dGPT_dl) / sinT
    print("ans =")
    print(f"     {l2_norm(V1):.15e}")

    # Published identity 2: divergence of the divergence-free part.
    _, _, _, dCPL_dl, _ = dfs_derivs_grid(CP_L)
    _, _, _, dCPT_dl, dCPT_dt = dfs_derivs_grid(CP_T)
    D2 = (dCPL_dl + cosT * CP_T + sinT * dCPT_dt) / sinT
    print("ans =")
    print(f"     {l2_norm(D2):.15e}")

    # Published identity 3: the decomposition reproduces f.
    R_L = FL - GP_L - CP_L
    R_T = FT - GP_T - CP_T
    print("ans =")
    print(f"     {l2_norm(np.sqrt(R_L**2 + R_T**2)):.15e}")

    # Decomposition panel (quiver of the three fields).
    _quiver_panel([(GP_L, GP_T, "Curl-free"),
                   (CP_L, CP_T, "Divergence-free"),
                   (FL, FT, "Tangent vector field")], lam, th)


def dfs_derivs_scalar(co):
    """dfs_derivs of a harmonic-sum scalar (EVEN glide parity)."""
    lam = -np.pi + 2 * np.pi * np.arange(N_LAM) / N_LAM
    th = np.pi * (np.arange(N_TH) + 0.5) / N_TH
    L, T = np.meshgrid(lam, th, indexing="ij")
    V = np.zeros_like(L)
    for (l, m), c in co.items():
        V = V + c * ylm_np(l, m, L, T)
    return _spec_diff(V, +1)


def dfs_derivs_grid(Vhalf):
    """dfs_derivs of an already-sampled VECTOR component (odd parity)."""
    return _spec_diff(Vhalf, -1)


def _spec_diff(V, sign):
    V2 = np.empty((N_LAM, 2 * N_TH))
    V2[:, :N_TH] = V
    V2[:, N_TH:] = sign * np.roll(V[:, ::-1], N_LAM // 2, axis=0)
    kl = np.fft.fftfreq(N_LAM, d=1.0 / N_LAM)
    kt = np.fft.fftfreq(2 * N_TH, d=1.0 / (2 * N_TH))
    C = np.fft.fft2(V2)
    dV_dlam = np.real(np.fft.ifft2(1j * kl[:, None] * C))[:, :N_TH]
    dV_dth = np.real(np.fft.ifft2(1j * kt[None, :] * C))[:, :N_TH]
    lam = -np.pi + 2 * np.pi * np.arange(N_LAM) / N_LAM
    th = np.pi * (np.arange(N_TH) + 0.5) / N_TH
    return lam, th, V, dV_dlam, dV_dth


def l2_norm(V):
    """Sphere L2 norm of a half-grid sample (uniform-theta weights via
    the sin(theta) area element and midpoint rule -- spectrally
    accurate for DFS-band-limited integrands)."""
    th = np.pi * (np.arange(N_TH) + 0.5) / N_TH
    w = np.sin(th) * (np.pi / N_TH) * (2 * np.pi / N_LAM)
    return float(np.sqrt(np.sum(V**2 * w[None, :])))


def _quiver_panel(fields, lam, th):
    from chebfunjax.plotting import chebfun_style
    chebfun_style()
    fig = plt.figure(figsize=(13.5, 4.8))
    sl = slice(0, N_LAM, 10)
    st = slice(4, N_TH - 4, 6)
    L, T = np.meshgrid(lam[sl], th[st], indexing="ij")
    X = np.cos(L) * np.sin(T)
    Y = np.sin(L) * np.sin(T)
    Z = np.cos(T)
    lx, ly = -np.sin(L), np.cos(L)
    tx, ty, tz = (np.cos(L) * np.cos(T), np.sin(L) * np.cos(T),
                  -np.sin(T))
    for i, (VL, VT, ttl) in enumerate(fields):
        vl, vt = VL[sl, st], VT[sl, st]
        U = vl * lx + vt * tx
        V = vl * ly + vt * ty
        W = vt * tz
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.quiver(X, Y, Z, U, V, W, length=0.25, lw=0.6,
                  color="tab:blue")
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.view_init(8, -36)
        ax.set_title(ttl)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, "HelmholtzDecomposition_repl_01.png"),
        dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
