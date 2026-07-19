# uses-numpy: the poloidal-toroidal subsystem solves dense per-Fourier-mode
# linear systems (surface-Laplacian inversion) and applies banded spectral
# multiplication operators — non-JIT host-side linear algebra transcribed
# directly from the MATLAB @ballfunv sources.
"""Poloidal-toroidal decomposition machinery for :class:`Ballfunv`.

A divergence-free vector field ``V`` on the unit ball admits the
decomposition ``V = curl(curl(r*P)) + curl(r*T)`` where ``P`` (poloidal)
and ``T`` (toroidal) are scalar :class:`Ballfun` fields and ``r`` is the
radial position vector.  ``P`` and ``T`` are recovered from ``r.V`` and
``r.curl(V)`` by inverting the surface Laplacian on each Fourier mode.

Provenance
----------
MATLAB source : @ballfunv/PTdecomposition.m, @ballfunv/PT2ballfunv.m,
                @ballfunv/HelmholtzDecomposition.m
Chebfun commit: 7574c77
Original authors: Copyright 2018-2019 by The University of Oxford and
    The Chebfun Developers.  See http://www.chebfun.org/ .
"""

from __future__ import annotations

import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun, _fourier_wavenumbers_1d

# ----------------------------------------------------------------------
# Spectral helpers
# ----------------------------------------------------------------------

def _fourier_multmat(n: int, vec) -> np.ndarray:
    """Fourier (trigspec) multiplication matrix.

    ``vec`` is a length ``2K+1`` array of centred trigonometric
    coefficients ``[c_{-K}, ..., c_0, ..., c_K]`` (MATLAB
    ``trigspec.multmat`` convention).  The returned ``n x n`` Toeplitz
    matrix satisfies ``M[out, in] = c_{out-in}`` in fftshift ordering.
    """
    vec = np.asarray(vec, dtype=np.complex128)
    L = len(vec)
    K = L // 2
    M = np.zeros((n, n), dtype=np.complex128)
    for dk in range(-K, K + 1):
        c = vec[K + dk]
        if c == 0:
            continue
        if dk >= 0:
            rows = np.arange(dk, n)
            cols = np.arange(0, n - dk)
        else:
            rows = np.arange(0, n + dk)
            cols = np.arange(-dk, n)
        M[rows, cols] = c
    return M


def _fourier_diffmat(n: int, k: int) -> np.ndarray:
    """Fourier (trigspec) k-th derivative matrix = diag((i*wavenumber)^k)."""
    w = _fourier_wavenumbers_1d(n)
    return np.diag((1j * w) ** k)


def _fourier_df1(n: int) -> np.ndarray:
    """First-derivative Fourier diagonal ``1i*diag(wavenumbers)``."""
    return np.diag(1j * _fourier_wavenumbers_1d(n))


def _resize_coeffs3(F: np.ndarray, m: int, n: int, p: int) -> np.ndarray:
    """Resize a CFF coefficient tensor to ``(m, n, p)``.

    Chebyshev (axis 0) is truncated/zero-padded at the tail; the two
    Fourier axes are truncated/zero-padded symmetrically about the DC
    mode (index ``//2``).  For the enlarging case used throughout the PT
    subsystem this is exactly MATLAB ``coeffs3`` (alias to a larger grid
    is a centred zero-pad).
    """
    F = np.asarray(F, dtype=np.complex128)
    mf, nf, pf = F.shape
    out = np.zeros((m, n, p), dtype=np.complex128)

    mc = min(m, mf)

    def _fourier_slices(dst, src):
        # centred placement: DC (index src//2) maps to DC (index dst//2)
        lo = min(src // 2, dst // 2)
        hi = min(src - src // 2, dst - dst // 2)
        d0 = dst // 2 - lo
        s0 = src // 2 - lo
        length = lo + hi
        return slice(d0, d0 + length), slice(s0, s0 + length)

    dn, sn = _fourier_slices(n, nf)
    dp, sp = _fourier_slices(p, pf)
    out[:mc, dn, dp] = F[:mc, sn, sp]
    return out


def _coeffs3(f: Ballfun, m: int, n: int, p: int) -> np.ndarray:
    return _resize_coeffs3(np.asarray(f.coeffs, dtype=np.complex128), m, n, p)


def _rdiv(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Right matrix division ``X / B == X @ inv(B)`` via a solve."""
    return np.linalg.solve(B.T, X.T).T


# ----------------------------------------------------------------------
# PT2ballfunv : inverse of the poloidal-toroidal decomposition
# ----------------------------------------------------------------------

def _rcurl(P: Ballfun):
    """Return ``curl(r*P)`` as a :class:`Ballfunv` (MATLAB ``rcurl``)."""
    from chebfunjax.ballfun.ballfunv import Ballfunv

    if P.isempty():
        return Ballfunv.empty()

    m, n, p = P.shape

    # If p is odd, make it even; the variable-coefficient chain rule adds
    # one wavenumber in lambda and theta.
    n_tilde = n + 2
    p_tilde = p + (p % 2) + 2

    Pexp = _coeffs3(P, m, n_tilde, p_tilde)

    MsinL = _fourier_multmat(n_tilde, [0.5j, 0.0, -0.5j])
    McosL = _fourier_multmat(n_tilde, [0.5, 0.0, 0.5])
    MsinT = _fourier_multmat(p_tilde, [0.5j, 0.0, -0.5j])
    McosT = _fourier_multmat(p_tilde, [0.5, 0.0, 0.5])

    k_lam = _fourier_wavenumbers_1d(n_tilde)
    k_th = _fourier_wavenumbers_1d(p_tilde)

    Pl = Pexp * (1j * k_lam)[None, :, None]    # dP/dlam
    Pt = Pexp * (1j * k_th)[None, None, :]     # dP/dth

    # permute [2,3,1] -> (n_tilde, p_tilde, m)
    Pl = np.transpose(Pl, (1, 2, 0))
    Pt = np.transpose(Pt, (1, 2, 0))

    Vx = np.zeros((n_tilde, p_tilde, m), dtype=np.complex128)
    Vy = np.zeros((n_tilde, p_tilde, m), dtype=np.complex128)
    Vz = -Pl

    McosT_T = McosT.T
    for k in range(m):
        Plk = Pl[:, :, k]
        Ptk = Pt[:, :, k]
        rd_l = _rdiv(McosL @ Plk @ McosT_T, MsinT.T)
        Vx[:, :, k] = rd_l + MsinL @ Ptk
        Vy[:, :, k] = _rdiv(MsinL @ Plk @ McosT_T, MsinT.T) - McosL @ Ptk

    # permute back [3,1,2] -> (m, n_tilde, p_tilde)
    Vx = np.transpose(Vx, (2, 0, 1))
    Vy = np.transpose(Vy, (2, 0, 1))
    Vz = np.transpose(Vz, (2, 0, 1))

    return Ballfunv(
        Ballfun.from_coeffs(Vx, is_real=P.is_real),
        Ballfun.from_coeffs(Vy, is_real=P.is_real),
        Ballfun.from_coeffs(Vz, is_real=P.is_real),
    )


def pt2ballfunv(P: Ballfun, T: Ballfun, nargout: int = 1):
    """Inverse PT decomposition ``V = curl(curl(r*P)) + curl(r*T)``.

    With ``nargout == 2`` returns the poloidal and toroidal vector fields
    ``(Pv, Tv)`` separately.
    """
    Pv = _rcurl(P).curl()   # curl(curl(r*P))
    Tv = _rcurl(T)          # curl(r*T)
    if nargout >= 2:
        return Pv, Tv
    return Pv + Tv


# ----------------------------------------------------------------------
# PTdecomposition : forward decomposition
# ----------------------------------------------------------------------

def _ptequation(Fcoeffs: np.ndarray, is_real: bool) -> Ballfun:
    """Solve ``sin^2(theta) Delta_S u = f`` per Fourier mode.

    The 0-th Fourier mode in lambda and theta of ``u`` is pinned to zero
    (the null space of the surface Laplacian).  ``Fcoeffs`` is the CFF
    tensor of the right-hand side.
    """
    m, n, p = Fcoeffs.shape

    # variables reordered to (theta, r, lambda): permute [3,1,2]
    F = np.transpose(Fcoeffs, (2, 0, 1))

    Msin2 = _fourier_multmat(p, [-0.25, 0.0, 0.5, 0.0, -0.25])
    DF1 = _fourier_df1(p)
    DF2 = _fourier_diffmat(p, 2)
    Mcossin = _fourier_multmat(p, [0.25j, 0.0, 0.0, 0.0, -0.25j])
    Ip = np.eye(p, dtype=np.complex128)
    DF2lam = _fourier_diffmat(n, 2)

    Lth = Msin2 @ DF2 + Mcossin @ DF1

    CFS = np.zeros((p, m, n), dtype=np.complex128)

    p_dc = p // 2
    n_dc = n // 2

    for k in range(n):
        ff = F[:, :, k]
        if np.max(np.abs(ff)) <= 1e-16:
            continue

        Lthlam = Lth + DF2lam[k, k] * Ip
        ff = ff.copy()

        if k == n_dc:
            # pin the 0-th theta mode of the DC lambda mode
            Lthlam = Lthlam.copy()
            Lthlam[p_dc, :] = 0.0
            Lthlam[p_dc, p_dc] = 1.0
            ff[p_dc, :] = 0.0

        # solve only the Chebyshev columns that are non-negligible
        cols = [i for i in range(m) if np.max(np.abs(ff[:, i])) > 1e-16]
        if cols:
            CFS[:, cols, k] = np.linalg.solve(Lthlam, ff[:, cols])

    # permute back [2,3,1] -> (m, n, p)
    CFS = np.transpose(CFS, (1, 2, 0))
    return Ballfun.from_coeffs(CFS, is_real=is_real)


def ptdecomposition(v, nargout: int = 2):
    """Poloidal-toroidal decomposition of a divergence-free ``Ballfunv``.

    Returns ``(P, T)`` such that ``v = curl(curl(r*P)) + curl(r*T)``.
    """
    if v.isempty() or any(c.isempty() for c in v.components):
        raise ValueError("ballfunv must not have an empty component")

    # discretization: max over the component sizes, then pad
    sizes = np.array([c.shape for c in v.components])
    S = sizes.max(axis=0)
    m = int(S[0]) + 1
    n = int(S[1]) + 2
    p = int(S[2]) + 6

    from chebfunjax.discretization.ultras import multmat as _ultra_multmat

    Mr = np.asarray(_ultra_multmat(m, np.array([0.0, 1.0]), 0), dtype=np.complex128)

    MsinL = _fourier_multmat(n, [0.5j, 0.0, -0.5j])
    McosL = _fourier_multmat(n, [0.5, 0.0, 0.5])
    DF1L = _fourier_df1(n)

    MsinT = _fourier_multmat(p, [0.5j, 0.0, -0.5j])
    Msin2T = _fourier_multmat(p, [-0.25, 0.0, 0.5, 0.0, -0.25])
    McossinT = _fourier_multmat(p, [0.25j, 0.0, 0.0, 0.0, -0.25j])
    Msin2cosT = _fourier_multmat(p, [-0.125, 0.0, 0.125, 0.0, 0.125, 0.0, -0.125])
    Msin3T = _fourier_multmat(p, [-0.125j, 0.0, 0.375j, 0.0, -0.375j, 0.0, 0.125j])
    DF1T = _fourier_df1(p)

    Vx, Vy, Vz = (c for c in v.components)
    Vx = _coeffs3(Vx, m, n, p)
    Vy = _coeffs3(Vy, m, n, p)
    Vz = _coeffs3(Vz, m, n, p)

    # permute [2,3,1] -> (n, p, m)
    Vx = np.transpose(Vx, (1, 2, 0))
    Vy = np.transpose(Vy, (1, 2, 0))
    Vz = np.transpose(Vz, (1, 2, 0))

    RhsP = np.zeros((n, p, m), dtype=np.complex128)
    RhsT = np.zeros((n, p, m), dtype=np.complex128)

    Msin3T_T = Msin3T.T
    Msin2cosT_T = Msin2cosT.T
    McossinT_T = McossinT.T
    MsinT_T = MsinT.T
    DF1T_T = DF1T.T
    Msin2T_T = Msin2T.T

    for k in range(m):
        Xk = Vx[:, :, k]
        Yk = Vy[:, :, k]
        Zk = Vz[:, :, k]
        RhsP[:, :, k] = (
            -McosL @ Xk @ Msin3T_T
            - MsinL @ Yk @ Msin3T_T
            - Zk @ Msin2cosT_T
        )
        RhsT[:, :, k] = (
            MsinL @ Xk @ MsinT_T @ DF1T_T @ MsinT_T
            + DF1L @ McosL @ Xk @ McossinT_T
            - McosL @ Yk @ MsinT_T @ DF1T_T @ MsinT_T
            + DF1L @ MsinL @ Yk @ McossinT_T
            - DF1L @ Zk @ Msin2T_T
        )

    # permute back [3,1,2] -> (m, n, p)
    RhsP = np.transpose(RhsP, (2, 0, 1))
    RhsT = np.transpose(RhsT, (2, 0, 1))

    # multiply the poloidal rhs by r
    for k in range(p):
        RhsP[:, :, k] = Mr @ RhsP[:, :, k]

    is_real = all(c.is_real for c in v.components)
    P = _ptequation(RhsP, is_real)
    T = _ptequation(RhsT, is_real)

    if nargout <= 1:
        return [P, T]
    return P, T


# ----------------------------------------------------------------------
# Helmholtz decomposition (poloidal-toroidal form)
# ----------------------------------------------------------------------

def _compute_normal_boundary(v, n: int, p: int) -> np.ndarray:
    """Return ``v.r`` at ``r = 1`` as an ``n x p`` Fourier-Fourier matrix.

    MATLAB ``ComputeNormalBoundary``: evaluate each Cartesian component at
    the boundary (sum over the Chebyshev coefficients) then combine with
    the spherical unit-normal weights.
    """
    Vx = _coeffs3(v.components[0], v.components[0].shape[0], n, p)
    Vy = _coeffs3(v.components[1], v.components[1].shape[0], n, p)
    Vz = _coeffs3(v.components[2], v.components[2].shape[0], n, p)

    # evaluate at r = 1 : sum over Chebyshev coefficients (T_j(1) = 1)
    Vx = Vx.sum(axis=0)
    Vy = Vy.sum(axis=0)
    Vz = Vz.sum(axis=0)

    MsinL = _fourier_multmat(n, [0.5j, 0.0, -0.5j])
    McosL = _fourier_multmat(n, [0.5, 0.0, 0.5])
    MsinT = _fourier_multmat(p, [0.5j, 0.0, -0.5j])
    McosT = _fourier_multmat(p, [0.5, 0.0, 0.5])

    return McosL @ Vx @ MsinT.T + MsinL @ Vy @ MsinT.T + Vz @ McosT.T


def helmholtz_decomposition(v, nargout: int = 3):
    """Helmholtz decomposition of a ``Ballfunv`` in poloidal-toroidal form.

    ``nargout == 3`` returns ``(f, Ppsi, Tpsi)`` with
    ``v = grad(f) + curl(curl(r*Ppsi)) + curl(r*Tpsi)``.
    ``nargout == 4`` returns ``(f, Ppsi, Tpsi, phi)`` with the extra
    ``+ grad(phi)`` gradient term.
    """
    if v.isempty() or any(c.isempty() for c in v.components):
        raise ValueError("ballfunv must not have an empty component")

    if nargout == 3:
        return _helmholtz_2(v)
    if nargout == 4:
        return _helmholtz_3(v)
    raise ValueError(
        f"HelmholtzDecomposition undefined for {nargout} output arguments")


def _helmholtz_2(v):
    div_v = v.div()
    m, n, p = div_v.shape
    m, n, p = max(m, 5), max(n, 5), max(p, 5)

    v_bdy = _compute_normal_boundary(v, n, p)

    f = Ballfun.helmholtz(div_v, 0.0, v_bdy, m, n, p, bc_type="neumann")

    v1 = v - _grad_ballfunv(f)
    Pv1, Tv1 = ptdecomposition(v1)
    return f, Pv1, Tv1


def _helmholtz_3(v):
    div_v = v.div()
    m, n, p = div_v.shape
    m, n, p = max(m, 5), max(n, 5), max(p, 5)

    # Delta f = div(v), homogeneous Dirichlet
    f = Ballfun.helmholtz(div_v, 0.0, None, m, n, p)
    v1 = v - _grad_ballfunv(f)

    sizes = np.array([c.shape for c in v1.components])
    S = sizes.max(axis=0)
    m, n, p = max(int(S[0]), 5), max(int(S[1]), 5), max(int(S[2]), 5)

    v_bdy = _compute_normal_boundary(v1, n, p)

    # Delta phi = 0 with Neumann boundary r.grad(phi) = r.v1
    zero = Ballfun.from_function(lambda x, y, z: 0.0 * x)
    phi = Ballfun.helmholtz(zero, 0.0, v_bdy, m, n, p, bc_type="neumann")

    v2 = v1 - _grad_ballfunv(phi)

    Pv, Tv = ptdecomposition(v2)

    mt, nt, pt = Tv.shape
    mt, nt, pt = max(mt, 5), max(nt, 5), max(pt, 5)
    Ppsi = Ballfun.helmholtz(-Tv, 0.0, None, mt, nt, pt)
    Tpsi = Pv
    return f, Ppsi, Tpsi, phi


def _grad_ballfunv(f: Ballfun):
    from chebfunjax.ballfun.ballfunv import Ballfunv

    fx, fy, fz = f.grad()
    return Ballfunv(fx, fy, fz)
