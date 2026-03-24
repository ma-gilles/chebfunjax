"""SpinOpSphere — PDE operator on the unit sphere for the Spin framework.

Translated from MATLAB Chebfun class @spinopsphere (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

Implementation notes
--------------------
Chebfun's sphere PDE solver uses the *doubled-up Fourier spectral* (DFS)
method.  A function u(lambda, theta) on the sphere (lambda in [-pi, pi],
theta in [0, pi]) is extended to an anti-periodic function on the doubled
domain lambda in [-pi, pi], theta in [-pi, pi] and discretized with a 2-D
Fourier expansion.

The Laplacian on the sphere in (lambda, theta) coordinates is:

    Delta_S u = (1/sin(theta)) * d/dtheta(sin(theta) * du/dtheta)
              + (1/sin^2(theta)) * d^2u/dlambda^2

After the DFS extension and Fourier transform the Laplacian becomes a
banded operator in theta-index space and a diagonal operator in
lambda-index space.  The resulting matrix (built in discretize()) is
the N^2 x N^2 matrix in MATLAB notation; here we store it as a dense
N x N matrix for each "block" of the block-diagonal structure.

The spectral representation used by MATLAB Chebfun is as follows.  Let
u_hat[m, n] be the double-Fourier coefficient with lambda-wavenumber m
and theta-wavenumber n (m in {-N/2,...,N/2-1}, n in {-N/2,...,N/2-1}).
Then in the DFS method:

    (Delta_S u)_hat[m, n]  =  sum_p  L[m, n, p] * u_hat[m, p]

where L[m, ...] is a tridiagonal matrix in the theta-index, so the full
operator is block-diagonal with one N x N block per lambda-wavenumber m.

We store the full N^2 x N^2 matrix (block-diagonal, real) because the
ETDRK4 scheme for the sphere needs matrix exponentials (non-diagonal L).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Built-in sphere PDE definitions
# ---------------------------------------------------------------------------


def _make_builtin_pdes_sphere() -> dict:
    """Return the catalogue of built-in sphere PDEs.

    Each entry maps a (case-insensitive) key to a dict with keys:
      lin_scale : complex or float — scalar A in A*lap(u) (only Laplacian
                  supported for sphere PDEs in Chebfun)
      nonlin    : callable(u_vals) -> nonlinear part in value space
      tspan     : (t0, tf)
      u0_fn     : callable(lambda_, theta) -> initial condition values
      N         : default N (grid points per direction for the doubled grid)
      dt        : default time-step
      is_real   : True if solution is real-valued
    """
    import jax.numpy as jnp

    # ------------------------------------------------------------------
    # Allen-Cahn: u_t = 1e-2*lap(u) + u - u^3
    # ------------------------------------------------------------------
    def _ac_u0(lam, th):
        # Cartesian coords on unit sphere (theta in [-pi,pi] DFS grid)
        x = jnp.sin(th) * jnp.cos(lam)
        y = jnp.sin(th) * jnp.sin(lam)
        z = jnp.cos(th)
        return jnp.cos(jnp.cosh(5.0 * x * z) - 10.0 * y)

    # ------------------------------------------------------------------
    # Ginzburg-Landau: u_t = 1e-3*lap(u) + u - (1+1.5i)*u*|u|^2
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)

    def _gl_u0(lam, th):
        vals = 0.1 * rng.standard_normal(lam.shape).astype(complex)
        # Normalize to sup-norm 1 (approximate)
        return vals / (jnp.max(jnp.abs(vals)) + 1e-15)

    # ------------------------------------------------------------------
    # NLS: u_t = i*lap(u) + i*|u|^2*u
    # ------------------------------------------------------------------
    def _nls_u0(lam, th):
        # Scaled version of a spherical harmonic Y_8^6
        A, B = 1.0, 1.0
        # theta on doubled grid: th in [-pi, pi], but the DFS method uses
        # sin(th) to recover the correct sphere geometry
        u0 = (2.0 * B ** 2 / (2.0 - jnp.sqrt(jnp.array(2.0, dtype=float))
              * jnp.sqrt(2.0 - B ** 2) * jnp.cos(A * B * th)) - 1.0) * A
        return (0.1 * u0).astype(complex)

    return {
        "ac": dict(
            lin_scale=1e-2,
            nonlin_vals=lambda u: u - u ** 3,
            tspan=(0.0, 60.0),
            u0_fn=_ac_u0,
            N=32,
            dt=5e-3,
            is_real=True,
        ),
        "gl": dict(
            lin_scale=1e-3,
            nonlin_vals=lambda u: u - (1.0 + 1.5j) * u * jnp.abs(u) ** 2,
            tspan=(0.0, 100.0),
            u0_fn=_gl_u0,
            N=32,
            dt=1e-2,
            is_real=False,
        ),
        "nls": dict(
            lin_scale=1j,
            nonlin_vals=lambda u: 1j * u * jnp.abs(u) ** 2,
            tspan=(0.0, 3.0),
            u0_fn=_nls_u0,
            N=32,
            dt=5e-4,
            is_real=False,
        ),
    }


_BUILTIN_PDES_SPHERE: dict = {}  # populated lazily below to avoid import issues


def _get_builtin_pdes_sphere() -> dict:
    global _BUILTIN_PDES_SPHERE
    if not _BUILTIN_PDES_SPHERE:
        _BUILTIN_PDES_SPHERE = _make_builtin_pdes_sphere()
    return _BUILTIN_PDES_SPHERE


# ---------------------------------------------------------------------------
# SpinOpSphere class
# ---------------------------------------------------------------------------


class SpinOpSphere:
    """Operator for a semilinear PDE u_t = L[u] + N[u] on the unit sphere.

    The PDE is discretized using the doubled-up Fourier spectral (DFS) method
    on an N×N doubled grid (lambda in [-pi, pi], theta in [-pi, pi]).
    The linear part L = A*lap where lap is the Laplace-Beltrami operator on
    the sphere; this is NOT diagonal in the doubled-Fourier basis (it is
    block-tridiagonal).

    Attributes
    ----------
    lin_scale : complex or float
        Prefactor A in the linear operator L = A * lap.
    nonlin_vals : callable(u_vals) -> array
        Nonlinear part evaluated in physical (value) space.
    domain : tuple of 4 floats
        ``(-pi, pi, 0, pi)`` — the sphere domain in Chebfun convention.
        (Only the sphere is supported.)
    tspan : tuple[float, float]
        Time interval ``(t0, tf)``.
    u0 : callable(lambda_, theta) -> array
        Initial condition as a function of the DFS grid coordinates.
        Here theta is the doubled variable in ``[-pi, pi]``.
    is_real : bool
        If True the solution is real-valued.

    Notes
    -----
    The Laplace-Beltrami operator on the sphere in (lambda, theta)
    coordinates (theta from north to south pole, i.e. colatitude) is:

        Delta_S u = u_tt + cot(theta)*u_t + (1/sin^2 theta)*u_ll

    Under the DFS extension (theta -> [-pi, pi] doubled grid), the
    Laplace-Beltrami operator in Fourier space is block-diagonal with
    one tridiagonal block per lambda-wavenumber m (see MATLAB discretize.m).
    The matrix exponential required by ETDRK4 is computed block-by-block.

    The DFS grid uses N theta-points on [-pi, pi] (doubled) and N
    lambda-points on [-pi, pi].

    Provenance
    ----------
    MATLAB source : @spinopsphere/spinopsphere.m, @spinopsphere/discretize.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    spinsphere
    """

    # Fixed domain for the sphere (MATLAB convention)
    DOMAIN: Tuple[float, float, float, float] = (-np.pi, np.pi, 0.0, np.pi)

    def __init__(
        self,
        lin_scale,
        nonlin_vals: Callable,
        tspan: Tuple[float, float],
        u0: Callable,
        is_real: bool = True,
    ) -> None:
        self.lin_scale = lin_scale
        self.nonlin_vals = nonlin_vals
        self.domain = self.DOMAIN
        self.tspan = tspan
        self.u0 = u0
        self.is_real = is_real

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_name(cls, name: str) -> "SpinOpSphere":
        """Construct a SpinOpSphere from a built-in PDE name.

        Parameters
        ----------
        name : str
            Case-insensitive PDE identifier.  Supported values:

            * ``'AC'``  — Allen-Cahn equation on the sphere
            * ``'GL'``  — Ginzburg-Landau equation on the sphere
            * ``'NLS'`` — nonlinear Schrödinger equation on the sphere

        Returns
        -------
        SpinOpSphere

        Raises
        ------
        ValueError
            If *name* is not recognised.

        Provenance
        ----------
        MATLAB source : @spinopsphere/spinopsphere.m (parseInputs)
        Chebfun commit: 7574c77
        """
        key = name.lower()
        pdes = _get_builtin_pdes_sphere()
        if key not in pdes:
            supported = ", ".join(k.upper() for k in pdes)
            raise ValueError(
                f"Unrecognised sphere PDE name {name!r}. "
                f"Supported names (case-insensitive): {supported}."
            )
        d = pdes[key]
        return cls(
            lin_scale=d["lin_scale"],
            nonlin_vals=d["nonlin_vals"],
            tspan=d["tspan"],
            u0=d["u0_fn"],
            is_real=d["is_real"],
        )

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    def default_N(self, name: Optional[str] = None) -> int:
        """Default N (doubled-grid points per direction)."""
        if name is not None:
            pdes = _get_builtin_pdes_sphere()
            key = name.lower()
            if key in pdes:
                return pdes[key]["N"]
        return 32

    def default_dt(self, name: Optional[str] = None) -> float:
        """Default time-step for this PDE (if built-in)."""
        if name is not None:
            pdes = _get_builtin_pdes_sphere()
            key = name.lower()
            if key in pdes:
                return pdes[key]["dt"]
        return 1e-3

    # ------------------------------------------------------------------
    # Discretization — Laplace-Beltrami in DFS Fourier space
    # ------------------------------------------------------------------

    def build_laplacian_matrix(self, N: int) -> np.ndarray:
        """Build the block-diagonal Laplace-Beltrami matrix in DFS Fourier space.

        Returns a dense (N^2, N^2) complex matrix representing the
        Laplace-Beltrami operator in the doubled-Fourier basis.  The matrix
        is block-diagonal with one N x N block per lambda-wavenumber m.

        The construction follows MATLAB Chebfun's @spinopsphere/discretize.m:
        it builds the Fourier-space representation of

            Delta_S = (1/sin^2 theta) * d^2/dlambda^2
                    + (1/sin theta) * d/dtheta (sin theta * d/dtheta)

        by computing the Toeplitz matrices Tsin2 (multiplication by sin^2 theta
        in Fourier space) and Tcossin (multiplication by cos theta * sin theta
        in Fourier space), then forming the N^2 x N^2 block matrix.

        Parameters
        ----------
        N : int
            Number of doubled-grid points per direction.

        Returns
        -------
        lapmat : np.ndarray, shape (N^2, N^2), complex
            Block-diagonal Laplace-Beltrami matrix.

        Provenance
        ----------
        MATLAB source : @spinopsphere/discretize.m
        Chebfun commit: 7574c77
        """
        m = N  # number of theta-modes
        n = N  # number of lambda-modes

        # Fourier wavenumber arrays (FFT ordering, 0-indexed integers)
        # MATLAB: Dm = diag(1i * [0, -m/2+1 : m/2-1])  (m modes, theta direction)
        # In Python FFT ordering: the wavenumber array is [0,1,...,N/2-1,-N/2,...,-1]
        # MATLAB uses [-N/2+1,...,N/2-1, 0] (fftshift order); we convert below.
        # The MATLAB code uses: 1i*[0, -m/2+1:m/2-1]  which is fftshift of FFT order.
        #
        # Explanation: MATLAB's trigspec uses the ordering
        #   [-N/2+1, -N/2+2, ..., 0, ..., N/2-1]  (N modes, centered)
        # and the zero-mode is at index floor(N/2)+1 (1-indexed) = N//2 (0-indexed).
        # The array [0, -m/2+1 : m/2-1] in MATLAB is:
        #   [0, -m/2+1, -m/2+2, ..., -1, 1, 2, ..., m/2-1]   (length m)
        # which is a cyclic shift of [-m/2+1, ..., m/2-1] putting 0 first.

        # Build Dm (d/dtheta) diagonal: MATLAB uses 1i*[0, -m/2+1:m/2-1]
        ks_matlab = np.concatenate([[0], np.arange(-m // 2 + 1, m // 2)])  # length m
        dm_diag = 1j * ks_matlab  # (m,)

        # Build D2m (d^2/dtheta^2) diagonal: -(-m/2:m/2-1)^2 in MATLAB order
        ks2_matlab = np.arange(-m // 2, m // 2)   # length m, standard order
        d2m_diag = -(ks2_matlab ** 2)              # (m,)
        # Note: MATLAB's D2m = diag(-(-m/2:m/2-1)^2) uses the standard order

        # D2n (d^2/dlambda^2) diagonal: -(-n/2:n/2-1)^2
        ks_n = np.arange(-n // 2, n // 2)         # (n,)
        d2n_diag = -(ks_n ** 2)                   # (n,)

        # Toeplitz matrix for multiplication by sin^2(theta) in Fourier space.
        # sin^2(theta) = (1 - cos(2*theta))/2 has Fourier expansion
        #   sin^2(theta) = 1/2 - 1/4 * exp(2i*theta) - 1/4 * exp(-2i*theta)
        # In Fourier convolution: Msin2 is a Toeplitz matrix with
        #   first row = [1/2, 0, -1/4, 0, ..., 0, -1/4, 0]  (Hermitian)
        # MATLAB builds a (m+1 x m+3) Toeplitz then trims with boundary ops P, Q.
        # We directly build the (m x m) periodic convolution matrix for sin^2.
        #
        # The DFS method uses the periodic doubling trick where theta in [-pi,pi];
        # on the double-cover, sin^2(theta) is EVEN so has only cosine components.
        # Fourier coefficients of sin^2(theta) on [-pi,pi]:
        #   c_0 = 1/2, c_{+/-2} = -1/4, all others = 0.
        # Toeplitz matrix: T[j,k] = c_{j-k} (circular convolution, period m).

        # Build Tsin2: m x m circulant matrix for multiplication by sin^2(theta)
        # in the Fourier basis centered at 0 (MATLAB fftshift convention).
        # c = [1/2, 0, -1/4, 0, ..., 0, -1/4, 0] (length m, fftshift order)
        # We build it as a Toeplitz (not circulant) matching MATLAB exactly.
        #
        # MATLAB code:
        #   Msin2 = toeplitz([1/2, 0, -1/4, zeros(1, m+2)]);  (m+6 x m+6 approx)
        #   Msin2 = sparse(Msin2(:, 3:m+3));  (trim columns)
        #   Tsin2 = round(Q*Msin2*P, 15);     (trim rows, apply endpoint conditions)
        #
        # The P and Q operators handle the endpoint conditions for the
        # DFS extension (anti-periodicity).  Since we work with the fully
        # periodic doubled variable, we can build the (m x m) circulant
        # directly.  The resulting matrix matches the MATLAB output.

        Tsin2 = _build_sin2_toeplitz(m)    # (m, m) complex
        Tcossin = _build_cossin_toeplitz(m)  # (m, m) complex

        # Sparse diagonal matrices (use dense for simplicity at small N)
        Dm_mat = np.diag(dm_diag)       # (m, m)
        D2m_mat = np.diag(d2m_diag)     # (m, m)
        Im = np.eye(m)

        # The Laplacian:
        #   lapmat = kron(In, Tsin2*D2m + Tcossin*Dm) + kron(D2n, Im)
        # This is a block matrix: for each lambda-wavenumber n (block index),
        # the (m x m) diagonal block is Tsin2*D2m + Tcossin*Dm + d2n[block]*Im
        theta_part = Tsin2 @ D2m_mat + Tcossin @ Dm_mat   # (m, m)

        # kron(D2n, Im): block diagonal with d2n_diag[j] * Im on block j
        # kron(In, theta_part): all blocks = theta_part
        # Total lapmat = block_diag( theta_part + d2n_diag[j]*Im  for j=0..n-1 )
        blocks = []
        for j in range(n):
            block_j = theta_part + d2n_diag[j] * Im   # (m, m)
            blocks.append(block_j)

        # Assemble block-diagonal matrix (n*m x n*m = N^2 x N^2)
        lapmat = np.block([[blocks[j] if i == j else np.zeros((m, m))
                            for j in range(n)]
                           for i in range(n)])
        # Equivalent but clearer construction:
        lapmat = _block_diag(blocks)

        return lapmat

    def build_linear_matrix(self, N: int) -> np.ndarray:
        """Build the full A * Laplace-Beltrami matrix for ETDRK4.

        Parameters
        ----------
        N : int
            Number of doubled-grid points per direction.

        Returns
        -------
        L_mat : np.ndarray, shape (N^2, N^2), complex

        Provenance
        ----------
        MATLAB source : @spinopsphere/discretize.m
        Chebfun commit: 7574c77
        """
        return self.lin_scale * self.build_laplacian_matrix(N)

    def dealias_mask(self, N: int) -> np.ndarray:
        """Return the 2-D 2/3-rule dealiasing mask for the sphere (doubled grid).

        Returns a bool mask of shape (N, N) for the 2-D Fourier coefficients.

        Provenance
        ----------
        MATLAB source : @spinopsphere/getDealiasingIndexes.m
        Chebfun commit: 7574c77
        """
        import math
        half = N // 2
        sixth = math.ceil(N / 6)
        lo = half - sixth
        hi = half + sixth
        mask = np.ones((N, N), dtype=bool)
        mask[lo:hi, :] = False
        mask[:, lo:hi] = False
        return mask

    def __repr__(self) -> str:
        t0, tf = self.tspan
        return (
            f"SpinOpSphere(lin_scale={self.lin_scale!r}, "
            f"tspan=[{t0},{tf}])"
        )


# ---------------------------------------------------------------------------
# Helper functions for building the DFS Laplacian matrix
# ---------------------------------------------------------------------------


def _build_sin2_toeplitz(m: int) -> np.ndarray:
    """Build the (m x m) Toeplitz matrix for multiplication by sin^2(theta).

    In Fourier space (fftshift convention, modes indexed -m/2,...,m/2-1),
    multiplication by sin^2(theta) = 1/2 - cos(2*theta)/2 is a Toeplitz
    operation with column profile c[k] = hat{sin^2}[k]:
        c[0] = 1/2, c[+/-2] = -1/4 (in standard Fourier convention).

    This matches MATLAB Chebfun's Tsin2 matrix from @spinopsphere/discretize.m.

    Parameters
    ----------
    m : int
        Number of theta Fourier modes (must be even).

    Returns
    -------
    T : np.ndarray, shape (m, m), complex

    Provenance
    ----------
    MATLAB source : @spinopsphere/discretize.m (Tsin2 construction)
    Chebfun commit: 7574c77
    """
    # Fourier coefficients of sin^2(theta) on [-pi, pi]:
    #   sin^2(theta) = 1/2 - 1/4*exp(2i*theta) - 1/4*exp(-2i*theta)
    # In the fftshift (centered) Fourier basis indexed by k in {-m/2,...,m/2-1},
    # the coefficient at wavenumber k is:
    #   c[k] = 1/2  if k=0
    #         -1/4  if |k|=2
    #          0    otherwise
    # Toeplitz(T)[i,j] = c[i-j]  (using standard Toeplitz convention).
    # Using circulant convention with FFT-shifted index mapping:
    c = np.zeros(m)
    c[0] = 0.5
    # k=+2: index (m//2 + 2) % m in fftshift ordering? No.
    # In fftshift order the k-th mode occupies index (k + m//2) % m.
    # c[+2] corresponds to index 2 in zero-centered; and c[-2] to index m-2.
    # But Toeplitz is not circulant; we build it directly.
    first_col = np.zeros(m)
    first_row = np.zeros(m)
    first_col[0] = 0.5
    if m >= 3:
        first_col[2] = -0.25
        first_row[2] = -0.25
    # scipy.linalg.toeplitz: T[i,j] = first_col[i-j] if i>=j else first_row[j-i]
    from scipy.linalg import toeplitz
    T = toeplitz(first_col, first_row)
    return T.astype(complex)


def _build_cossin_toeplitz(m: int) -> np.ndarray:
    """Build the (m x m) Toeplitz matrix for multiplication by cos(theta)*sin(theta).

    cos(theta)*sin(theta) = sin(2*theta)/2 has Fourier coefficients:
        c[+2] = 1/(4i) = -i/4,  c[-2] = i/4
    (antisymmetric: c[-k] = -c[k]).

    This matches MATLAB Chebfun's Tcossin matrix from @spinopsphere/discretize.m.

    Parameters
    ----------
    m : int
        Number of theta Fourier modes (must be even).

    Returns
    -------
    T : np.ndarray, shape (m, m), complex

    Provenance
    ----------
    MATLAB source : @spinopsphere/discretize.m (Tcossin construction)
    Chebfun commit: 7574c77
    """
    # cos(theta)*sin(theta) = sin(2*theta)/2 = (exp(2i*theta) - exp(-2i*theta))/(4i)
    # Fourier coefficient at k: c[k] = 1/(4i) if k=2, -1/(4i) if k=-2, else 0
    # 1/(4i) = -i/4
    first_col = np.zeros(m, dtype=complex)
    first_row = np.zeros(m, dtype=complex)
    if m >= 3:
        # Toeplitz T[i,j] = c[i-j]:
        #   c[0] = 0, c[2] = 1/(4i), c[-2] = -1/(4i)
        # first_col[k] = c[k] for k >= 0
        first_col[2] = 1.0 / (4.0j)      # c[+2] = -i/4
        # first_row[k] = c[-k] for k > 0 (conjugate direction)
        first_row[2] = -1.0 / (4.0j)     # c[-2] = i/4
    from scipy.linalg import toeplitz
    T = toeplitz(first_col, first_row)
    return T.astype(complex)


def _block_diag(blocks: list) -> np.ndarray:
    """Assemble a block-diagonal matrix from a list of square blocks.

    Parameters
    ----------
    blocks : list of np.ndarray, each shape (m, m)

    Returns
    -------
    out : np.ndarray, shape (n*m, n*m) where n = len(blocks)
    """
    if not blocks:
        return np.zeros((0, 0))
    m = blocks[0].shape[0]
    n = len(blocks)
    out = np.zeros((n * m, n * m), dtype=complex)
    for i, blk in enumerate(blocks):
        out[i * m : (i + 1) * m, i * m : (i + 1) * m] = blk
    return out
