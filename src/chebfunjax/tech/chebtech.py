"""Chebyshev technology -- smooth function approximation on [-1, 1].

Translated from MATLAB Chebfun class @chebtech2 (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# =========================================================================
# Pure-function helpers (JIT-safe where noted)
# =========================================================================


def _clenshaw(coeffs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate a Chebyshev series at point(s) *x* via Clenshaw's algorithm.

    JIT-safe: yes (fixed shapes).

    Parameters
    ----------
    coeffs : jnp.ndarray, shape (n,)
        Chebyshev coefficients c_0, c_1, ..., c_{n-1}.
    x : jnp.ndarray
        Evaluation points in [-1, 1].

    Returns
    -------
    jnp.ndarray
        Values of the Chebyshev series at *x*.
    """
    n = coeffs.shape[0]
    if n == 0:
        return jnp.zeros_like(x, dtype=jnp.float64)
    if n == 1:
        return jnp.broadcast_to(coeffs[0], x.shape)

    x = jnp.asarray(x, dtype=jnp.float64)
    bk1 = jnp.zeros_like(x)
    bk2 = jnp.zeros_like(x)

    for k in range(n - 1, 0, -1):
        bk1, bk2 = 2.0 * x * bk1 - bk2 + coeffs[k], bk1

    return x * bk1 - bk2 + coeffs[0]


def _coeffs_to_values(coeffs: jnp.ndarray) -> jnp.ndarray:
    """Convert Chebyshev coefficients to values at 2nd-kind Chebyshev points.

    JIT-safe: yes.
    """
    from chebfunjax.utils.transforms import coeffs2vals

    return coeffs2vals(coeffs)


def _values_to_coeffs(values: jnp.ndarray) -> jnp.ndarray:
    """Convert values at 2nd-kind Chebyshev points to Chebyshev coefficients.

    JIT-safe: yes.
    """
    from chebfunjax.utils.transforms import vals2coeffs

    return vals2coeffs(values)


# =========================================================================
# Coefficient-level operations (all JIT-safe)
# =========================================================================


def _diff_coeffs_once(c: jnp.ndarray) -> jnp.ndarray:
    """Single differentiation via the Chebyshev coefficient recurrence.

    Given Chebyshev coefficients c_0, ..., c_{n-1} of a polynomial p,
    returns coefficients d_0, ..., d_{n-2} of p'.

    The recurrence (Mason & Handscomb, p. 34):
        d_{n-1} = d_n = 0
        d_r     = d_{r+2} + 2*(r+1)*c_{r+1}   for r = n-2, n-3, ..., 1
        d_0     = d_2 / 2 + c_1

    JIT-safe: yes.

    Provenance
    ----------
    MATLAB source : @chebtech/diff.m  (computeDerCoeffs)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: Page 34 of Mason & Handscomb, "Chebyshev Polynomials",
        Chapman & Hall/CRC, 2003.
    """
    n = c.shape[0]
    if n <= 1:
        return jnp.zeros(1, dtype=jnp.float64)

    # w[k] = 2*(k+1) for k = 0 .. n-2
    w = 2.0 * jnp.arange(1, n, dtype=jnp.float64)
    v = w * c[1:]  # v[k] = 2*(k+1)*c_{k+1}

    # Accumulate from the tail, even and odd indices separately:
    # cout[n-2:-2:0] = cumsum(v[n-2:-2:0])   (descending, step -2)
    out = jnp.zeros(n - 1, dtype=jnp.float64)

    # Odd-indexed outputs: out[n-2], out[n-4], ...
    # Even-indexed outputs: out[n-3], out[n-5], ...
    # Use reversed cumsum on interleaved slices.

    # Build the full derivative using the scan-from-tail approach:
    # We reverse v, accumulate, then reverse back.
    # Odd positions (n-2, n-4, ...): start from index n-2 of v
    # Even positions (n-3, n-5, ...): start from index n-3 of v

    # Slice1: indices n-2, n-4, ..., i.e. v[-1], v[-3], ...
    s1 = v[::-1][::2]  # reversed, take every other
    cs1 = jnp.cumsum(s1)
    # Slice2: indices n-3, n-5, ..., i.e. v[-2], v[-4], ...
    s2 = v[::-1][1::2]
    cs2 = jnp.cumsum(s2)

    # Place back
    out = out.at[::-1].set(0.0)  # reset
    out = out.at[-1::-2].set(cs1)
    if cs2.shape[0] > 0:
        out = out.at[-2::-2].set(cs2)

    # Fix the c_0 coefficient: d_0 = d_2/2 + c_1 => already in out but halved
    out = out.at[0].multiply(0.5)

    return out


def _diff_coeffs(coeffs: jnp.ndarray, k: int) -> jnp.ndarray:
    """Differentiate Chebyshev coefficients *k* times.

    JIT-safe: yes (k must be a static integer).

    Provenance
    ----------
    MATLAB source : @chebtech/diff.m
    Chebfun commit: 7574c77
    """
    c = coeffs
    for _ in range(k):
        c = _diff_coeffs_once(c)
    return c


def _cumsum_coeffs(c: jnp.ndarray) -> jnp.ndarray:
    """Antiderivative via the Chebyshev coefficient recurrence, with F(-1)=0.

    Given c_0, ..., c_{n-1}, returns b_0, ..., b_n where
        b_1 = c_0 - c_2/2,
        b_r = (c_{r-1} - c_{r+1}) / (2*r)  for r >= 2,
        b_0 = sum_{r=1}^{n} (-1)^{r+1} b_r   (ensures F(-1)=0).

    JIT-safe: yes.

    Provenance
    ----------
    MATLAB source : @chebtech/cumsum.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: Pages 32-33 of Mason & Handscomb, "Chebyshev Polynomials",
        Chapman & Hall/CRC, 2003.
    """
    n = c.shape[0]
    if n == 0:
        return jnp.zeros(1, dtype=jnp.float64)

    # Pad with two zeros so that c_{n} = c_{n+1} = 0
    cp = jnp.concatenate([c, jnp.zeros(2, dtype=jnp.float64)])

    b = jnp.zeros(n + 1, dtype=jnp.float64)

    # b[r] = (c[r-1] - c[r+1]) / (2*r) for r = 2, ..., n
    rk = jnp.arange(2, n + 1, dtype=jnp.float64)
    b = b.at[2:n + 1].set((cp[1:n] - cp[3:n + 2]) / (2.0 * rk))

    # b[1] = c[0] - c[2]/2
    b = b.at[1].set(cp[0] - cp[2] / 2.0)

    # b[0]: choose so that F(-1) = 0
    # F(-1) = sum_r b_r * T_r(-1) = sum_r b_r * (-1)^r = 0
    # => b_0 = - sum_{r=1}^{n} (-1)^r * b_r = sum_{r=1}^{n} (-1)^{r+1} * b_r
    v = jnp.ones(n, dtype=jnp.float64)
    v = v.at[1::2].set(-1.0)
    b = b.at[0].set(jnp.dot(v, b[1:n + 1]))

    return b


def _definite_integral(coeffs: jnp.ndarray) -> jnp.ndarray:
    r"""Definite integral of a Chebyshev expansion over [-1, 1].

    Uses the fact that \int_{-1}^{1} T_k(x) dx = 2/(1-k^2) for even k, 0 for odd k.
    (Trefethen, ATAP, Thm 19.2.)

    JIT-safe: yes.

    Provenance
    ----------
    MATLAB source : @chebtech/sum.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    n = coeffs.shape[0]
    if n == 0:
        return jnp.array(0.0, dtype=jnp.float64)
    if n == 1:
        return 2.0 * coeffs[0]

    # Chebyshev moments: m_k = 2/(1-k^2) for even k, 0 for odd k
    k = jnp.arange(n, dtype=jnp.float64)
    moments = jnp.where(
        k % 2 == 0,
        2.0 / (1.0 - k**2),
        0.0,
    )
    # Fix k=0: moment is 2 (from the formula, 2/(1-0)=2)
    # The formula already gives 2/(1-0)=2 for k=0, which is correct.
    return jnp.dot(coeffs, moments)


def _inner_product(f_coeffs: jnp.ndarray, g_coeffs: jnp.ndarray) -> jnp.ndarray:
    r"""L^2 inner product <f, g> = \int_{-1}^{1} f(x) g(x) dx.

    Computed by prolonging both to length n_f + n_g (so quadrature is exact),
    converting to values, and applying Clenshaw-Curtis quadrature weights.

    JIT-safe: yes (shapes fixed once called).

    Provenance
    ----------
    MATLAB source : @chebtech/innerProduct.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    from chebfunjax.utils.quadrature import chebweights

    nf = f_coeffs.shape[0]
    ng = g_coeffs.shape[0]
    n = nf + ng

    # Prolong both to length n
    fc = jnp.zeros(n, dtype=jnp.float64).at[:nf].set(f_coeffs)
    gc = jnp.zeros(n, dtype=jnp.float64).at[:ng].set(g_coeffs)

    # Convert to values
    fv = _coeffs_to_values(fc)
    gv = _coeffs_to_values(gc)

    # Clenshaw-Curtis weights
    w = chebweights(n, kind=2)

    return jnp.dot(w * fv, gv)


def _coeff_multiply(fc: jnp.ndarray, gc: jnp.ndarray) -> jnp.ndarray:
    """Multiply two Chebyshev series in coefficient space via FFT.

    Given coefficients f_0, ..., f_{m-1} and g_0, ..., g_{p-1},
    returns the coefficients of f*g (length m+p-1).

    Uses the Toeplitz-plus-Hankel-plus-rank-one embedding into a circulant
    matrix and applied using the FFT (Olver & Townsend, SIAM Review, 2013).

    JIT-safe: yes.

    Provenance
    ----------
    MATLAB source : @chebtech/times.m  (coeff_times)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    nf = fc.shape[0]
    ng = gc.shape[0]
    mn = nf + ng - 1

    # Pad both to length mn
    f = jnp.zeros(mn, dtype=jnp.float64).at[:nf].set(fc)
    g = jnp.zeros(mn, dtype=jnp.float64).at[:ng].set(gc)

    # Embed into circulant: double the first coefficient
    t = jnp.concatenate([2.0 * f[:1], f[1:]])
    x = jnp.concatenate([2.0 * g[:1], g[1:]])

    # Circulant multiply via FFT
    t_ext = jnp.concatenate([t, t[-1:0:-1]])
    x_ext = jnp.concatenate([x, x[-1:0:-1]])
    product = jnp.real(jnp.fft.ifft(jnp.fft.fft(t_ext) * jnp.fft.fft(x_ext)))

    # Extract result
    hc = 0.25 * jnp.concatenate([product[:1], product[1:mn] + product[-1:mn - 1:-1]])

    return hc


def _prolong_coeffs(coeffs: jnp.ndarray, n: int) -> jnp.ndarray:
    """Zero-pad or truncate Chebyshev coefficients to length *n*."""
    m = coeffs.shape[0]
    if m >= n:
        return coeffs[:n]
    return jnp.concatenate([coeffs, jnp.zeros(n - m, dtype=jnp.float64)])


def _roots_colleague(coeffs: jnp.ndarray) -> jnp.ndarray:
    """Find all real roots of a Chebyshev expansion in [-1, 1].

    Uses recursive subdivision for degree > 50 and colleague matrix
    eigenvalue computation for degree <= 50.

    NOT JIT-safe (variable output size, recursive subdivision).

    Provenance
    ----------
    MATLAB source : @chebtech/roots.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm:
        [1] I. J. Good, "The colleague matrix, a Chebyshev analogue of the
            companion matrix", QJM 12, 1961.
        [2] J. A. Boyd, "Computing zeros on a real interval through Chebyshev
            expansion and polynomial rootfinding", SIAM J. Numer. Anal. 40, 2002.
        [3] L. N. Trefethen, ATAP, SIAM, 2013, Chapter 18.
    """
    c = np.asarray(coeffs, dtype=np.float64)
    htol = 100.0 * np.finfo(np.float64).eps

    # Normalize
    vscl = np.max(np.abs(c))
    if vscl == 0.0:
        return jnp.array([0.0], dtype=jnp.float64)
    c_scaled = c / vscl

    r = _roots_main(c_scaled, htol)
    r = np.sort(r)
    return jnp.asarray(r, dtype=jnp.float64)


def _roots_main(c: np.ndarray, htol: float) -> np.ndarray:
    """Recursive root-finding engine (numpy, NOT JIT-safe).

    Follows MATLAB Chebfun's roots.m strategy:
    - Trim trailing small coefficients.
    - If degree > 50, subdivide at a slightly off-center point and recurse.
    - If degree <= 50, form the colleague matrix and compute eigenvalues.
    """
    SPLIT_POINT = -0.004849834917525
    MAX_EIG_SIZE = 50

    # Trim small trailing coefficients
    tail_max = 5.0 * np.finfo(np.float64).eps * np.linalg.norm(c, 1)
    idx = np.where(np.abs(c) > tail_max)[0]
    if idx.size == 0:
        return np.array([0.0])
    n = int(idx[-1]) + 1
    c = c[:n]

    # Trivial cases
    if n == 1:
        if c[0] == 0.0:
            return np.array([0.0])
        return np.array([], dtype=np.float64)

    if n == 2:
        r = np.array([-c[0] / c[1]])
        mask_im = np.abs(np.imag(r)) < htol
        r = np.real(r[mask_im])
        r = r[(r >= -(1.0 + htol)) & (r <= (1.0 + htol))]
        r = np.clip(r, -1.0, 1.0)
        return r

    if n - 1 <= MAX_EIG_SIZE:
        # Form the colleague matrix
        c_adj = -0.5 * c[:-1] / c[-1]
        c_adj[-2] += 0.5

        nn = n - 1
        oh = 0.5 * np.ones(nn - 1)
        A = np.diag(oh, 1) + np.diag(oh, -1)
        A[-2, -1] = 1.0
        A[:, 0] = c_adj[::-1]

        rts = np.linalg.eigvals(A)

        # Filter: keep roots with small imaginary part and inside [-1, 1]
        mask = np.abs(np.imag(rts)) < htol
        rts = np.real(rts[mask])
        rts = rts[np.abs(rts) <= 1.0 + htol]
        rts = np.sort(rts)
        if rts.size > 0:
            rts[0] = max(rts[0], -1.0)
            rts[-1] = min(rts[-1], 1.0)
        return rts

    # Subdivide and recurse
    # Evaluate on left and right subintervals
    from chebfunjax.utils.quadrature import chebpts as _chebpts

    pts = np.asarray(_chebpts(n, kind=2))

    # Map Chebyshev points to left and right subintervals
    a_left, b_left = -1.0, SPLIT_POINT
    a_right, b_right = SPLIT_POINT, 1.0

    x_left = 0.5 * ((b_left - a_left) * pts + (b_left + a_left))
    x_right = 0.5 * ((b_right - a_right) * pts + (b_right + a_right))

    # Evaluate using Clenshaw
    def _eval_cheb(x_arr, cc):
        """Evaluate Chebyshev series at numpy points."""
        nn = cc.shape[0]
        bk1 = np.zeros_like(x_arr)
        bk2 = np.zeros_like(x_arr)
        for k in range(nn - 1, 0, -1):
            bk1_new = 2.0 * x_arr * bk1 - bk2 + cc[k]
            bk2 = bk1
            bk1 = bk1_new
        return x_arr * bk1 - bk2 + cc[0]

    v_left = _eval_cheb(x_left, c)
    v_right = _eval_cheb(x_right, c)

    # Convert values to coefficients (using numpy FFT)
    from chebfunjax.utils.transforms import vals2coeffs as _v2c

    c_left = np.asarray(_v2c(jnp.asarray(v_left)))
    c_right = np.asarray(_v2c(jnp.asarray(v_right)))

    # Recurse
    r_left = _roots_main(c_left, 2.0 * htol)
    r_right = _roots_main(c_right, 2.0 * htol)

    # Map back to original interval
    r_left_mapped = 0.5 * (SPLIT_POINT - 1.0) + 0.5 * (SPLIT_POINT + 1.0) * r_left
    r_right_mapped = 0.5 * (SPLIT_POINT + 1.0) + 0.5 * (1.0 - SPLIT_POINT) * r_right

    return np.concatenate([r_left_mapped, r_right_mapped])


# =========================================================================
# The Chebtech2 class
# =========================================================================


class Chebtech2(eqx.Module):
    """Chebyshev interpolant on 2nd-kind points.

    Represents a smooth function on [-1, 1] via Chebyshev coefficients
    (1st-kind series: T_0, T_1, ..., T_{n-1}).  This is an immutable
    Equinox module and a valid JAX pytree: the ``coeffs`` field is traced
    by JIT/vmap, while ``ishappy`` and ``epslevel`` are static.

    Attributes
    ----------
    coeffs : jax.Array, shape (n,)
        Chebyshev series coefficients c_0, c_1, ..., c_{n-1}.
    ishappy : bool
        True if the representation is resolved to the requested tolerance.
    epslevel : float
        Estimate of the relative accuracy of the representation.

    Provenance
    ----------
    MATLAB source : @chebtech2/chebtech2.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    chebfunjax.utils.transforms.vals2coeffs, chebfunjax.utils.transforms.coeffs2vals
    """

    coeffs: jax.Array
    ishappy: bool = eqx.field(static=True, default=True)
    epslevel: float = eqx.field(static=True, default=2.220446049250313e-16)

    # =================================================================
    # Construction (classmethods)
    # =================================================================

    @classmethod
    def from_coeffs(cls, coeffs: jnp.ndarray, **kwargs) -> "Chebtech2":
        """Construct from Chebyshev coefficients.

        Parameters
        ----------
        coeffs : jnp.ndarray, shape (n,)
            Chebyshev series coefficients.

        Returns
        -------
        Chebtech2
        """
        coeffs = jnp.asarray(coeffs, dtype=jnp.float64)
        return cls(coeffs=coeffs, **kwargs)

    @classmethod
    def from_values(cls, values: jnp.ndarray, **kwargs) -> "Chebtech2":
        """Construct from function values at 2nd-kind Chebyshev points.

        Parameters
        ----------
        values : jnp.ndarray, shape (n,)
            Function values at ascending Chebyshev-2 points on [-1, 1].

        Returns
        -------
        Chebtech2
        """
        values = jnp.asarray(values, dtype=jnp.float64)
        coeffs = _values_to_coeffs(values)
        return cls(coeffs=coeffs, **kwargs)

    # =================================================================
    # Evaluation
    # =================================================================

    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate at point(s) *x* in [-1, 1] via Clenshaw's algorithm.

        Parameters
        ----------
        x : jnp.ndarray
            Evaluation points.

        Returns
        -------
        jnp.ndarray
            Function values at *x*.
        """
        return _clenshaw(self.coeffs, jnp.asarray(x, dtype=jnp.float64))

    # =================================================================
    # Properties
    # =================================================================

    @property
    def n(self) -> int:
        """Number of Chebyshev points / coefficients."""
        return self.coeffs.shape[0]

    @property
    def values(self) -> jnp.ndarray:
        """Function values at ascending 2nd-kind Chebyshev points."""
        return _coeffs_to_values(self.coeffs)

    @property
    def vscale(self) -> float:
        """Estimate of the vertical scale (max absolute value)."""
        return float(jnp.max(jnp.abs(self.values)))

    def __len__(self) -> int:
        """Number of coefficients (length of the Chebyshev series)."""
        return self.n

    def __repr__(self) -> str:
        return f"Chebtech2(n={self.n}, vscale={self.vscale:.4g})"

    # =================================================================
    # Arithmetic operators
    # =================================================================

    def __add__(self, other) -> "Chebtech2":
        """Add a Chebtech2 or scalar.

        Provenance
        ----------
        MATLAB source : @chebtech/plus.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, Chebtech2):
            # Prolong to the same length (zero-pad shorter one)
            nf = self.n
            ng = other.n
            n = max(nf, ng)
            fc = _prolong_coeffs(self.coeffs, n)
            gc = _prolong_coeffs(other.coeffs, n)
            return Chebtech2.from_coeffs(fc + gc)
        else:
            # Scalar addition: only the c_0 coefficient changes
            c = self.coeffs.at[0].add(jnp.float64(other))
            return Chebtech2.from_coeffs(c)

    def __radd__(self, other) -> "Chebtech2":
        return self.__add__(other)

    def __sub__(self, other) -> "Chebtech2":
        """Subtract a Chebtech2 or scalar.

        Provenance
        ----------
        MATLAB source : @chebtech/minus.m
        Chebfun commit: 7574c77
        """
        return self + (-other)

    def __rsub__(self, other) -> "Chebtech2":
        return -(self - other)

    def __neg__(self) -> "Chebtech2":
        """Unary minus.

        Provenance
        ----------
        MATLAB source : @chebtech/uminus.m
        Chebfun commit: 7574c77
        """
        return Chebtech2.from_coeffs(-self.coeffs)

    def __pos__(self) -> "Chebtech2":
        """Unary plus (identity)."""
        return self

    def __mul__(self, other) -> "Chebtech2":
        """Pointwise multiplication.

        Chebtech2 * Chebtech2 uses coefficient-space FFT multiplication.
        Chebtech2 * scalar scales all coefficients.

        Provenance
        ----------
        MATLAB source : @chebtech/times.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, Chebtech2):
            hc = _coeff_multiply(self.coeffs, other.coeffs)
            return Chebtech2.from_coeffs(hc)
        else:
            return Chebtech2.from_coeffs(self.coeffs * jnp.float64(other))

    def __rmul__(self, other) -> "Chebtech2":
        return self.__mul__(other)

    def __truediv__(self, other) -> "Chebtech2":
        """Division: Chebtech2 / scalar or Chebtech2 / Chebtech2.

        Division by a scalar simply scales the coefficients.
        Division by another Chebtech2 evaluates on a fine grid and
        re-interpolates (NOT JIT-safe when dividing by a Chebtech2).

        Provenance
        ----------
        MATLAB source : @chebtech/rdivide.m
        Chebfun commit: 7574c77
        """
        if isinstance(other, Chebtech2):
            # Evaluate both on a fine grid and divide
            n = self.n + other.n
            from chebfunjax.utils.quadrature import chebpts

            x = chebpts(n, kind=2)
            fv = _clenshaw(self.coeffs, x)
            gv = _clenshaw(other.coeffs, x)
            return Chebtech2.from_values(fv / gv)
        else:
            return Chebtech2.from_coeffs(self.coeffs / jnp.float64(other))

    def __rtruediv__(self, other) -> "Chebtech2":
        """Scalar / Chebtech2."""
        n = max(self.n, 17)
        from chebfunjax.utils.quadrature import chebpts

        x = chebpts(n, kind=2)
        fv = jnp.float64(other) / _clenshaw(self.coeffs, x)
        return Chebtech2.from_values(fv)

    def __pow__(self, exponent) -> "Chebtech2":
        """Raise to a power.

        Integer powers via repeated multiplication.
        Non-integer powers via evaluation on a grid and re-interpolation.

        Provenance
        ----------
        MATLAB source : @chebtech/power.m
        Chebfun commit: 7574c77
        """
        if isinstance(exponent, int) and exponent >= 0:
            if exponent == 0:
                return Chebtech2.from_coeffs(jnp.array([1.0], dtype=jnp.float64))
            result = self
            for _ in range(exponent - 1):
                result = result * self
            return result
        elif isinstance(exponent, Chebtech2):
            # f^g via evaluation
            n = self.n + exponent.n
            from chebfunjax.utils.quadrature import chebpts

            x = chebpts(n, kind=2)
            fv = _clenshaw(self.coeffs, x)
            gv = _clenshaw(exponent.coeffs, x)
            return Chebtech2.from_values(fv**gv)
        else:
            # Fractional power: evaluate on a grid
            n = max(2 * self.n, 17)
            from chebfunjax.utils.quadrature import chebpts

            x = chebpts(n, kind=2)
            fv = _clenshaw(self.coeffs, x) ** jnp.float64(exponent)
            return Chebtech2.from_values(fv)

    def __abs__(self) -> "Chebtech2":
        """Absolute value (evaluated on a grid, re-interpolated).

        NOT JIT-safe (may introduce kinks).
        """
        n = max(2 * self.n, 17)
        from chebfunjax.utils.quadrature import chebpts

        x = chebpts(n, kind=2)
        fv = jnp.abs(_clenshaw(self.coeffs, x))
        return Chebtech2.from_values(fv)

    # =================================================================
    # Calculus
    # =================================================================

    def diff(self, k: int = 1) -> "Chebtech2":
        """Differentiate *k* times.

        Uses the Chebyshev coefficient recurrence (Mason & Handscomb, p. 34).

        JIT-safe: yes (k must be a static integer).

        Parameters
        ----------
        k : int, default 1
            Order of differentiation.

        Returns
        -------
        Chebtech2
            The k-th derivative.

        Provenance
        ----------
        MATLAB source : @chebtech/diff.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        Algorithm: Page 34 of Mason & Handscomb, "Chebyshev Polynomials",
            Chapman & Hall/CRC, 2003.

        See Also
        --------
        cumsum, sum
        """
        if k == 0:
            return self
        new_coeffs = _diff_coeffs(self.coeffs, k)
        return Chebtech2.from_coeffs(new_coeffs)

    def cumsum(self) -> "Chebtech2":
        """Indefinite integral (antiderivative with F(-1) = 0).

        Uses the Chebyshev coefficient recurrence.

        JIT-safe: yes.

        Returns
        -------
        Chebtech2
            The antiderivative satisfying F(-1) = 0.

        Provenance
        ----------
        MATLAB source : @chebtech/cumsum.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        Algorithm: Pages 32-33 of Mason & Handscomb, "Chebyshev Polynomials",
            Chapman & Hall/CRC, 2003.

        See Also
        --------
        diff, sum
        """
        new_coeffs = _cumsum_coeffs(self.coeffs)
        return Chebtech2.from_coeffs(new_coeffs)

    def sum(self) -> jnp.ndarray:
        """Definite integral over [-1, 1].

        Uses the Chebyshev moments: integral of T_k = 2/(1-k^2) for even k.

        JIT-safe: yes.

        Returns
        -------
        jnp.ndarray (scalar)
            The definite integral.

        Provenance
        ----------
        MATLAB source : @chebtech/sum.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        Algorithm: Trefethen, ATAP, Thm 19.2.

        See Also
        --------
        diff, cumsum, inner
        """
        return _definite_integral(self.coeffs)

    def inner(self, other: "Chebtech2") -> jnp.ndarray:
        r"""L^2 inner product <self, other> = \int_{-1}^{1} f(x) g(x) dx.

        Computed by prolonging to sum of degrees and applying Clenshaw-Curtis
        quadrature (exact for polynomials of this combined degree).

        JIT-safe: yes (shapes fixed once called).

        Parameters
        ----------
        other : Chebtech2
            The other function.

        Returns
        -------
        jnp.ndarray (scalar)
            The inner product.

        Provenance
        ----------
        MATLAB source : @chebtech/innerProduct.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        sum, norm
        """
        return _inner_product(self.coeffs, other.coeffs)

    def norm(self, p: float = 2.0) -> jnp.ndarray:
        """Lp norm of the Chebtech2.

        Parameters
        ----------
        p : float, default 2.0
            The exponent for the Lp norm.
            - ``p=2``: L2 norm via inner product (= sqrt(<f, f>)).
            - ``p=jnp.inf``: L-infinity norm via max of |values| on a fine grid.
            - Other p: computed via quadrature of |f|^p.

        Returns
        -------
        jnp.ndarray (scalar)

        Provenance
        ----------
        MATLAB source : @chebtech/normest.m (and norm.m at the chebfun level)
        Chebfun commit: 7574c77
        """
        if p == 2:
            return jnp.sqrt(jnp.abs(self.inner(self)))
        elif p == jnp.inf or p == float("inf"):
            # Sample on a fine grid
            n = max(2 * self.n + 1, 65)
            x = jnp.linspace(-1.0, 1.0, n, dtype=jnp.float64)
            return jnp.max(jnp.abs(_clenshaw(self.coeffs, x)))
        else:
            # General Lp: integrate |f|^p via (f^p).sum()
            fp = self.__abs__().__pow__(p)
            return fp.sum() ** (1.0 / p)

    # =================================================================
    # Rootfinding
    # =================================================================

    def roots(self) -> jnp.ndarray:
        """Real roots in [-1, 1] via colleague matrix eigenvalues.

        NOT JIT-safe (variable output size, recursive subdivision).

        Returns
        -------
        jnp.ndarray, shape (n_roots,)
            Sorted roots in [-1, 1].

        Provenance
        ----------
        MATLAB source : @chebtech/roots.m
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.
        Algorithm:
            [1] I. J. Good, "The colleague matrix, a Chebyshev analogue of the
                companion matrix", QJM 12, 1961.
            [2] L. N. Trefethen, ATAP, SIAM, 2013, Chapter 18.

        See Also
        --------
        diff, sum
        """
        return _roots_colleague(self.coeffs)

    # =================================================================
    # Prolongation
    # =================================================================

    def prolong(self, n: int) -> "Chebtech2":
        """Return a Chebtech2 of length *n* (truncate or zero-pad)."""
        return Chebtech2.from_coeffs(_prolong_coeffs(self.coeffs, n))
