# uses-numpy: random coefficient generation uses numpy's RNG
"""Random smooth functions on intervals, the disk, and the sphere.

Translated from MATLAB Chebfun (commit 7574c77): smoothie.m, randnfundisk.m,
randnfunsphere.m, randnfun.m.
Original: Copyright 2017-2020 by The University of Oxford and The Chebfun
Developers.  See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# ===========================================================================
# randnfun — band-limited random function on an interval
# ===========================================================================


def randnfun(
    lam: float,
    domain: tuple[float, float] = (0.0, 1.0),
    *,
    seed: int | None = None,
    big: bool = False,
) -> object:
    """Band-limited random function on an interval.

    Returns a callable ``f(t)`` that evaluates a smooth random function with
    minimal wavelength ``lam`` on the given ``domain``.  The function is a
    trigonometric polynomial with random Gaussian coefficients for wavenumbers
    ``|k| <= round(L/lam)``, where ``L = domain[1] - domain[0]``.

    Parameters
    ----------
    lam : float
        Minimum wavelength of the random function.  Smaller ``lam`` gives
        higher-frequency randomness.
    domain : (a, b), default (0, 1)
        Interval on which the function is defined.
    seed : int or None, default None
        Seed for the numpy RNG to give reproducible results.
    big : bool, default False
        If True, use the "big" normalization: multiply the result by
        ``sqrt(L/lam)`` so that the variance of ``f`` is approximately 1
        regardless of ``lam``.  This is the normalization that makes the
        integral (Brownian motion) have variance proportional to time.

    Returns
    -------
    f : callable
        A function ``f(t)`` where ``t`` can be a float or numpy array.

    Notes
    -----
    Implements the MATLAB Chebfun ``randnfun(lambda, domain, 'big')`` function.
    The random function is the real part of a trigonometric polynomial:

        f(t) = sum_{k=-N}^{N} c_k * exp(2*pi*i*k*t / L)

    where ``c_k`` are i.i.d. standard Gaussian and ``N = round(L/lam)``.
    With ``big=True``, the result is multiplied by ``sqrt(2*N)`` to give
    unit variance.

    Provenance
    ----------
    MATLAB source : randnfun.m
    Original authors: Nick Trefethen, Kevin Burrage (May 2017)
    Chebfun commit: 7574c77
    """
    a, b = float(domain[0]), float(domain[1])
    L = b - a
    if L <= 0:
        raise ValueError(f"domain must have positive length, got {domain}")

    # Number of wavenumbers
    N = max(1, int(round(L / lam)))

    rng = np.random.default_rng(seed)

    # Random Fourier coefficients for wavenumbers -N..N (except k=0)
    # Use complex Gaussian for k=1..N, then impose Hermitian symmetry for real result
    c_real = rng.standard_normal(N + 1)
    c_imag = rng.standard_normal(N + 1)
    c_imag[0] = 0.0  # k=0 must be real

    # Normalization:
    # Without 'big': each path has std ~ 1 (divide by sqrt(2*N+1))
    # With 'big': std ~ sqrt(N) ~ 1/sqrt(lambda) (the 'big' normalization of MATLAB Chebfun)
    #   This ensures the integral (Brownian path) has var proportional to time
    if big:
        norm = 1.0  # no division: each mode has unit std, total std ~ sqrt(2N+1)
    else:
        norm = 1.0 / np.sqrt(2 * N + 1)  # normalize to unit std

    c_complex = (c_real + 1j * c_imag) * norm

    def f(t: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the random function at point(s) t."""
        t_arr = np.asarray(t, dtype=float)
        scalar = t_arr.ndim == 0
        t_arr = np.atleast_1d(t_arr)

        # f(t) = Re[ sum_{k=-N}^{N} c_k * exp(2*pi*i*k*(t-a)/L) ]
        result = np.full_like(t_arr, c_complex[0].real, dtype=float)
        phase = 2.0 * np.pi * (t_arr - a) / L
        for k in range(1, N + 1):
            result += 2.0 * (c_complex[k].real * np.cos(k * phase)
                             - c_complex[k].imag * np.sin(k * phase))
        if scalar:
            return float(result[0])
        return result

    return f


# ===========================================================================
# Smoothie
# ===========================================================================


def smoothie(
    n: int = 1,
    key: jax.Array | None = None,
    *,
    domain: tuple[float, float] = (-1.0, 1.0),
    trig: bool = False,
) -> jnp.ndarray:
    """Random smooth (C-infinity but not analytic) function coefficients.

    Returns the Fourier/Chebyshev coefficients for a function that is
    C-infinity but not analytic, with root-exponentially decaying random
    Fourier coefficients.

    Parameters
    ----------
    n : int, default 1
        Number of independent function samples (columns).
    key : jax.Array or None
        JAX PRNG key.  If None, uses numpy's global RNG (non-reproducible).
    domain : (a, b), default (-1, 1)
        Interval for the function.
    trig : bool, default False
        If True, return a periodic (trigonometric) smoothie as Fourier
        coefficients.  If False, return Chebyshev coefficients for the
        non-periodic case (obtained by restricting a periodic smoothie to
        an interval 20% shorter).

    Returns
    -------
    coeffs : jnp.ndarray, shape (m,) or (m, n)
        Coefficients.  For ``trig=True``, Fourier coefficients (length
        2*m_trig-1 with conjugate symmetry for real output).  For
        ``trig=False``, Chebyshev coefficients at 2nd-kind points of the
        original domain.

    Notes
    -----
    Coefficients have root-exponential decay: c[k] ~ exp(-sqrt(k/L)) * randn,
    where L = b - a.  The function is real (imaginary part negligible).

    Provenance
    ----------
    MATLAB source : smoothie.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2020 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    randnfundisk, randnfunsphere
    """
    a, b = float(domain[0]), float(domain[1])
    L = b - a

    if L <= 0:
        raise ValueError(f"domain must have positive length, got {domain}")

    m = int(round(np.ceil(2000 * L))) + 1

    if key is not None:
        key_np = np.array(jax.random.normal(key, shape=(m, n if n > 1 else 1)), dtype=np.float64)
        key2_np = np.array(jax.random.normal(
            jax.random.fold_in(key, 1), shape=(m, n if n > 1 else 1)), dtype=np.float64)
        c_real = key_np
        c_imag = key2_np
    else:
        rng = np.random.default_rng()
        c_real = rng.standard_normal((m, n if n > 1 else 1))
        c_imag = rng.standard_normal((m, n if n > 1 else 1))

    # Random Fourier coefficients with root-exponential decay
    c = (c_real + 1j * c_imag)
    c[0, :] = np.sqrt(2) * np.real(c[0, :])
    decay = np.exp(-np.sqrt(np.arange(1, m + 1) / L))
    c = (decay[:, None] * c) / np.sqrt(L)

    # Symmetrize for real result: c[-k] = conj(c[k])
    c_sym = np.concatenate([np.conj(c[::-1, :]), c], axis=0)  # length 2m-1

    if n == 1:
        c_sym = c_sym[:, 0]

    if trig:
        return jnp.array(c_sym, dtype=jnp.complex128)

    # Non-periodic (MATLAB smoothie.m): build the periodic smoothie on
    # a 20%-longer interval, evaluate at 2nd-kind Chebyshev points of
    # the target domain, and convert to Chebyshev coefficients.  The
    # previous code returned the raw Fourier coefficients here while
    # the docstring promised Chebyshev coefficients.
    L2 = 1.2 * L
    m2 = int(round(np.ceil(2000 * L2))) + 1
    if key is not None:
        cr = np.array(jax.random.normal(
            key, shape=(m2, n if n > 1 else 1)), dtype=np.float64)
        ci = np.array(jax.random.normal(
            jax.random.fold_in(key, 1),
            shape=(m2, n if n > 1 else 1)), dtype=np.float64)
    else:
        rng = np.random.default_rng()
        cr = rng.standard_normal((m2, n if n > 1 else 1))
        ci = rng.standard_normal((m2, n if n > 1 else 1))
    c2 = cr + 1j * ci
    c2[0, :] = np.sqrt(2) * np.real(c2[0, :])
    decay2 = np.exp(-np.sqrt(np.arange(1, m2 + 1) / L2))
    c2 = (decay2[:, None] * c2) / np.sqrt(L2)
    # evaluate the real periodic series sum Re(c_k e^{2 pi i k t/L2})
    npts = int(round(2.5 * m2)) + 20
    tk = np.cos(np.pi * np.arange(npts - 1, -1, -1) / (npts - 1))
    xk = a + (b - a) * (tk + 1) / 2
    theta = 2 * np.pi * (xk - a) / L2
    ks = np.arange(1, m2)
    E = np.exp(1j * np.outer(theta, ks))
    # real series: f = c_0 + 2 sum_{k>=1} Re(c_k e^{ik theta})
    vals = (np.real(c2[0, :])[None, :]
            + 2 * np.real(E @ c2[1:, :]))
    # values -> Chebyshev coefficients (DCT-I)
    def _v2c(v):
        nn = len(v)
        tmp = np.concatenate([v[nn - 1:0:-1], v[:nn - 1]])
        cc = np.real(np.fft.ifft(tmp))[:nn]
        cc[1:nn - 1] *= 2.0
        return cc
    cols = []
    for j in range(vals.shape[1]):
        cc = _v2c(vals[:, j])
        acc = np.abs(cc)
        mx = acc.max() if acc.size else 0.0
        if mx > 0:
            nz = np.where(acc > 1e-15 * mx)[0]
            cc = cc[:nz[-1] + 1] if nz.size else cc[:1]
        cols.append(cc)
    if n == 1:
        return jnp.array(cols[0], dtype=jnp.float64)
    mlen = max(len(cv) for cv in cols)
    out = np.zeros((mlen, n))
    for j, cv in enumerate(cols):
        out[:len(cv), j] = cv
    return jnp.array(out, dtype=jnp.float64)


# ===========================================================================
# Random function on disk
# ===========================================================================


def randnfundisk(
    n: int,
    key: jax.Array | None = None,
    *,
    lam: float = 1.0,
) -> jnp.ndarray:
    """Random smooth function on the unit disk (polar grid values).

    Returns values of a random smooth function on the unit disk, sampled
    on a polar tensor product grid (r, theta).  The maximum frequency is
    approximately 2*pi/lam.

    Parameters
    ----------
    n : int
        Resolution parameter.  The output grid has shape (n_r, n_theta) where
        n_r and n_theta are chosen automatically based on n and lam.
    key : jax.Array or None
        JAX PRNG key.  If None, uses numpy global RNG.
    lam : float, default 1.0
        Length scale.  Smaller lam means higher-frequency randomness.

    Returns
    -------
    F : jnp.ndarray, shape (n_r, n_theta)
        Function values on the polar grid.  F[i, j] = f(r[i], theta[j]).

    Notes
    -----
    Implements the method from MATLAB Chebfun's randnfundisk: generates
    a random 2D function on a square via a Fourier-Wiener series, then
    restricts to the unit disk by sampling on a polar grid.

    Provenance
    ----------
    MATLAB source : randnfundisk.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    smoothie, randnfunsphere
    """
    # Resolution: m = max wave number ≈ n / lam
    m = max(2, int(round(n / lam)))

    n_r = max(4, 2 * m + 1)
    n_theta = max(4, 4 * m)
    n_theta = n_theta + (n_theta % 2)  # make even

    # Generate random 2D Fourier coefficients
    if key is not None:
        c = np.array(jax.random.normal(key, shape=(2 * m + 1, 2 * m + 1)), dtype=np.float64)
        cs = np.array(jax.random.normal(
            jax.random.fold_in(key, 2), shape=(2 * m + 1, 2 * m + 1)), dtype=np.float64)
    else:
        rng = np.random.default_rng()
        c = rng.standard_normal((2 * m + 1, 2 * m + 1))
        cs = rng.standard_normal((2 * m + 1, 2 * m + 1))

    # Fourier-Wiener series: f(x,y) = sum_{|k|,|l|<=m} c[k,l] * exp(i*(kx+ly)*2pi/L)
    # where L = 2.5 (domain is 1.25*[-1,1])
    L = 2.5  # 1.25 * 2
    kk = np.arange(-m, m + 1, dtype=np.float64)
    decay = 1.0 / (2 * m + 1)

    # Polar grid on unit disk
    r_vals = np.linspace(0, 1, n_r + 1)[1:]  # avoid r=0
    theta_vals = np.linspace(-np.pi, np.pi, n_theta + 1)[:-1]

    # Convert to Cartesian
    rr, tt = np.meshgrid(r_vals, theta_vals, indexing='ij')
    xx = rr * np.cos(tt)
    yy = rr * np.sin(tt)

    # Evaluate 2D Fourier series on the disk
    F = np.zeros((n_r, n_theta), dtype=np.float64)
    for ki, k in enumerate(kk):
        for li, l_val in enumerate(kk):
            phase = (c[ki, li] * np.cos((k * xx + l_val * yy) * 2 * np.pi / L)
                     - cs[ki, li] * np.sin((k * xx + l_val * yy) * 2 * np.pi / L))
            F += decay * phase

    return jnp.array(F, dtype=jnp.float64)


# ===========================================================================
# Random function on sphere
# ===========================================================================


def randnfunsphere(
    n: int,
    key: jax.Array | None = None,
    *,
    lam: float = 1.0,
    monochromatic: bool = False,
) -> jnp.ndarray:
    """Random smooth function on the unit sphere (longitude-latitude grid).

    Returns values of a random smooth function on S^2, sampled on a
    (lambda, theta) tensor product grid.  The maximum frequency is
    approximately 2*pi/lam.

    Parameters
    ----------
    n : int
        Resolution parameter.  The maximum spherical harmonic degree is
        approximately n.
    key : jax.Array or None
        JAX PRNG key.  If None, uses numpy global RNG.
    lam : float, default 1.0
        Length scale parameter.  deg = floor(2*pi/lam) is the max degree.
    monochromatic : bool, default False
        If True, use only the single degree deg.

    Returns
    -------
    F : jnp.ndarray, shape (n_theta, n_lambda)
        Function values on the spherical grid.  Rows index theta (colatitude
        in [0,pi]), columns index lambda (longitude in [-pi,pi]).

    Notes
    -----
    Implements a combination of spherical harmonics with random Gaussian
    coefficients, matching MATLAB Chebfun's randnfunsphere.

    Provenance
    ----------
    MATLAB source : randnfunsphere.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    smoothie, randnfundisk
    """
    deg = max(1, int(np.floor(2 * np.pi / lam)))
    deg = min(deg, n)

    # Sampling grid
    n_lambda = 2 * deg + 2
    n_theta = 2 * deg + 2

    lam_grid = np.linspace(-np.pi, np.pi, n_lambda + 1)[:-1]
    theta_grid = np.linspace(0, np.pi, n_theta + 1)[1:]  # avoid poles

    if monochromatic:
        n_coeffs = 2 * deg + 1
    else:
        n_coeffs = (deg + 1) ** 2

    if key is not None:
        coeffs = np.array(jax.random.normal(key, shape=(n_coeffs,)), dtype=np.float64)
    else:
        rng = np.random.default_rng()
        coeffs = rng.standard_normal(n_coeffs)

    # Normalize
    coeffs = coeffs * np.sqrt(4 * np.pi / max(1, n_coeffs - n_coeffs // 2))

    F = np.zeros((n_theta, n_lambda), dtype=np.float64)

    if monochromatic:
        F = _sph_harm_sum_fixed_deg(lam_grid, theta_grid, deg, coeffs)
    else:
        F = _sph_harm_sum(lam_grid, theta_grid, deg, coeffs)

    return jnp.array(F, dtype=jnp.float64)


def _sph_harm_sum(
    lam: np.ndarray,
    theta: np.ndarray,
    deg: int,
    coeffs: np.ndarray,
) -> np.ndarray:
    """Sum of spherical harmonics up to degree deg over a tensor grid."""
    from scipy.special import lpmv

    cos_theta = np.cos(theta)  # (n_theta,)
    F = np.zeros((len(theta), len(lam)), dtype=np.float64)

    c_idx = 0
    # l=0 term
    c = coeffs[c_idx]
    c_idx += 1
    F += (1.0 / np.sqrt(4 * np.pi)) * c

    for l_deg in range(1, deg + 1):
        m_vals = np.arange(l_deg + 1)
        # Normalization: a[m] = (-1)^m / sqrt((1 + delta_{m,0}) * pi)
        a = ((-1.0) ** m_vals) / np.sqrt((1.0 + (m_vals == 0).astype(float)) * np.pi)

        # Associated Legendre: G[m, theta]
        import math
        G = np.zeros((l_deg + 1, len(theta)))
        for m_idx, m_val in enumerate(m_vals):
            # lpmv(m, l, x) = P_l^m(x) (unnormalized)
            # normalized: sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!) * P_l^m
            norm = np.sqrt((2 * l_deg + 1) / (4 * np.pi)
                           * math.factorial(l_deg - m_val)
                           / math.factorial(l_deg + m_val))
            G[m_idx, :] = norm * lpmv(m_val, l_deg, cos_theta)

        # Extract coefficients for this degree
        n_this = 2 * l_deg + 1
        c_this = coeffs[c_idx:c_idx + n_this]
        c_idx += n_this

        # Positive orders (including m=0)
        c_pos = a * c_this[l_deg:]  # (l_deg+1,)
        # Negative orders
        c_neg = a[1:] * c_this[:l_deg][::-1]  # (l_deg,)

        # Tensor product
        Gp = G  # (l_deg+1, n_theta)
        Gn = G[1:, :]  # (l_deg, n_theta)

        for m_idx in range(l_deg + 1):
            F += np.outer(c_pos[m_idx] * Gp[m_idx, :],
                          np.cos(m_vals[m_idx] * lam))

        for m_idx in range(l_deg):
            F += np.outer(c_neg[m_idx] * Gn[m_idx, :],
                          np.sin((m_idx + 1) * lam))

    return F


def _sph_harm_sum_fixed_deg(
    lam: np.ndarray,
    theta: np.ndarray,
    l_deg: int,
    coeffs: np.ndarray,
) -> np.ndarray:
    """Sum of spherical harmonics of a single fixed degree."""
    import math

    from scipy.special import lpmv

    cos_theta = np.cos(theta)
    F = np.zeros((len(theta), len(lam)), dtype=np.float64)

    m_vals = np.arange(l_deg + 1)
    a = ((-1.0) ** m_vals) / np.sqrt((1.0 + (m_vals == 0).astype(float)) * np.pi)

    G = np.zeros((l_deg + 1, len(theta)))
    for m_idx, m_val in enumerate(m_vals):
        norm = np.sqrt((2 * l_deg + 1) / (4 * np.pi)
                       * math.factorial(l_deg - m_val)
                       / math.factorial(l_deg + m_val))
        G[m_idx, :] = norm * lpmv(m_val, l_deg, cos_theta)

    Gp = G
    Gn = G[1:, :]

    c_pos = a * coeffs[l_deg:]
    c_neg = a[1:] * coeffs[:l_deg][::-1]

    for m_idx in range(l_deg + 1):
        F += np.outer(c_pos[m_idx] * Gp[m_idx, :],
                      np.cos(m_vals[m_idx] * lam))

    for m_idx in range(l_deg):
        F += np.outer(c_neg[m_idx] * Gn[m_idx, :],
                      np.sin((m_idx + 1) * lam))

    return F


def randnfun2(
    lam: float,
    domain: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
    *,
    seed: int | None = None,
    big: bool = False,
    trig: bool = False,
):
    """Band-limited random Chebfun2 with minimal wavelength ``lam``
    (MATLAB randnfun2).

    Random Fourier coefficients with unit variance are confined to an
    elliptical disc of wavenumbers (for isotropy) and normalized so the
    pointwise variance is ~1.  The non-periodic case samples the
    periodic series on a padded domain, so translated domains with the
    same seed and aspect give the same (shifted) function -- the same
    property the MATLAB rng-based version has.

    Parameters
    ----------
    lam : float
        Minimal wavelength.
    domain : (xa, xb, ya, yb)
    seed : int or None
        Numpy RNG seed for reproducibility.
    big : bool
        Divide by sqrt(lam) (2D white-noise normalization).
    trig : bool
        Doubly periodic function.

    Provenance
    ----------
    MATLAB source : randnfun2.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    from chebfunjax.chebfun2d.chebfun2 import Chebfun2
    xa, xb, ya, yb = (float(v) for v in domain)
    if not trig:
        if np.isinf(lam):
            rng = np.random.default_rng(seed)
            c = float(rng.standard_normal())
            if big:
                c = 0.0
            return Chebfun2.from_function(
                lambda x, y: c + 0.0 * x, domain=domain)
        # periodic construction on a padded domain, then restrict
        m = int(round(1.2 * (xb - xa) / lam + 2))
        n = int(round(1.2 * (yb - ya) / lam + 2))
        dom2 = (xa, xa + m * lam, ya, ya + n * lam)
        f2 = randnfun2(lam, dom2, seed=seed, big=big, trig=True)
        return f2.restrict(domain)

    Lx = xb - xa
    Ly = yb - ya
    m = int(round(Lx / lam))
    n = int(round(Ly / lam))
    rng = np.random.default_rng(seed)
    kx, ky = np.meshgrid(np.arange(-m, m + 1), np.arange(-n, n + 1))
    c = (rng.standard_normal(kx.shape)
         + 1j * rng.standard_normal(kx.shape))
    if m > 0 and n > 0:
        c = c * (((kx / m) ** 2 + (ky / n) ** 2) <= 1)
    nnz = np.count_nonzero(c)
    c = c / np.sqrt(max(nnz, 1))
    if big:
        c = c / np.sqrt(lam)

    kxu = np.arange(-m, m + 1)
    kyu = np.arange(-n, n + 1)

    def f(x, y):
        # real part of the double Fourier series (vectorised):
        # Re( sum_ij c[i,j] exp(i*(ky_i*py + kx_j*px)) )
        px = 2.0 * np.pi * (np.asarray(x, dtype=float) - xa) / Lx
        py = 2.0 * np.pi * (np.asarray(y, dtype=float) - ya) / Ly
        px, py = np.broadcast_arrays(px, py)
        shp = px.shape
        Ex = np.exp(1j * px.ravel()[:, None] * kxu[None, :])
        Ey = np.exp(1j * py.ravel()[:, None] * kyu[None, :])
        out = np.einsum("pi,ij,pj->p", Ey, c, Ex).real
        return jnp.asarray(out.reshape(shp), dtype=jnp.float64)

    return Chebfun2.from_function(f, domain=domain)
