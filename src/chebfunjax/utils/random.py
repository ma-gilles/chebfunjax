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


def randnfun(*args, **kwargs):
    """Construct the source-shaped random Chebfun through the shared JAX engine.

    NumPy global seeding no longer controls this route. Use key= or seed= for
    explicit JAX reproducibility; default calls advance a private JAX stream.

    Provenance
    ----------
    MATLAB source : randnfun.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    from chebfunjax.utils._randnfun import randnfun as source_randnfun
    return source_randnfun(*args, **kwargs)


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

    # Oversample the output grid relative to the band limit so the
    # returned samples resolve the smooth field (2 samples/mode was
    # angularly under-resolved, rendering as pixel noise).
    n_r = max(8, 4 * m + 1)
    n_theta = max(16, 8 * m)
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

    # Evaluate the 2D Fourier series on the disk, vectorized:
    #   c*cos(A) - cs*sin(A) = Re[(c + i*cs) * e^{iA}],
    #   e^{iA} = e^{2*pi*i*k*x/L} * e^{2*pi*i*l*y/L},
    # so F[p] = Re( U[p,:] @ W @ V[p,:].T ) with mode matrices U, V.
    # (The previous per-mode Python loop was O((2m+1)^2) full-grid
    # trig evaluations — minutes at lam = 0.1.)
    W = c + 1j * cs
    xf = xx.ravel()
    yf = yy.ravel()
    U = np.exp(2j * np.pi * np.outer(xf, kk) / L)
    V = np.exp(2j * np.pi * np.outer(yf, kk) / L)
    Fflat = np.real(np.einsum('pk,pk->p', U @ W, V))
    F = decay * Fflat.reshape(n_r, n_theta)

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


def _norm_legendre(l_deg: int, m: int, theta: np.ndarray) -> jax.Array:
    """MATLAB fully normalized associated Legendre function (no CS phase).

    The existing SciPy spherical harmonic evaluator includes the
    Condon--Shortley factor. MATLAB ``legendre(n,x,'norm')`` cancels that
    factor, so the harmonic value needs an additional ``(-1)**m``.
    This preserves the stable degree-210 evaluator used by SpherefunRotate;
    replacing the existing SciPy dependency remains separate work.

    Provenance
    ----------
    MATLAB source : randnfunsphere.m, sphHarmSum/sphHarmSumFixedDeg
    Chebfun commit: 7574c77
    Normalization: MATLAB legendre(..., 'norm') definition.
    """
    from scipy.special import sph_harm_y
    cs_value = np.sqrt(2.0 * np.pi) * np.real(sph_harm_y(l_deg, m, np.asarray(theta), 0.0))
    return jnp.asarray(cs_value) * (-1 if m % 2 else 1)


def _sph_harm_sum(
    lam: np.ndarray,
    theta: np.ndarray,
    deg: int,
    coeffs: np.ndarray,
) -> np.ndarray:
    """Sum of spherical harmonics up to degree deg over a tensor grid."""

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
        G = np.zeros((l_deg + 1, len(theta)))
        for m_idx, m_val in enumerate(m_vals):
            # normalized: sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!) * P_l^m
            G[m_idx, :] = _norm_legendre(l_deg, int(m_val), theta)

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


    F = np.zeros((len(theta), len(lam)), dtype=np.float64)

    m_vals = np.arange(l_deg + 1)
    a = ((-1.0) ** m_vals) / np.sqrt((1.0 + (m_vals == 0).astype(float)) * np.pi)

    G = np.zeros((l_deg + 1, len(theta)))
    for m_idx, m_val in enumerate(m_vals):
        G[m_idx, :] = _norm_legendre(l_deg, int(m_val), theta)

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
    *args,
    seed: int | None = None,
    lam: float | None = None,
    domain: tuple[float, float, float, float] | None = None,
    big: bool | None = None,
    trig: bool | None = None,
):
    """Construct a smooth random Chebfun2 using the native source route.

    Positional inputs follow ``randnfun2.m``: scalar values set the wavelength,
    nonscalars set ``[xa, xb, ya, yb]``, and strings beginning with ``n``/``b``
    or ``t`` select ``big`` or trigonometric construction.  The MATLAB random
    stream is not reproduced; ``seed`` is a deterministic JAX PRNG adapter.

    Provenance
    ----------
    MATLAB source : randnfun2.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford and The
        Chebfun Developers.
    """
    import math

    from chebfunjax.chebfun2d.chebfun2 import Chebfun2
    from chebfunjax.utils import _randnfun

    supplied = list(args)
    if lam is not None:
        supplied.append(lam)
    if domain is not None:
        supplied.append(domain)

    wavelength = math.nan
    dom = None
    makebig = False
    periodic = False
    for value in supplied:
        if isinstance(value, str):
            first = value[:1]
            if first in ('n', 'b'):
                makebig = True
            elif first == 't':
                periodic = True
            else:
                raise ValueError('CHEBFUN:randnfun2: Unrecognized string input')
            continue
        array = jnp.asarray(value)
        if array.size != 1:
            dom = tuple(float(v) for v in array.reshape(-1))
        else:
            wavelength = float(array.reshape(()))

    if big is not None:
        makebig = bool(big)
    if trig is not None:
        periodic = bool(trig)
    if math.isnan(wavelength):
        wavelength = 1.0
    if dom is None or (len(dom) == 4 and all(math.isnan(v) for v in dom)):
        dom = (-1.0, 1.0, -1.0, 1.0)
    if len(dom) != 4:
        raise ValueError('CHEBFUN:randnfun2: domain must have four endpoints')
    xa, xb, ya, yb = dom
    if not (xb > xa and yb > ya):
        raise ValueError('CHEBFUN:randnfun2: domain endpoints must be increasing')
    if not (wavelength > 0.0):
        raise ValueError('CHEBFUN:randnfun2: wavelength must be positive')

    def draw_pair(rows, columns):
        if seed is None:
            key_re = _randnfun._next_key()
            key_im = _randnfun._next_key()
        else:
            seed_key = jax.random.key(int(seed))
            key_re, key_im = jax.random.split(seed_key)
        return (_randnfun._normal_draw(key_re, rows, columns)
                + 1j * _randnfun._normal_draw(key_im, rows, columns))

    def draw_real_scalar():
        if seed is None:
            key = _randnfun._next_key()
        else:
            key, _ = jax.random.split(jax.random.key(int(seed)))
        return _randnfun._normal_draw(key, 1, 1)[0, 0].real

    if not periodic:
        if math.isinf(wavelength):
            value = draw_real_scalar()
            if makebig:
                value = value / math.sqrt(wavelength)
            return Chebfun2.from_function(
                lambda x, y: value + 0.0 * x, domain=dom)

        # MATLAB pads the requested box by a wavelength-scaled margin, builds
        # the periodic source function there, then restricts in that order.
        m_pad = int(math.floor(1.2 * (xb - xa) / wavelength + 2.5))
        n_pad = int(math.floor(1.2 * (yb - ya) / wavelength + 2.5))
        padded = (xa, xa + m_pad * wavelength,
                  ya, ya + n_pad * wavelength)
        f = randnfun2(wavelength, padded, seed=seed,
                      big=makebig, trig=True)
        return f.restrict(dom)

    Lx = xb - xa
    Ly = yb - ya
    m = int(math.floor(Lx / wavelength + 0.5))
    n = int(math.floor(Ly / wavelength + 0.5))
    c = draw_pair(2 * n + 1, 2 * m + 1)
    kx = jnp.arange(-m, m + 1)[None, :]
    ky = jnp.arange(-n, n + 1)[:, None]
    if m > 0 and n > 0:
        c = jnp.where((kx / m) ** 2 + (ky / n) ** 2 <= 1.0, c, 0.0)
    c = c / jnp.sqrt(jnp.count_nonzero(c))

    # Keep the native construction order: centered trig coefficients, real
    # projection, big normalization, and (for nonperiodic mode) restriction.
    f = Chebfun2.from_trig_coeffs(c, domain=dom).real()
    if makebig:
        f = f / math.sqrt(wavelength)
    return f
