"""Zero-curve rootfinding for Chebfun2 (marching squares + Newton polish).

Traces the zero level set of a real Chebfun2 as a set of complex-valued
parametrized Chebfun curves ``c(t) = x(t) + 1i*y(t)``, ``t in [-1, 1]``.
This is the engine behind :meth:`Chebfun2.roots` and, via coordinate
wrapping, Diskfun/Spherefun roots.

The algorithm mirrors MATLAB ``@separableApprox/roots.m``:

1. Rank-1 shortcut -- a separable ``f`` vanishes on horizontal/vertical
   lines given by the roots of its 1D column/row slices; these are emitted
   with the exact MATLAB line parametrization.
2. Otherwise a dense grid is sampled and marching squares
   (``skimage.measure.find_contours``) locates the zero level set; each
   traced polyline is refined to near machine precision by an
   arclength-reparametrized Chebfun fit with a complex-Newton polish
   (``p <- p - f/conj(grad f)``) repeated until the residual stops
   improving.

Provenance
----------
MATLAB source : @separableApprox/roots.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import Chebfun, Domain

__all__ = ["zero_curves"]


def _chebpts(n: int) -> np.ndarray:
    """Chebyshev points of the 2nd kind on [-1, 1], ascending."""
    if n == 1:
        return np.array([0.0])
    k = np.arange(n)
    return np.sort(np.cos(np.pi * k / (n - 1)))


def _grid_size(f) -> int:
    """Odd grid size resolving f's slices; MATLAB uses n = 502."""
    try:
        deg = max(
            max(len(c.coeffs) for c in f.approx.cols),
            max(len(r.coeffs) for r in f.approx.rows),
        )
    except Exception:
        deg = 100
    return int(min(max(4 * deg + 1, 769), 2049)) | 1


def _rank1_curves(f):
    """Zero curves of a rank-1 (separable) Chebfun2.

    ``f`` vanishes where a column slice (function of y) or a row slice
    (function of x) vanishes -- i.e. on horizontal / vertical lines.  The
    parametrization matches MATLAB @separableApprox/roots.m exactly:
    a horizontal line at height ``y0`` is ``t -> X(t) + 1i*y0`` with the
    real part sweeping the x-domain, and a vertical line at ``x0`` is
    ``t -> x0 + 1i*Y(t)``.
    """
    xa, xb, ya, yb = (float(v) for v in f.domain)

    # The single column (function of y) and row (function of x).
    col = f.approx.cols[0]
    row = f.approx.rows[0]

    # Build 1D Chebfuns for the slices from their Chebyshev coefficients.
    def _slice_chebfun(tech, a, b):
        return Chebfun.from_coeffs(jnp.asarray(tech.coeffs), Domain((a, b)))

    col_fun = _slice_chebfun(col, ya, yb)
    row_fun = _slice_chebfun(row, xa, xb)

    y_roots = np.atleast_1d(np.asarray(col_fun.roots()))
    x_roots = np.atleast_1d(np.asarray(row_fun.roots()))
    y_roots = np.sort(y_roots[np.isfinite(y_roots)])
    x_roots = np.sort(x_roots[np.isfinite(x_roots)])

    curves = []
    # Horizontal lines (column slice zero): real part sweeps [xa, xb].
    lenx = 0.5 * (xb - xa)
    for y0 in y_roots:
        curves.append(Chebfun.from_function(
            lambda t, _y=float(y0): (lenx * (t + 1.0) + xa) + 1j * _y,
            Domain((-1.0, 1.0))))
    # Vertical lines (row slice zero): imag part sweeps [ya, yb].
    leny = 0.5 * (yb - ya)
    for x0 in x_roots:
        curves.append(Chebfun.from_function(
            lambda t, _x=float(x0): _x + 1j * (leny * (t + 1.0) + ya),
            Domain((-1.0, 1.0))))
    return curves


def _marching_squares(f, n: int):
    """Zero level-set polylines of f via skimage marching squares.

    Returns a list of ``(m, 2)`` arrays of physical ``(x, y)`` samples,
    one per traced contour.  The grid excludes the domain boundary
    (matching MATLAB's ``n = 502`` interior grid) to avoid edge artefacts.
    """
    from skimage import measure

    xa, xb, ya, yb = (float(v) for v in f.domain)
    x = np.linspace(xa, xb, n)
    y = np.linspace(ya, yb, n)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    vals = np.asarray(f(jnp.asarray(xx), jnp.asarray(yy)), dtype=np.float64)

    contours = measure.find_contours(vals, 0.0)
    out = []
    nx = x.shape[0]
    ny = y.shape[0]
    for c in contours:
        # c columns are (row, col) = (y-index, x-index), fractional.
        ri = c[:, 0]
        ci = c[:, 1]
        px = x[0] + ci * (x[-1] - x[0]) / (nx - 1)
        py = y[0] + ri * (y[-1] - y[0]) / (ny - 1)
        pts = np.column_stack([px, py])
        if pts.shape[0] >= 3:
            out.append(pts)
    return out


def _simplify(cf):
    """Chop interpolation noise from a fitted real Chebfun so a smooth
    zero curve collapses to its true (low) polynomial degree; the raw
    interpolant through hundreds of arclength samples otherwise carries a
    Runge-oscillation tail that inflates the arc-length integral."""
    tech = cf.funs[0].tech.simplify()
    return Chebfun.from_coeffs(jnp.asarray(tech.coeffs), cf.domain)


def _snap(dc, dom, tol):
    """Snap points within ``tol`` of a domain edge exactly onto it and clip
    any Newton overshoot back into the closed domain (MATLAB ``snap``)."""
    xa, xb, ya, yb = dom
    rx = np.real(dc).copy()
    ry = np.imag(dc).copy()
    rx[np.abs(rx - xa) < tol] = xa
    rx[np.abs(rx - xb) < tol] = xb
    ry[np.abs(ry - ya) < tol] = ya
    ry[np.abs(ry - yb) < tol] = yb
    rx = np.clip(rx, xa, xb)
    ry = np.clip(ry, ya, yb)
    return rx + 1j * ry


def _snap_endpoints(dc, dom, tol):
    """Snap only the first/last sample of an open curve onto a domain edge
    when it lands within ``tol`` (about a grid cell) of it -- marching
    squares stops one cell short of the boundary, and that missing stub
    otherwise costs the open curve a little arc length."""
    xa, xb, ya, yb = dom
    dc = dc.copy()
    for i in (0, -1):
        rx = float(np.real(dc[i]))
        ry = float(np.imag(dc[i]))
        if abs(rx - xa) < tol:
            rx = xa
        elif abs(rx - xb) < tol:
            rx = xb
        if abs(ry - ya) < tol:
            ry = ya
        elif abs(ry - yb) < tol:
            ry = yb
        dc[i] = rx + 1j * ry
    return dc


def _polish_endpoints(f, fx, fy, dc, dom, tol):
    """1D Newton on an open curve's boundary endpoints: if the endpoint sits
    on a vertical edge (x fixed), solve f(x, y)=0 for y; on a horizontal edge
    (y fixed), solve for x.  Leaves the endpoint exactly on the zero set at
    the true boundary crossing."""
    xa, xb, ya, yb = dom
    dc = dc.copy()
    for i in (0, -1):
        x0 = float(np.real(dc[i]))
        y0 = float(np.imag(dc[i]))
        on_vert = abs(x0 - xa) == 0.0 or abs(x0 - xb) == 0.0
        on_horiz = abs(y0 - ya) == 0.0 or abs(y0 - yb) == 0.0
        if not (on_vert or on_horiz):
            continue
        for _ in range(6):
            fv = float(f(jnp.asarray(x0), jnp.asarray(y0)))
            if on_vert and not on_horiz:
                d = float(fy(jnp.asarray(x0), jnp.asarray(y0)))
                if d == 0:
                    break
                yn = min(max(y0 - fv / d, ya), yb)
                if abs(yn - y0) < 1e-15:
                    y0 = yn
                    break
                y0 = yn
            elif on_horiz and not on_vert:
                d = float(fx(jnp.asarray(x0), jnp.asarray(y0)))
                if d == 0:
                    break
                xn = min(max(x0 - fv / d, xa), xb)
                if abs(xn - x0) < 1e-15:
                    x0 = xn
                    break
                x0 = xn
            else:
                break
        dc[i] = x0 + 1j * y0
    return dc


def _fit_curve(f, fx, fy, pts, scl, vscale, dom, snap_tol):
    """Refine a polyline into an accurate complex Chebfun zero curve.

    Mirrors the MATLAB refinement loop: reparametrize by empirical
    arclength, interpolate to Chebyshev points, sample the current curve
    finely, take one complex-Newton step onto the zero set, and rebuild
    the Chebfun.  Iterate while the residual improves.
    """
    from scipy.interpolate import interp1d

    data = pts[:, 0] + 1j * pts[:, 1]
    # Drop near-duplicate consecutive points.
    keep = np.concatenate([[True],
                           np.abs(np.diff(data)) > 1e-8 * scl])
    data = data[keep]
    if data.shape[0] < 4:
        return None
    # Cap the number of fit points: the fine grid gives an accurate
    # polyline, but fitting a Chebfun through >~few-hundred points is both
    # slow and over-resolved.  Downsample uniformly along the polyline.
    max_fit = 600
    if data.shape[0] > max_fit:
        idx = np.linspace(0, data.shape[0] - 1, max_fit).round().astype(int)
        data = data[idx]
    npts = data.shape[0]

    def _grad(z):
        rx = np.real(z)
        ry = np.imag(z)
        gx = np.asarray(fx(jnp.asarray(rx), jnp.asarray(ry)),
                        dtype=np.float64)
        gy = np.asarray(fy(jnp.asarray(rx), jnp.asarray(ry)),
                        dtype=np.float64)
        return gx + 1j * gy

    def _fval(z):
        return np.asarray(f(jnp.asarray(np.real(z)),
                            jnp.asarray(np.imag(z))), dtype=np.float64)

    err = np.inf
    errnew = 1e-2
    curve = None
    step = 0
    closed = abs(data[0] - data[-1]) < 1e-6 * scl
    while errnew < err and step < 8:
        step += 1
        err = errnew
        # Empirical normalized arclength in [-1, 1].
        s = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(data)))])
        if s[-1] == 0:
            return curve
        s = 2.0 * s / s[-1] - 1.0
        cp = _chebpts(npts)
        # Interpolate real/imag separately (cubic when possible).
        kind = "cubic" if npts >= 4 else "linear"
        fr = interp1d(s, np.real(data), kind=kind,
                      fill_value="extrapolate")
        fi = interp1d(s, np.imag(data), kind=kind,
                      fill_value="extrapolate")
        dc = fr(cp) + 1j * fi(cp)
        dc = _snap(dc, dom, 1e-10 * scl)
        # Newton polish each sample onto the zero set.
        g = _grad(dc)
        ff = _fval(dc)
        errnew = float(np.max(np.abs(ff))) / max(vscale, 1e-300)
        with np.errstate(divide="ignore", invalid="ignore"):
            step_vec = ff * (g / np.abs(g) ** 2)
        step_vec = np.where(np.abs(g) > 0, step_vec, 0.0)
        dc = dc - step_vec
        dc = _snap(dc, dom, 1e-10 * scl)
        if closed:
            dc[-1] = dc[0]
        else:
            # Pin open-curve endpoints onto the boundary, then a 1D Newton
            # in the free coordinate places them exactly on the zero set.
            dc = _snap_endpoints(dc, dom, snap_tol)
            dc = _polish_endpoints(f, fx, fy, dc, dom, snap_tol)
        data = dc
        # Chebfun.from_values casts to float64 (dropping the imaginary
        # part), so build the real and imaginary components separately and
        # recombine into a complex curve.
        cx = _simplify(Chebfun.from_values(jnp.asarray(np.real(dc)),
                                           Domain((-1.0, 1.0))))
        cy = _simplify(Chebfun.from_values(jnp.asarray(np.imag(dc)),
                                           Domain((-1.0, 1.0))))
        curve = cx + 1j * cy
    return curve


def zero_curves(f):
    """Return the zero curves of a real Chebfun2 as complex Chebfuns.

    Parameters
    ----------
    f : Chebfun2

    Returns
    -------
    list of Chebfun
        One complex-valued Chebfun ``c(t) = x(t) + 1i*y(t)`` per traced
        zero contour (empty if f has no sign change in the domain).
    """
    xa, xb, ya, yb = (float(v) for v in f.domain)
    scl = max(abs(xa), abs(xb), abs(ya), abs(yb), 1.0)

    if f.rank == 1:
        return _rank1_curves(f)

    n = _grid_size(f)
    polylines = _marching_squares(f, n)
    if not polylines:
        return []

    fx = f.diff(dim=2)   # d/dx
    fy = f.diff(dim=1)   # d/dy

    def _fx(x, y):
        return fx(x, y)

    def _fy(x, y):
        return fy(x, y)

    # vscale of f over the domain (for relative Newton residual).
    gx = np.linspace(xa, xb, 33)
    gy = np.linspace(ya, yb, 33)
    XX, YY = np.meshgrid(gx, gy)
    vscale = float(np.max(np.abs(
        np.asarray(f(jnp.asarray(XX), jnp.asarray(YY))))))

    # Endpoints of an open contour land within ~one grid cell of the
    # boundary; snap anything closer than a couple of cells.
    snap_tol = 2.5 * max((xb - xa), (yb - ya)) / (n - 1)

    curves = []
    for pts in polylines:
        c = _fit_curve(f, _fx, _fy, pts, scl, vscale, (xa, xb, ya, yb),
                       snap_tol)
        if c is not None:
            curves.append(c)
    return curves
