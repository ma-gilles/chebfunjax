"""Independent finite-real integration for Chebfun's old convolution route.

Provenance
----------
Chebfun commit: 7574c77, @chebfun/conv.m oldConv/convIntegral.
MATLAB R2025b funfun/private/{integralCalc,integralParseArgs,Gauss7Kronrod15}.m.
Finite real scalar-valued branch only; no contour or infinite integration API.
The source clamps requested relative tolerance to 100 binary64 eps.
"""

import warnings

import jax
import jax.numpy as jnp

_EPS = 2.220446049250313e-16
_NODES = (
    -0.9914553711208126,
    -0.9491079123427585,
    -0.8648644233597691,
    -0.7415311855993944,
    -0.5860872354676911,
    -0.4058451513773972,
    -0.2077849550078985,
    0.0,
    0.2077849550078985,
    0.4058451513773972,
    0.5860872354676911,
    0.7415311855993944,
    0.8648644233597691,
    0.9491079123427585,
    0.9914553711208126,
)
_HIGH = (
    0.02293532201052922,
    0.06309209262997855,
    0.1047900103222502,
    0.1406532597155259,
    0.1690047266392679,
    0.1903505780647854,
    0.2044329400752989,
    0.2094821410847278,
    0.2044329400752989,
    0.1903505780647854,
    0.1690047266392679,
    0.1406532597155259,
    0.1047900103222502,
    0.06309209262997855,
    0.02293532201052922,
)
_LOW = (
    0.0,
    0.1294849661688697,
    0.0,
    0.2797053914892767,
    0.0,
    0.3818300505051189,
    0.0,
    0.4179591836734694,
    0.0,
    0.3818300505051189,
    0.0,
    0.2797053914892767,
    0.0,
    0.1294849661688697,
    0.0,
)


@jax.jit
def _panel_product(f, g, query, a, b, subs):
    mid = (subs[0] + subs[1]) / 2
    half = (subs[1] - subs[0]) / 2
    t = jnp.asarray(_NODES)[:, None] * half[None, :] + mid[None, :]
    shift = jnp.abs(t) - 1
    offset = 0.25 * (b - a) * shift**2 * (shift + 3)
    x = 0.25 * (b - a) * t * (3 - t * t) + 0.5 * (b + a)
    x = jnp.where(t < -0.25, a + offset, jnp.where(t > 0.25, b - offset, x))
    w = jnp.where(
        jnp.abs(t) > 0.25, -0.75 * (b - a) * shift * (shift + 2), 0.75 * (b - a) * (1 - t * t)
    )
    values = f(x) * g(query - x) * w
    high = jnp.asarray(_HIGH)
    low = jnp.asarray(_LOW)
    return (high @ values) * half, ((high - low) @ values) * half, x


def _integral_product(f, g, query, a, b):
    # Source split([-1,1],10): interior=start+(1:9)/10*width.
    edges = jnp.concatenate((jnp.array([-1.0]), -1 + jnp.arange(1, 10) / 10 * 2, jnp.array([1.0])))
    subs = jnp.stack((edges[:-1], edges[1:]))
    q_ok, err_ok = 0.0, 0.0
    first = True
    while True:
        qs, es, points = _panel_product(f, g, query, a, b, subs)
        if not first:
            flat = points.T.reshape(-1)
            if bool(
                jnp.any(
                    jnp.abs(jnp.diff(flat))
                    <= 100 * _EPS * jnp.maximum(jnp.abs(flat[:-1]), jnp.abs(flat[1:]))
                )
            ):
                warnings.warn("MATLAB:integral:MinStepSize", stacklevel=3)
                break
        first = False
        q = jnp.sum(qs) + q_ok
        tol = jnp.maximum(1e-15, (100 * _EPS) * jnp.abs(q))
        half = (subs[1] - subs[0]) / 2
        accepted = jnp.abs(es) <= tol * jnp.abs(half)
        err_ok = err_ok + jnp.sum(jnp.where(accepted, es, 0))
        errbnd = jnp.abs(err_ok) + jnp.sum(jnp.where(accepted, 0, jnp.abs(es)))
        if not bool(jnp.isfinite(q) & jnp.isfinite(errbnd)):
            warnings.warn("MATLAB:integral:NonFiniteValue", stacklevel=3)
            break
        if bool(errbnd <= tol):
            break
        q_ok = q_ok + jnp.sum(jnp.where(accepted, qs, 0))
        subs = subs[:, ~accepted]
        if subs.shape[1] == 0:
            break
        if 2 * subs.shape[1] > 16384:
            warnings.warn("MATLAB:integral:MaxIntervalCountReached", stacklevel=3)
            break
        middle = (subs[0] + subs[1]) / 2
        subs = jnp.stack(
            (
                jnp.stack((subs[0], middle), axis=1).reshape(-1),
                jnp.stack((middle, subs[1]), axis=1).reshape(-1),
            )
        )
    return q


def old_convolution(f, g):
    """Construct the source convolution by independent adaptive quadrature.

    Provenance
    ----------
    @chebfun/conv.m oldConv/convIntegral, Chebfun commit: 7574c77.
    R2025b integral finite-real scalar branch and its embedded 7/15 rule.
    """
    from chebfunjax.chebfun1d._convolution import _as_chebfun
    from chebfunjax.chebfun1d.chebfun import _Piece

    a, b, c, d = map(float, (f.domain.a, f.domain.b, g.domain.a, g.domain.b))
    fb, gb = list(f.domain.breakpoints), list(g.domain.breakpoints)
    dom = sorted({x + y for x in fb for y in gb})
    tolerance = 10 * _EPS * max(abs(dom[0]), abs(dom[-1]))
    dom = [v for k, v in enumerate(dom) if k == len(dom) - 1 or dom[k + 1] - v >= tolerance]

    def values(x):
        shape = jnp.shape(x)
        out = []
        for query in jnp.asarray(x).reshape(-1).tolist():
            lo, hi = max(a, query - d), min(b, query - c)
            total = 0.0
            if lo < hi:
                interior = sorted({*fb, *(query - v for v in gb)})
                bounds = [lo] + [v for v in interior if lo < v < hi] + [hi]
                for left, right in zip(bounds[:-1], bounds[1:]):
                    total = total + _integral_product(f, g, query, left, right)
            out.append(total)
        return jnp.asarray(out).reshape(shape)

    hs = max(abs(a), abs(b), abs(c), abs(d))
    vs = 2 * max(float(f.vscale), float(g.vscale))
    pieces = []
    for left, right in zip(dom[:-1], dom[1:]):
        p = _Piece.from_function(
            values,
            left,
            right,
            vscale=vs,
            hscale=hs / (right - left),
            extrapolate=True,
            sample_test=False,
            refinement_function="nested",
        )
        vs = max(vs, float(p.vscale))
        pieces.append(p)
    return _as_chebfun(pieces)
