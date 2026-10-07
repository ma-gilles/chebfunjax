"""Literal first-kind barycentric formula on Chebyshev TYPE2 grids.

Provenance
----------
MATLAB source : ratinterp.m/constructRatApproxCheb2 and ratbary;
    @chebtech2/coeffs2vals.m
Chebfun commit: 7574c77
"""
import jax
import jax.numpy as jnp

from chebfunjax.utils.quadrature import chebpts


def _source_type2_values(coeffs):
    """JAX-only source FFT conversion, including sequential symmetry repair.

    Provenance
    ----------
    MATLAB source : @chebtech2/coeffs2vals.m
    Chebfun commit: 7574c77
    """
    c = jnp.asarray(coeffs)
    n = c.shape[0]
    if n <= 1:
        return c
    even = jnp.all(c[1::2] == 0)
    odd = jnp.all(c[::2] == 0)
    scaled = c.at[1:n-1].divide(2)
    tmp = jnp.concatenate((scaled, scaled[n-2:0:-1]))
    general = jnp.fft.fft(tmp)
    if not jnp.iscomplexobj(c):
        values = jnp.real(general)
    elif bool(jax.device_get(jnp.all(jnp.imag(c) == 0))):
        values = jnp.real(general)
    elif bool(jax.device_get(jnp.all(jnp.real(c) == 0))):
        values = 1j * jnp.real(jnp.fft.fft(jnp.imag(tmp)))
    else:
        values = general
    values = values[n-1::-1]
    values = jnp.where(even, (values + values[::-1])/2, values)
    values = jnp.where(odd, (values - values[::-1])/2, values)
    return values


def _source_complex_real_scalar(value, factor, *, divide=False):
    """Paired scalar complex-by-real operation for the observed source scope.

    Provenance
    ----------
    MATLAB source : ratinterp.m/ratbary scalar weighting/division/product stages
    Chebfun commit: 7574c77
    MATLAB R2025b primitive captures distinguish scalar operations from dots.
    This is not a replacement for general complex arithmetic.
    """
    real, imag = jnp.real(value), jnp.imag(value)
    r = real / factor if divide else real * factor
    i = imag / factor if divide else imag * factor
    r = jnp.where((real == 0) & (imag != 0), jnp.zeros_like(real), r)
    i = jnp.where(imag == 0, jnp.zeros_like(imag), i)
    return jax.lax.complex(r, i)


def _source_nonfinite_offaxis(value, factor, *, divide=False):
    """Observed nonfinite scalar complex arithmetic with finite off-axis factor.

    Provenance
    ----------
    MATLAB source : ratinterp.m/ratbary scalar numerator dot and quotient
    Chebfun commit: 7574c77
    MATLAB R2025b all-stages/42-primitive capture motivates this bounded
    adaptation. Axis/zero/nonfinite factors retain native behavior; no general
    MATLAB complex primitive or C99 equivalence is claimed.
    """
    native = value / factor if divide else value * factor
    if not jnp.iscomplexobj(value) or not jnp.iscomplexobj(factor):
        return native
    a, b = jnp.real(value), jnp.imag(value)
    c, d = jnp.real(factor), jnp.imag(factor)
    use = ~jnp.isfinite(value) & jnp.isfinite(factor) & (c != 0) & (d != 0)
    if divide:
        # Smith scaled component quotient, deliberately without infinity
        # recovery. This preserves the observed Inf +/- Inf -> NaN channel.
        ratio_c = d / c
        denom_c = c + d * ratio_c
        re_c = (a + b * ratio_c) / denom_c
        im_c = (b - a * ratio_c) / denom_c
        ratio_d = c / d
        denom_d = d + c * ratio_d
        re_d = (a * ratio_d + b) / denom_d
        im_d = (b * ratio_d - a) / denom_d
        real = jnp.where(jnp.abs(c) >= jnp.abs(d), re_c, re_d)
        imag = jnp.where(jnp.abs(c) >= jnp.abs(d), im_c, im_d)
    else:
        an = jnp.copysign(jnp.where(jnp.isinf(a), 1., 0.), a)
        bn = jnp.copysign(jnp.where(jnp.isinf(b), 1., 0.), b)
        real = jnp.inf * (an*c - bn*d)
        imag = jnp.inf * (an*d + bn*c)
    invalid = jnp.isnan(a) | jnp.isnan(b)
    alternative = jax.lax.complex(jnp.where(invalid, jnp.nan, real),
                                  jnp.where(invalid, jnp.nan, imag))
    return jnp.where(use, alternative, native)


def _source_denominator_sum(qx, wq, inverse):
    """Real denominator row sums with a rounded-product boundary.

    Provenance
    ----------
    MATLAB source : ratinterp.m/ratbary, qxw * dxqinv
    Chebfun commit: 7574c77
    MATLAB R2025b stage capture confirms separately rounded products in the
    bounded cancellation control. A non-unrolled scan keeps those products
    separate from accumulation in the qualified CPU lowering. Sequential
    accumulation is not a claim of general MATLAB BLAS reduction bit identity.
    Storage is proportional to query count times denominator length.
    """
    weighted = qx * wq
    if not jnp.iscomplexobj(weighted) and not jnp.iscomplexobj(inverse):
        products = jax.lax.optimization_barrier(inverse * weighted)

        def add_product(total, term):
            return total + term, None

        total, _ = jax.lax.scan(
            add_product,
            jnp.zeros_like(products[..., 0]),
            jnp.moveaxis(products, -1, 0),
            unroll=1,
        )
        return total
    return inverse @ weighted


def _source_ratbary_vector(x, px, qx, xp, xq, wp, wq):
    """Source vector loop, node hits, and all-nonfinite product fallbacks.

    Provenance
    ----------
    MATLAB source : ratinterp.m/ratbary (601–675)
    Chebfun commit: 7574c77
    """
    dp, dq = x[:, None] - xp[None, :], x[:, None] - xq[None, :]
    dxp = 1 / jnp.where(dp == 0, 1, dp)
    dxq = 1 / jnp.where(dq == 0, 1, dq)
    hitp, hitq = (dp == 0) | ~jnp.isfinite(dxp), (dq == 0) | ~jnp.isfinite(dxq)
    # MATLAB assigns px(ind) / qx(ind) into a scalar y(i). Multiple
    # reciprocal nonfinite hits therefore raise, rather than selecting one.
    traced = isinstance(x, jax.core.Tracer)
    if not traced and bool(jax.device_get(jnp.any(jnp.sum(hitp, axis=1) > 1))):
        raise ValueError("ratbary numerator node selection has multiple indices")
    if not traced and bool(jax.device_get(jnp.any(jnp.sum(hitq, axis=1) > 1))):
        raise ValueError("ratbary denominator node selection has multiple indices")
    paired = (jnp.iscomplexobj(px) and not jnp.iscomplexobj(x)
              and not jnp.iscomplexobj(qx))
    if paired:
        weighted = _source_complex_real_scalar(px, wp)
        # Matrix dot has separate source semantics: sum each channel without
        # cross-channel zero*Inf terms or an absent-channel bypass.
        summed = jax.lax.complex(dxp @ jnp.real(weighted), dxp @ jnp.imag(weighted))
        numerator = jnp.where(jnp.any(hitp, axis=1), px[jnp.argmax(hitp, axis=1)], summed)
        denominator = jnp.where(jnp.any(hitq, axis=1), qx[jnp.argmax(hitq, axis=1)],
                                _source_denominator_sum(qx, wq, dxq))
        y = _source_complex_real_scalar(numerator, denominator, divide=True)
    else:
        weighted = px * wp
        summed = (_source_nonfinite_offaxis(weighted[0], dxp[:, 0])
                  if px.size == 1 else dxp @ weighted)
        numerator = jnp.where(jnp.any(hitp, axis=1), px[jnp.argmax(hitp, axis=1)], summed)
        denominator = jnp.where(jnp.any(hitq, axis=1), qx[jnp.argmax(hitq, axis=1)],
                                _source_denominator_sum(qx, wq, dxq))
        y = _source_nonfinite_offaxis(numerator, denominator, divide=True)
    llp, llq = x[:, None] - xp[None, :], x[:, None] - xq[None, :]
    lp, lq = jnp.prod(llp, axis=1), jnp.prod(llq, axis=1)
    if traced:
        # Fixed-dtype traced contract: valid ordinary values remain supported.
        # Dynamic source errors/logarithmic dtype changes are explicitly outside
        # this contract and return NaN rather than silently choosing a node.
        invalid = ((jnp.sum(hitp, axis=1) > 1)
                   | (jnp.sum(hitq, axis=1) > 1)
                   | ~jnp.isfinite(lp) | ~jnp.isfinite(lq))
        p, q = jnp.where(lp == 0, 1, lp), jnp.where(lq == 0, 1, lq)
        result = (_source_complex_real_scalar(
            _source_complex_real_scalar(y, p), q, divide=True) if paired else _source_nonfinite_offaxis(
                _source_nonfinite_offaxis(y, p), q, divide=True))
        return jnp.where(invalid, jnp.asarray(jnp.nan, dtype=y.dtype), result)
    # Source if(vector) means all elements. Eager host decisions preserve
    # MATLAB's data-dependent real/complex logarithm result dtype.
    if bool(jax.device_get(jnp.all(~jnp.isfinite(lp)))):
        if bool(jax.device_get(jnp.any(llp < 0))) if not jnp.iscomplexobj(llp) else True:
            llp = llp.astype(jnp.result_type(llp, 1j))
        lp = jnp.exp(jnp.sum(jnp.log(llp), axis=1))
    if bool(jax.device_get(jnp.all(~jnp.isfinite(lq)))):
        if bool(jax.device_get(jnp.any(llq < 0))) if not jnp.iscomplexobj(llq) else True:
            llq = llq.astype(jnp.result_type(llq, 1j))
        lq = jnp.exp(jnp.sum(jnp.log(llq), axis=1))
    lp, lq = jnp.where(lp == 0, 1, lp), jnp.where(lq == 0, 1, lq)
    if paired and not jnp.iscomplexobj(lp) and not jnp.iscomplexobj(lq):
        return _source_complex_real_scalar(
            _source_complex_real_scalar(y, lp), lq, divide=True)
    return _source_nonfinite_offaxis(
        _source_nonfinite_offaxis(y, lp), lq, divide=True)


def _source_type2_rational_handle(a, b, domain):
    """Build TYPE2 handle with ordinary finite-input transformations.

    Eager evaluation preserves source errors and dynamic overflow dtype.
    Traced invalid/overflow inputs yield NaN and are not source-qualified.
    Query derivatives use the polynomial quotient where the source expression
    is continuous; isolated source node discontinuities have NaN tangents.


    Provenance
    ----------
    MATLAB source : ratinterp.m/constructRatApproxCheb2 and ratbary
    Chebfun commit: 7574c77
    """
    a, b = jnp.asarray(a), jnp.asarray(b)
    if isinstance(a, jax.core.Tracer) or isinstance(b, jax.core.Tracer):
        raise NotImplementedError("Literal TYPE2 handle construction is eager-only")
    # Legacy fitting forces complex storage for real sampled data. Restore
    # exactly-real singleton storage; never threshold genuine imaginary parts.
    if a.size == 1 and jnp.iscomplexobj(a):
        if bool(jax.device_get(jnp.all(jnp.imag(a) == 0))):
            a = jnp.real(a)
    mu, nu = a.size - 1, b.size - 1
    assert nu > 0
    px, qx = _source_type2_values(a), _source_type2_values(b)
    xp, xq = chebpts(mu+1, kind=2), chebpts(nu+1, kind=2)
    wp = jnp.ones(mu+1).at[1::2].set(-1)
    wp = wp.at[0].set(.5)
    wp = wp.at[-1].multiply(.5)
    # Singleton numerator intentionally receives BOTH endpoint assignments.
    wp = wp * (-2.0)**(mu-nu) / jnp.asarray(mu, dtype=wp.dtype) * nu
    wq = jnp.ones(nu+1).at[1::2].set(-1)
    wq = wq.at[0].set(.5)
    wq = wq.at[-1].multiply(.5)
    midpoint = .5 * (domain[0] + domain[1])
    inverse_halfwidth = 2.0 / (domain[1] - domain[0])

    def primal(x):
        t = inverse_halfwidth * (jnp.asarray(x) - midpoint)
        if t.ndim > 2:
            raise ValueError("MATLAB ratbary accepts scalar, vector or matrix queries")

        def column(v):
            return _source_ratbary_vector(v, px, qx, xp, xq, wp, wq)

        if t.ndim == 2 and min(t.shape) > 1:
            return jnp.stack([column(t[:, k]) for k in range(t.shape[1])], axis=1)
        return column(t.reshape(-1)).reshape(t.shape)

    from chebfunjax.tech.chebtech import _clenshaw

    @jax.custom_jvp
    def evaluate(x):
        return primal(x)

    @evaluate.defjvp
    def evaluate_jvp(primals, tangents):
        (x,), (dx,) = primals, tangents
        x, dx = jnp.asarray(x), jnp.asarray(dx)
        value = primal(x)

        def ratio(z):
            t = inverse_halfwidth * (z-midpoint)
            return _clenshaw(a, t) / _clenshaw(b, t)

        # The quotient is elementwise and holomorphic in complex queries.
        # A unit direction therefore gives its diagonal derivative/slope.
        ideal, slope = jax.jvp(ratio, (x,), (jnp.ones_like(x),))
        t = inverse_halfwidth * (x-midpoint)
        hp = jnp.any(t[..., None] == xp, axis=-1)
        hq = jnp.any(t[..., None] == xq, axis=-1)
        # For ascending degree-nu Lobatto nodes, the source alternating
        # half-endpoint weights are C_nu / node_polynomial'(x_j), where
        # C_nu=(-1)**nu * nu / 2**(nu-1). Source wp rescales its numerator
        # weights to the SAME C_nu. Away from nodes these factors cancel.
        # A numerator-only hit bypasses C_nu (value ratio/C_nu); a
        # denominator-only hit bypasses its reciprocal (value C_nu*ratio).
        # Both hits cancel. C_nu=1 precisely at nu=2 among positive integer
        # degrees; zero numerator values also remove the isolated jump.
        continuous = (~(hp | hq) | (hp & hq) | (nu == 2) | (ideal == 0)) & (mu > 0)
        # Mask the linear coefficient, not the directional result. A constant
        # NaN branch in the tangent has zero reverse transpose, incorrectly
        # reporting derivative zero at the source's isolated discontinuity.
        slope = jnp.where(continuous & jnp.isfinite(value), slope, jnp.nan)
        tangent = slope * dx
        if not jnp.iscomplexobj(value):
            tangent = jnp.real(tangent)
        return value, tangent.astype(value.dtype)

    return evaluate
