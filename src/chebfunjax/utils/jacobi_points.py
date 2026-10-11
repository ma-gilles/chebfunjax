"""Jacobi recurrence kernels.

Provenance
----------
MATLAB source: jacpts.m, rec/rec_main/eval_Jac.
Chebfun commit: 7574c77. Authors: Nick Hale and Alex Townsend.
"""
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=('n', 'count'))
def _rec_half(n, a, b, count):
    r = jnp.arange(count, 0, -1, dtype=jnp.float64)
    c = (2*r+a-.5)*jnp.pi/(2*n+a+b+1)
    t = c + 1/(2*n+a+b+1)**2*((.25-a*a)/jnp.tan(.5*c)
                                            - (.25-b*b)*jnp.tan(.5*c))
    x = jnp.cos(t)

    def evaluate(x):
        ab = a+b
        p = .5*(a-b+(ab+2)*x)
        pp = jnp.full_like(x, .5*(ab+2))

        def step(k, state):
            p, pm, pp, ppm = state
            aa = 2*(k+1)*(k+ab+1)*(2*k+ab)
            bb = (2*k+ab+1)*(a*a-b*b)
            cc = (2*k+ab)*(2*k+ab+1)*(2*k+ab+2)
            dd = 2*(k+a)*(k+b)*(2*k+ab+2)
            pn = ((bb+cc*x)*p-dd*pm)/aa
            ppn = ((bb+cc*x)*pp+cc*p-dd*ppm)/aa
            return pn, p, ppn, pp

        p, _, pp, _ = jax.lax.fori_loop(
            1, n, step, (p, jnp.ones_like(x), pp, jnp.zeros_like(x)))
        return p, pp

    def cond(state):
        _, dx, iteration = state
        return (jnp.max(jnp.abs(dx)) > jnp.sqrt(jnp.finfo(jnp.float64).eps)/1000) & (iteration < 10)

    def body(state):
        x, _, iteration = state
        p, pp = evaluate(x)
        dx = -p/pp
        return x+dx, dx, iteration+1

    x, _, _ = jax.lax.while_loop(cond, body, (x, jnp.full_like(x, jnp.inf), 0))
    return x, evaluate(x)[1]


def _jacobi_rec(n, a, b):
    x1, d1 = _rec_half(n, a, b, (n+1)//2)
    x2, d2 = _rec_half(n, b, a, n//2)
    x = jnp.concatenate((-x2[::-1], x1))
    d = jnp.concatenate((d2[::-1], d1))
    return x, 1/((1-x*x)*d*d), 1/d


@jax.jit
def _jacobi_ratio(m, delta):
    # jacpts.m local ratioOfGammaFunctions: no generic integer stripping.
    from chebfunjax.utils.gamma_ratio import _stirling_correction

    ds = .5*delta**2/(m-1)

    def cond(state):
        _, ds, total = state
        return jnp.abs(ds/total) > jnp.finfo(jnp.float64).eps/100

    def body(state):
        j, ds, total = state
        j = j+1
        ds = -delta*(j-1)/(j+1)/(m-1)*ds
        return j, ds, total+ds

    _, _, total = jax.lax.while_loop(cond, body, (1., ds, ds))
    p2 = jnp.exp(total)*jnp.sqrt(1+delta/(m-1))*(m-1)**delta
    return p2*(_stirling_correction(m+delta-1)/_stirling_correction(m-1))


# Licensed MATLAB R2025b code-generation gammaln.m integer branch.
# Kept local to Jacobi mass evaluation; nonintegers use JAX's backend.
_GAMMALN_INTEGERS = (
    0.00000000000000000e+00,
    0.00000000000000000e+00,
    6.93147180559945309e-01,
    1.79175946922805500e+00,
    3.17805383034794562e+00,
    4.78749174278204599e+00,
    6.57925121201010100e+00,
    8.52516136106541430e+00,
    1.06046029027452502e+01,
    1.28018274800814696e+01,
    1.51044125730755153e+01,
    1.75023078458738858e+01,
    1.99872144956618861e+01,
    2.25521638531234229e+01,
    2.51912211827386815e+01,
    2.78992713838408916e+01,
    3.06718601060806728e+01,
    3.35050734501368889e+01,
    3.63954452080330536e+01,
    3.93398841871994940e+01,
    4.23356164607534850e+01,
    4.53801388984769080e+01,
    4.84711813518352239e+01,
    5.16066755677643736e+01,
    5.47847293981123192e+01,
    5.80036052229805199e+01,
    6.12617017610020020e+01,
    6.45575386270063311e+01,
    6.78897431371815350e+01,
    7.12570389671680090e+01,
    7.46582363488301644e+01,
    7.80922235533153106e+01,
    8.15579594561150372e+01,
    8.50544670175815174e+01,
    8.85808275421976788e+01,
    9.21361756036870925e+01,
    9.57196945421432025e+01,
    9.93306124547874269e+01,
    1.02968198614513813e+02,
    1.06631760260643459e+02,
    1.10320639714757395e+02,
    1.14034211781461703e+02,
    1.17771881399745072e+02,
    1.21533081515438634e+02,
    1.25317271149356895e+02,
    1.29123933639127215e+02,
    1.32952575035616310e+02,
    1.36802722637326368e+02,
    1.40673923648234259e+02,
    1.44565743946344886e+02,
    1.48477766951773032e+02,
    1.52409592584497358e+02,
    1.56360836303078785e+02,
    1.60331128216630907e+02,
    1.64320112263195181e+02,
    1.68327445448427652e+02,
    1.72352797139162802e+02,
    1.76395848406997352e+02,
    1.80456291417543771e+02,
    1.84533828861449491e+02,
    1.88628173423671591e+02,
    1.92739047287844902e+02,
    1.96866181672889994e+02,
    2.01009316399281527e+02,
    2.05168199482641199e+02,
    2.09342586752536836e+02,
    2.13532241494563261e+02,
    2.17736934113954227e+02,
    2.21956441819130334e+02,
    2.26190548323727593e+02,
    2.30439043565776952e+02,
    2.34701723442818268e+02,
    2.38978389561834323e+02,
    2.43268849002982714e+02,
    2.47572914096186884e+02,
    2.51890402209723194e+02,
    2.56221135550009525e+02,
    2.60564940971863209e+02,
    2.64921649798552801e+02,
    2.69291097651019823e+02,
    2.73673124285693704e+02,
    2.78067573440366143e+02,
    2.82474292687630396e+02,
    2.86893133295426994e+02,
    2.91323950094270308e+02,
    2.95766601350760624e+02,
    3.00220948647014132e+02,
    3.04686856765668715e+02,
    3.09164193580146922e+02,
    3.13652829949879062e+02,
    3.18152639620209327e+02,
    3.22663499126726177e+02,
    3.27185287703775217e+02,
    3.31717887196928473e+02,
    3.36261181979198477e+02,
    3.40815058870799018e+02,
    3.45379407062266854e+02,
    3.49954118040770237e+02,
    3.54539085519440809e+02,
    3.59134205369575399e+02,
)


def _jacobi_mass_gammaln(x):
    """Integer-table branch of MATLAB code-generation gammaln.

    Provenance
    ----------
    Chebfun commit: 7574c77 (jacpts.m beta mass caller).
    MATLAB R2025b toolbox/eml/lib/matlab/specfun/gammaln.m,
    scalar_gammaln integer branch and full table100. Noninteger evaluations
    retain JAX gammaln; full MATLAB special-function parity is not claimed.
    """
    from jax.scipy.special import gammaln

    x = jnp.asarray(x, dtype=jnp.float64)
    use_table = (x >= 1) & (x <= 100) & (x == jnp.floor(x))
    index = jnp.clip(x-1, 0, 99).astype(jnp.int32)
    table = jnp.asarray(_GAMMALN_INTEGERS, dtype=jnp.float64)
    return jnp.where(use_table, table[index], gammaln(x))



def _jacobi_symmetric(n, a, interval, method):
    """Native jacpts symmetric special cases, including method forwarding."""
    from .quadrature import chebpts, ultrapts

    if a not in (0., -.5, .5):
        # Native ignores jacpts METHOD here and calls ultrapts default.
        return ultrapts(n, a+.5, interval, bary=True)
    if a == 0.:
        if n <= 20 or method == 'rec':
            from .legendre_rec import _legpts_rec
            x, w, v = _legpts_rec(n)
        elif method == 'gw':
            k = jnp.arange(1, n, dtype=jnp.float64)
            beta = .5/jnp.sqrt(1-(2*k)**(-2))
            matrix = jnp.diag(beta, 1)+jnp.diag(beta, -1)
            x, vectors = jnp.linalg.eigh(matrix)
            w = 2*vectors[0]**2
            v = jnp.sqrt(1-x*x)*jnp.abs(vectors[0])
            m = n//2
            vmid = v[m:m+1]
            x, w, v = x[:m], w[:m], v[:m]
            if n % 2:
                x = jnp.concatenate((x, jnp.zeros(1), -x[::-1]))
                w = jnp.concatenate((w, jnp.asarray([2-jnp.sum(2*w)]), w[::-1]))
                v = jnp.concatenate((v, vmid, v[::-1]))
            else:
                x, w, v = (jnp.concatenate((x, -x[::-1])),
                           jnp.concatenate((w, w[::-1])),
                           jnp.concatenate((v, v[::-1])))
            v = jnp.abs(v)/jnp.max(jnp.abs(v))*jnp.where(jnp.arange(n)%2, -1., 1.)
        else:
            # legpts(n, 'default') treats the string as an explicit method;
            # therefore it selects ASY above20, even below100.
            from .legendre_fast import _legpts_asy_with_theta
            x, w, v, _ = _legpts_asy_with_theta(n)
    elif a == -.5:
        x = chebpts(n, 1)
        w = jnp.full(n, jnp.pi/n)
        # chebtech1.barywts uses the opposite initial sign for even n and
        # does not normalize the maximum to1.
        v = jnp.sin((jnp.arange(n-1, -1, -1, dtype=jnp.float64)+.5)*jnp.pi/n)
        m = n//2
        v = v.at[:m].set(v[-m:][::-1])
        v = v*jnp.where((n-1-jnp.arange(n))%2, -1., 1.)
        if tuple(interval) != (-1., 1.):
            from chebfunjax.domain import _linear_forward_map
            x = _linear_forward_map(x, interval[0], interval[1])
        return x, w, v  # width exponent2a+1 is zero.
    else:
        x = chebpts(n+2, 2)[1:-1]
        w = jnp.pi/(n+1)*(1-x*x)
        v = (1-x*x)*jnp.where(jnp.arange(n)%2, -1., 1.)
    if tuple(interval) != (-1., 1.):
        c1 = .5*(interval[0]+interval[1])
        c2 = .5*(interval[1]-interval[0])
        x = c1+c2*x
        w = c2**(2*a+1)*w
    return x, w, v
