"""All19 literal default predicates of tests/chebfun/test_roots.m,7574c77.

MATLAB MAX ignores NaNs; preserve that grammar in array/quasimatrix predicates.
Existing test_roots_matlab.py and authored source controls remain separate.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.utils.besselj import besselj

EPS = np.finfo(float).eps


def maximum_abs(x, *, ignore_nan=False):
    x = np.asarray(x)
    reduce = np.nanmax if ignore_nan else np.max
    return reduce(np.abs(x)) if x.size else 0.0


def check_error(f, r, exact, bound, *, ignore_nan=False):
    # Native exact(:)-r(:) uses MATLAB column-major flattening.
    error = maximum_abs(np.asarray(r).ravel(order='F')
                        - np.asarray(exact).ravel(order='F'), ignore_nan=ignore_nan)
    print('roots', np.asarray(r).tolist(), 'error', error, 'bound', bound,
          'lengths', [p.n for p in f.funs] if hasattr(f, 'funs') else None)
    assert error < bound


@pytest.mark.parametrize('clause', range(1, 20), ids=lambda k: f'{k:02d}')
def test_original_predicate(clause):
    if clause == 1:
        m = 1000
        f = cj.chebfun(lambda x: jnp.sin(m * jnp.pi * x), domain=(0., 1.))
        r = np.asarray(f.roots())
        assert len(r) == m + 1
        check_error(f, r, np.linspace(0, 1, m + 1), 100 * EPS)
    elif clause == 2:
        f = cj.chebfun('(x-.1).*(x+.9).*x.*(x-.9) + 1e-14*x.^5')
        r = np.asarray(f.roots())
        assert len(r) == 4
        assert maximum_abs(f(jnp.asarray(r))) < 1000 * EPS
    elif clause in (3, 4):
        fs = cj.chebfun([-1., 2.], domain=(-2., 0., 1.))
        fh = cj.chebfun([-1., 0., 1.], domain=(-2., -1., 0., 2.))
        if clause == 3:
            assert fs.roots(nojump=True).size == 0
            assert fh.roots(nojump=True, nozerofun=True).size == 0
        else:
            rs = np.asarray(fs.roots())
            rh = np.asarray(fh.roots(nozerofun=True))
            assert rs.size == 1 and rs[0] == 0
            assert rh.size == 2 and rh[0] == -1 and rh[1] == 0
    elif clause in (5, 6, 8, 9):
        factory = cj.quasimatrix if clause in (6, 9) else cj.chebfun
        if clause in (5, 6):
            f = factory(lambda x: jnp.stack((jnp.sin(2*jnp.pi*x), jnp.sign(x),
                       x*x-.5, 1+0*x), axis=-1), domain=(-1., 0., 1.), extrapolate=True)
            exact = np.full((5, 4), np.nan)
            exact[:, 0] = np.linspace(-1, 1, 5)
            exact[0, [1, 3]] = [0, 1]
            exact[[0, 1], 2] = np.asarray([-1, 1])/np.sqrt(2)
            r = np.asarray(f.roots())
            assert r.shape == (5, 4)
            check_error(f, r, exact, 10*EPS, ignore_nan=True)
        else:
            f = factory(lambda x: jnp.stack((1j*x, x, jnp.sin(2*jnp.pi*x)), axis=-1))
            exact = np.full((5, 3), np.nan)
            exact[0, :2] = [0, 0]
            exact[:, 2] = np.linspace(-1, 1, 5)
            r = np.asarray(f.roots())
            # Source checks size(exact), then subtracts actual r.
            assert exact.shape == (5, 3)
            check_error(f, r, exact, 100*EPS, ignore_nan=True)
    elif clause == 7:
        f = cj.chebfun(lambda x: besselj(0, x), domain=(0., 100.))
        r = np.asarray(f.roots())
        exact = np.asarray([
            2.40482555769577276862163,5.52007811028631064959660,
            8.65372791291101221695437,11.7915344390142816137431,
            14.9309177084877859477626,18.0710639679109225431479,
            21.2116366298792589590784,24.3524715307493027370579,
            27.4934791320402547958773,30.6346064684319751175496,
            33.7758202135735686842385,36.9170983536640439797695,
            40.0584257646282392947993,43.1997917131767303575241,
            46.3411883716618140186858,49.4826098973978171736028,
            52.6240518411149960292513,55.7655107550199793116835,
            58.9069839260809421328344,62.0484691902271698828525,
            65.1899648002068604406360,68.3314693298567982709923,
            71.4729816035937328250631,74.6145006437018378838205,
            77.7560256303880550377394,80.8975558711376278637723,
            84.0390907769381901578795,87.1806298436411536512617,
            90.3221726372104800557177,93.4637187819447741711905,
            96.6052679509962687781216,99.7468198586805964702799])
        assert len(r) == 32
        check_error(f, r, exact, 100*100*EPS)
    elif clause == 10:
        f = cj.chebfun(lambda x: (x+2)**(-.5)*jnp.cos(30*x), domain=(-2., 7.),
                       exps=(-.5, 0.), splitting=True)
        r = np.asarray(f.roots())
        exact = (np.arange(-19, 67)+.5)*np.pi/30
        check_error(f, r, exact, 100*float(f.vscale)*EPS)
    elif clause in (11, 12, 13):
        if clause == 11:
            f = cj.chebfun(lambda x: (1-jnp.exp(-x*x))/x, domain=(-jnp.inf, jnp.inf))
            r = np.asarray(f.roots())
            r = r[np.isfinite(r)]
            exact = 0.
        elif clause == 12:
            f = cj.chebfun(lambda x: x*x*(1-jnp.exp(-x*x))-2,
                           domain=(-jnp.inf, jnp.inf), exps=(2., 2.))
            r = f.roots()
            exact = [-1.4962104914103104707,1.4962104914103104707]
        else:
            f = cj.chebfun(lambda x: .15+jnp.sin(10*x)/jnp.exp(x), domain=(0.,jnp.inf))
            r = f.roots()
            exact = [.33529141416564289113,.60061694515161002799,
                     .98375750309184861332,1.2042605667187311146,
                     1.6619482204330474390,1.7760894757659030239]
        check_error(f, r, exact, 100*EPS*float(f.vscale))
    elif clause in (14, 15, 16):
        if clause == 14:
            f = cj.chebfun('exp(x)')
            exact = EPS
            bound = 10*EPS
        elif clause == 15:
            f = cj.chebfun('exp(1e-50*x)', domain=(-1e50,1e50))
            exact = 1e50*EPS
            bound = 1e51*EPS
        else:
            f = cj.chebfun('exp(1e50*x)', domain=(-1e-50,1e-50))
            exact = 1e-50*EPS
            bound = 1e-49*EPS
        r = np.asarray((f-1).roots())
        print('roots',r.tolist(),'exact',exact,'bound',bound)
        assert r.size == 1
        assert np.all(np.abs(r-exact) < bound)
    elif clause in (17, 18):
        expr = ('1i+' if clause == 18 else '')+'cos(5*x).*exp(cos(x))'
        f = cj.chebfun(expr, domain=(-np.pi,np.pi), periodic=True)
        r = f.roots()
        if clause == 18:
            assert r.size == 0
        else:
            exact = np.arange(-9,10,2)*np.pi/10
            check_error(f,r,exact,100*EPS*float(f.vscale))
    else:
        f = cj.chebfun(lambda x: jnp.sin(jnp.exp(2*jnp.tanh(jnp.sin(10*x)))),
                       domain=(0.,10.), splitting=True, split_max_length=300)
        r = f.roots()
        print('count',r.size,'lengths',[p.n for p in f.funs], 'breaks',f.domain.breakpoints)
        assert r.size == 32
