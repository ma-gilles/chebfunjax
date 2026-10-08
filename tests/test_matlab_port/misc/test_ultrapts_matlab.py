"""All40 original ultrapts predicates; 1D vectors adapt MATLAB row/column outputs.

Provenance
----------
MATLAB source : tests/misc/test_ultrapts.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import pytest
from jax.scipy.special import gammaln

from chebfunjax.utils.quadrature import ultrapts

TOL = 1e-14


@pytest.fixture(scope="module")
def rules():
    return {}


def rule(cache, n, lam, interval=None, method=None):
    key = (n, lam, interval, method)
    if key not in cache:
        cache[key] = ultrapts(n, lam, interval, method, bary=True)
    return cache[key]


@pytest.mark.parametrize("clause", range(1, 41))
def test_source(clause, rules):
    if clause <= 10:
        x, w, v = rule(rules, 42, 0.3, (0.0, 10.0) if clause >= 7 else None)
        checks = {
            1: lambda: x.shape == (42,),
            2: lambda: x.shape == w.shape == v.shape == (42,),
            3: lambda: abs(w @ x) < TOL and abs(w @ (x * x) - 0.8843414686338345) < TOL,
            4: lambda: abs(x[36] - 0.9131896381993957) < TOL,
            5: lambda: abs(w[36] - 0.04332670514309510) < TOL,
            6: lambda: abs(v[36] - 0.3115587460502451) < TOL,
            7: lambda: (
                abs(w @ x - 30.1957169274024) < 31 * TOL
                and abs(w @ (x * x) - 209.0472710358626) < 300 * TOL
            ),
            8: lambda: abs(x[37] - 9.704505777068543) < TOL,
            9: lambda: abs(w[37] - 0.1018229378664342) < TOL,
            10: lambda: abs(v[37] + 0.2449177929215358) < TOL,
        }
    elif clause <= 20:
        x, w, v = rule(rules, 251, 0.8, (0.0, 10.0) if clause >= 17 else None)
        checks = {
            11: lambda: x.shape == (251,),
            12: lambda: x.shape == w.shape == v.shape == (251,),
            13: lambda: abs(w @ x) < TOL and abs(w @ (x * x) - 0.4744211549960596) < TOL,
            14: lambda: abs(x[36] + 0.8958806879214126) < TOL,
            15: lambda: abs(w[36] - 0.003406945649865882) < TOL,
            16: lambda: abs(v[36] - 0.2321704534650446) < TOL,
            17: lambda: (
                abs(w @ x - 112.1472319135050) < 100 * TOL
                and abs(w @ (x * x) - 716.4962038918374) < 100 * TOL
            ),
            18: lambda: abs(x[37] - 0.5486606034997460) < TOL,
            19: lambda: abs(w[37] - 0.04655102134393607) < TOL,
            20: lambda: abs(v[37] + 0.2427562206703888) < TOL,
        }
    elif clause <= 26:
        x, w, v = rule(rules, 551, 7.0, method="asy")
        checks = {
            21: lambda: x.shape == (551,),
            22: lambda: x.shape == w.shape == v.shape == (551,),
            23: lambda: abs(w @ x) < TOL and abs(w @ (x * x) - (429 / 32768) * jnp.pi) < TOL,
            24: lambda: abs(x[36] + 0.9748144265829347) < TOL,
            25: lambda: abs(w[36] - 4.244751593204416e-12) < TOL,
            26: lambda: abs(v[36] - 6.123401324799126e-6) < TOL,
        }
    elif clause <= 30:
        x, w, v = rule(rules, 5000, 25.5, method="asy")
        checks = {
            27: lambda: x.shape == (5000,),
            28: lambda: x.shape == w.shape == v.shape == (5000,),
            29: lambda: (
                abs(w @ x) < TOL and abs(w @ (x * x) - 281474976710656 / 42710983650155457) < TOL
            ),
            30: lambda: abs(jnp.sum(w) - 281474976710656 / 805867616040669) < TOL,
        }
    elif clause <= 34:
        x, w, v = rule(rules, 1, 0.6, (-10.0, 3.0) if clause >= 33 else None)
        checks = {
            31: lambda: x[0] == 0,
            32: lambda: abs(w[0] - 1.887181162535959) < TOL,
            33: lambda: abs((x[0] + 10) + (x[0] - 3)) < TOL,
            34: lambda: abs((w @ x) + 62.42774750787926) < 2 * TOL,
        }
    else:
        x, w, v = rule(rules, 2, 0.6, (-10.0, 3.0) if clause >= 37 else None)
        checks = {
            35: lambda: jnp.all(abs(x - jnp.array([-1.0, 1.0]) * jnp.sqrt(5) / 4) < TOL),
            36: lambda: jnp.all(
                abs(w - 0.5 * jnp.exp(gammaln(1.1) - gammaln(1.6)) * jnp.sqrt(jnp.pi)) < TOL
            ),
            37: lambda: abs(jnp.sum(w) - 17.83649928796550) < TOL,
            38: lambda: abs(w @ x + 62.42774750787926) < 10 * TOL,
            39: lambda: abs(w @ (x * x) - 453.9946459389969) < 100 * TOL,
            40: lambda: abs(w @ (x**3) + 3237.463968416426) < 100 * TOL,
        }
    assert checks[clause]()
