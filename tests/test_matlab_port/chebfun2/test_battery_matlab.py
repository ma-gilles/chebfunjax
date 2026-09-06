"""Port of MATLAB Chebfun tests/chebfun2/test_battery.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_battery.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebpref import ChebfunPref

jax.config.update("jax_enable_x64", True)

pi = np.pi
BATTERY = [
    lambda x, y: jnp.cos(pi * x * y),
    lambda x, y: jnp.cos(2 * pi * x * y),
    lambda x, y: jnp.cos(3 * pi * x * y),
    lambda x, y: jnp.cos(4 * pi * x * y),
    lambda x, y: jnp.cos(5 * pi * x * y),
    lambda x, y: jnp.cos(6 * pi * x * y),
    lambda x, y: jnp.cos(7 * pi * x * y),
    lambda x, y: jnp.sin(pi * x * y),
    lambda x, y: jnp.sin(8 * pi * x * (1 - x) * y * (1 - y)),
    lambda x, y: jnp.sin(8 * pi * x * (1 - x) * y * (1 - y) * (x - y) ** 2),
    lambda x, y: jnp.cos(0 * pi * (x - y) ** 2),
    lambda x, y: jnp.cos(pi * (x - y) ** 2),
    lambda x, y: jnp.cos(2 * pi * (x - y) ** 2),
    lambda x, y: jnp.exp(jnp.sin(4 * pi / (1 + x)) * jnp.sin(4 * pi / (1 + y))),
    lambda x, y: jnp.log(1 + x * y),
    lambda x, y: jnp.cos(pi * x * jnp.sin(pi * y)) + jnp.cos(pi * y * jnp.sin(pi * x)),
    lambda x, y: jnp.cos(2 * pi * x * jnp.sin(pi * y)) + jnp.cos(2 * pi * y * jnp.sin(pi * x)),
    lambda x, y: (1 - x * y) / (1 + x ** 2 + y ** 2),
    lambda x, y: jnp.cos(pi * x * y ** 2) * jnp.cos(pi * y * x ** 2),
    lambda x, y: jnp.cos(2 * pi * x * y ** 2) * jnp.cos(2 * pi * y * x ** 2),
    lambda x, y: jnp.cos(3 * pi * x * y ** 2) * jnp.cos(3 * pi * y * x ** 2),
    lambda x, y: (x - y) / (2 - x ** 2 + y ** 2) + (y - x) / (2 - y ** 2 + x ** 2),
    lambda x, y: jnp.exp(-y * x ** 2) + jnp.exp(-x * y ** 2),
    lambda x, y: jnp.exp((1 - x ** 2) / (1 + y ** 2)) + jnp.exp((1 - y ** 2) / (1 + x ** 2)),
    lambda x, y: 10.0 ** (-x * y),
    lambda x, y: 10.0 ** (-10 * x * y),
    lambda x, y: jnp.sqrt(1e-15) ** ((x + y) ** 6),
    lambda x, y: jnp.sin(x + y),
    lambda x, y: jnp.exp(x + y) * jnp.cos(x + y),
    lambda x, y: jnp.cos(pi / 3 + x + y),
    lambda x, y: jnp.cos(pi / 3 + 5 * x + 5 * y),
    lambda x, y: (x + y) ** 4,
    lambda x, y: (x + y) ** 8,
    lambda x, y: (x + y) ** 12,
    lambda x, y: jnp.exp(x ** 2 * y ** 2) * jnp.cos(x + y),
    lambda x, y: jnp.sin(x * y),
    lambda x, y: jnp.sin(x * y) + x + y,
    lambda x, y: (1 + x + y) ** (-3),
    lambda x, y: (1 + 5 * x + 5 * y) ** (-3),
    lambda x, y: jnp.cos(pi * x * jnp.sin(pi * y)) - jnp.cos(pi * y * jnp.sin(pi * x)),
    lambda x, y: jnp.exp(-y * x ** 2) - jnp.exp(-x * y ** 2),
    lambda x, y: jnp.exp((1 - x ** 2) / (1 + y ** 2)) - jnp.exp((1 - y ** 2) / (1 + x ** 2)),
    lambda x, y: jnp.cos(100 * x) - jnp.cos(100 * y),
    lambda x, y: x - y,
    lambda x, y: -jnp.sin(x) * (jnp.sin(x ** 2 / pi) ** 2 - -jnp.sin(y) * jnp.sin(y ** 2 / pi) ** 2),
    lambda x, y: jnp.cos(3 * pi * x * y ** 2) * jnp.cos(3 * pi * y * x ** 2) + jnp.sin(x),
    lambda x, y: jnp.cos(10 * x * (y - 1)),
    lambda x, y: (x - y) ** 2 * jnp.cos(x) * jnp.log(1 + y),
    lambda x, y: jnp.cos(x + y + jnp.log(1 + x * y)),
    lambda x, y: jnp.cos(100 * x),
    lambda x, y: x * (1 - y + jnp.cos(10 * x)),
    lambda x, y: -x * jnp.sin(jnp.sqrt(jnp.abs(x))) - y * jnp.sin(jnp.sqrt(jnp.abs(y))),
    lambda x, y: (x ** 2 + y ** 2) / 4000 - jnp.cos(x) * jnp.cos(y / jnp.sqrt(2.0)) + 1,
    lambda x, y: (jnp.exp(-(x - 3) ** 2 / pi - (y - 5) ** 2 / pi) * jnp.cos(pi * (x - 3) ** 2 + pi * (y - 5) ** 2)
                  + 2 * jnp.exp(-(x - 5) ** 2 / pi - (y - 2) ** 2 / pi) * jnp.cos(pi * (x - 5) ** 2 + pi * (y - 2) ** 2)
                  + 5 * jnp.exp(-(x - 2) ** 2 / pi - (y - 1) ** 2 / pi) * jnp.cos(pi * (x - 2) ** 2 + pi * (y - 1) ** 2)
                  + 2 * jnp.exp(-(x - 1) ** 2 / pi - (y - 4) ** 2 / pi) * jnp.cos(pi * (x - 1) ** 2 + pi * (y - 4) ** 2)
                  + 3 * jnp.exp(-(x - 7) ** 2 / pi - (y - 9) ** 2 / pi) * jnp.cos(pi * (x - 7) ** 2 + pi * (y - 9) ** 2)),
    lambda x, y: (4 - 2.1 * (x / 3) ** 2 + (x / 3) ** 4 / 3) * (x / 3) ** 2 + (x / 3) * (y / 3) + (-4 - 4 * (y / 3) ** 2) * y ** 2,
    lambda x, y: -(1 + jnp.cos(12 * jnp.sqrt((x / 5) ** 2 + (y / 5) ** 2))) / (.5 * ((x / 5) ** 2 + (y / 5) ** 2) + 2),
    lambda x, y: jnp.cos(x * (1 + y)),
    lambda x, y: jnp.exp(-1000 * ((x - .5) ** 2 + (y - .5) ** 2 - .125) ** 2),
    lambda x, y: jnp.exp(1.0 / (1 + (((x + 1) / 2) * ((y + 1) / 2)) ** 2)),
    lambda x, y: 2 * jnp.exp(jnp.exp((x + 1) / 2)),
    lambda x, y: (1.0 / (1 + 1e-3 * (((x + 1) / 2 - .3) ** 2 + ((y + 1) / 2 - .3) ** 2))),
]
EXACT = [
    0.5894898722360821, 0.2257058333950713, 0.1776977458727282, 0.1187424174709143,
    0.1040214328382758, 0.08053420291629177, 0.07348799958904405, 0.5246630675753145,
    0.5735519176658511, 0.06955139313890799, 1, 0.7479656668315220, 0.4882534060753284,
    1.171460474510744, 0.2087613945440329, 0.857861895707081, 0.241651766729031165,
    0.5086983159291771, 0.7231765864851378, 0.5891599692839595, 0.4769961059000557,
    0.09422407514560267, 1.723055413592619, 3.492035304274583, 0.6269639737246870,
    0.1612897266836097, 0.1727371891, 0.7736445427901117, 1.072069561535282,
    -0.4216201034137229, 0.05571869648415638, 2.066666666666667, 11.35555555555556,
    90.01098901098886, 0.5014624050192373, 0.2398117420005848, 1.239811742000715,
    0.1666666666666678, 0.01515151515151557, 0, 0, 0, 0, 0, -0.021260414733480,
    0.9366938000318889, 0.1658347594218874, 0.05658954650254417, 0.30723021414730670,
    -0.005063656411097588, 0.1772071736202985, -0.7083943110937996, 0.2270859249178878,
    0.230747707969006, -1.251416160428506, -0.3882055297584357, 0.6593299064355118,
    0.1760859897887420, 2.146572044542719, 17.81651365967990, 0.9995535671520096,
]


class TestChebfun2Battery:
    def test_all_matlab_assertions(self):
        tol = 1e8 * ChebfunPref().cheb2Prefs.chebfun2eps
        assert len(BATTERY) == len(EXACT)
        failures = []
        for jj, (F, ex) in enumerate(zip(BATTERY, EXACT)):
            g = chebfun2(F, domain=(0.0, 1.0, 0.0, 1.0))
            err = abs(float(g.sum2()) - ex)
            if not err < tol:
                failures.append((jj + 1, err))
        assert not failures, failures
