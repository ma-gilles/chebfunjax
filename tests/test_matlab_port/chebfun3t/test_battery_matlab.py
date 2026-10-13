"""Literal port of MATLAB tests/chebfun3t/test_battery.m (all 40 cases).

Each dense Chebfun3T is integrated over [0,1]^3 and compared with the
source's symbolic reference integral using its original ``1e8*chebfun3eps``
bound.

Provenance
----------
MATLAB source : tests/chebfun3t/test_battery.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d import chebfun3t

EPS = float(np.finfo(np.float64).eps)
TOL = 1e8 * EPS

# MATLAB source tests/chebfun3t/test_battery.m: Battery and exact vectors.
_CASES = [
    (1, lambda x, y, z: jnp.cos(jnp.pi*x*y*z), 0.8461106227350253),
    (2, lambda x, y, z: jnp.cos(2*jnp.pi*x*y*z), 0.6052157118483288),
    (3, lambda x, y, z: jnp.cos(3*jnp.pi*x*y*z), 0.4697661749231274),
    (4, lambda x, y, z: jnp.cos(4*jnp.pi*x*y*z), 0.3886397051915788),
    (5, lambda x, y, z: jnp.cos(5*jnp.pi*x*y*z), 0.3330920107414529),
    (6, lambda x, y, z: jnp.cos(6*jnp.pi*x*y*z), 0.2928314960173565),
    (7, lambda x, y, z: jnp.cos(7*jnp.pi*x*y*z), 0.2619772790103158),
    (8, lambda x, y, z: jnp.sin(jnp.pi*x*y*z), 0.3226674912855009),
    (9, lambda x, y, z: jnp.sin(8*jnp.pi*x*(1-x)*y*(1-y)*z*(1-z)),
     0.1153949594968427),
    (10, lambda x, y, z: jnp.sin(8*jnp.pi*x*(1-x)*y*(1-y)*z*(1-z)*(x-y-z)**2),
     0.0463743346003271),
    (11, lambda x, y, z: jnp.cos(0*jnp.pi*(x-y-z)**2), 1.0),
    (12, lambda x, y, z: jnp.cos(jnp.pi*(x-y-z)**2), 0.3854484437166375),
    (13, lambda x, y, z: jnp.cos(2*jnp.pi*(x-y-z)**2), 0.2595975105701650),
    (14, lambda x, y, z: jnp.exp(jnp.sin(4*jnp.pi/(1+x))*jnp.sin(4*jnp.pi/(1+y))*jnp.sin(4*jnp.pi/(1+z))),
     1.0700418045865364),
    (15, lambda x, y, z: jnp.log(1+x*y*z), 0.1103040719136995),
    (16, lambda x, y, z: jnp.cos(jnp.pi*x*jnp.sin(jnp.pi*y))+jnp.cos(jnp.pi*y*z*jnp.sin(jnp.pi*x*z)),
     1.1941278093664705),
    (17, lambda x, y, z: jnp.cos(2*jnp.pi*x*jnp.sin(jnp.pi*y)*jnp.cos(jnp.pi*z))+jnp.cos(2*jnp.pi*y*jnp.sin(jnp.pi*x)),
     0.4916086077355209),
    (18, lambda x, y, z: (1-x*y*z)/(1+x**2+y**2), 0.5741043338997425),
    (19, lambda x, y, z: jnp.cos(jnp.pi*x*y**2*z**3)*jnp.cos(jnp.pi*z*y**2*x**3),
     0.9381453489074456),
    (20, lambda x, y, z: jnp.cos(2*jnp.pi*x*y**2*z**3)*jnp.cos(2*jnp.pi*y*x**2*z**3),
     0.8830048694732021),
    (21, lambda x, y, z: jnp.cos(3*jnp.pi*x*y**2*(1+z))*jnp.cos(3*jnp.pi*y*x**2*(1+z)),
     0.3411136775418946),
    (22, lambda x, y, z: (x-y)/(3-x**2+z**2), 0.0101828389788635),
    (23, lambda x, y, z: jnp.exp(-y*x**2*z)+jnp.exp(-x*y**2*z),
     1.8529116285910792),
    (24, lambda x, y, z: jnp.exp((1-x**2*z)/(1+y**2*z))+jnp.exp((1-y**2*z)/(1+x**2*z)),
     4.2715143503929390),
    (25, lambda x, y, z: 10.0**(-x*y*z), 0.7859343211742496),
    (26, lambda x, y, z: 10.0**(-10*x*y*z), 0.3352208605577725),
    (27, lambda x, y, z: jnp.sin(x+y+z), 0.8793549306454008),
    (28, lambda x, y, z: jnp.exp(x+y+z)*jnp.cos(x+y+z), -0.801590008944990),
    (29, lambda x, y, z: jnp.cos(jnp.pi/4+x+y+z), -0.577703137905825),
    (30, lambda x, y, z: jnp.cos(jnp.pi/3+5*x+5*y+5*z), -0.008766419956637),
    (31, lambda x, y, z: (x+y+z)**12, 5220.0021978021978),
    (32, lambda x, y, z: jnp.exp(x**2*y**2*z**2)*jnp.cos(x+y+z), 0.0366931467095498),
    (33, lambda x, y, z: jnp.sin(x*y*z)+x+y+z, 1.6224340287967378),
    (34, lambda x, y, z: jnp.cos(100*x)-jnp.cos(100*y)-jnp.cos(100*z), 0.0050636564110975),
    (35, lambda x, y, z: x-y+z, 0.5),
    (36, lambda x, y, z: -jnp.sin(x)*(jnp.sin(x**2/jnp.pi)**2-jnp.sin(y)*jnp.sin(y**2/jnp.pi)**2)-jnp.sin(z)*jnp.sin(z**2/jnp.pi)**2,
     -0.022434416365347),
    (37, lambda x, y, z: -x*jnp.sin(jnp.sqrt(jnp.abs(0.2+x)))-y*jnp.sin(jnp.sqrt(jnp.abs(0.3+y)))-z*jnp.sin(jnp.sqrt(jnp.abs(0.1+z))),
     -1.181086511303566),
    (38, lambda x, y, z: (x**2+y**2+z**2)/4000-jnp.cos(x)*jnp.cos(y/jnp.sqrt(2))*jnp.cos(z/jnp.sqrt(3))+1,
     0.2694080277004217),
    (39, lambda x, y, z: jnp.exp(1/(1+(((x+1)/2)*((y+1)/2)*(z+1)/2)**2)),
     2.3340368233541618),
    (40, lambda x, y, z: 2*jnp.exp(jnp.exp((x+1)/2)), 17.816513659679896),
]


@pytest.mark.slow
class TestChebfun3tBattery:
    def test_all_matlab_assertions(self):
        dom = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        for idx, fn, exact in _CASES:
            g = chebfun3t(fn, dom)
            err = abs(float(g.sum3()) - exact)
            assert err < TOL, f"battery #{idx}: sum3 err {err} !< {TOL}"
