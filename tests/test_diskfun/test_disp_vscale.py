"""MATLAB-format disp() for diskfun/chebfun2 and Diskfun.vscale (Fable 5).

Provenance
----------
MATLAB source : @diskfun/disp.m, @chebfun2/disp.m, @separableApprox/vscale.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.diskfun.diskfun import Diskfun

jax.config.update("jax_enable_x64", True)


def test_diskfun_vscale_and_disp():
    f = Diskfun.from_function(lambda t, r: 2 + r ** 2 * jnp.cos(2 * t))
    assert abs(f.vscale() - 3.0) < 1e-12
    lines = f.disp().split("\n")
    assert lines[0] == "     diskfun object "
    assert lines[1] == "       domain        rank    vertical scale"
    assert lines[2] == "      unit disk   %6i          %3.2g" % (f.rank, 3.0)


def test_chebfun2_disp():
    f = Chebfun2.from_function(lambda x, y: x * y + 1.0, domain=(-5, 5, -5, 5))
    lines = f.disp().split("\n")
    assert lines[0] == "   chebfun2 object"
    assert lines[2] == "[  -5,   5] x [  -5,   5]        2     [  26  -24  -24   26]"
    assert lines[3] == "vertical scale =  26 "
