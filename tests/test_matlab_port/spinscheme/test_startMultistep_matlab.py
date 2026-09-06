"""Port of MATLAB Chebfun tests/spinscheme/test_startMultistep.m (Fable 5).

MATLAB's ``spin(S, N, dt, 'plot', 'off', 'scheme', name)`` is
``spin(S, N, dt, scheme=name)`` (the trailing MATLAB pairs are accepted
too).  ``norm(u - v, inf)`` is the max over the solution grid.

Provenance
----------
MATLAB source : tests/spinscheme/test_startMultistep.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.spinop import Spinop, spin
from chebfunjax.operators.spinop2 import Spinop2, spin2
from chebfunjax.operators.spinopsphere import Spinopsphere, spinsphere

jax.config.update("jax_enable_x64", True)


def _rel_inf_1d(u, v, dom, n=2001):
    x = jnp.linspace(dom[0], dom[1], n)
    a = np.asarray(u(x))
    b = np.asarray(v(x))
    return np.max(np.abs(a - b)) / np.max(np.abs(a))


class TestSpinschemeStartmultistep:
    def test_kdv_pecec736_vs_etdrk4(self):
        tol = 1e-10
        S = Spinop("KDV")
        dt = 1e-7
        N = 256
        S.tspan = (0.0, 10 * dt)
        umulti = spin(S, N, dt, "plot", "off", "scheme", "pecec736")
        u = spin(S, N, dt, "plot", "off", "scheme", "etdrk4")
        assert _rel_inf_1d(u, umulti, S.domain) < tol                # pass(1)

    def test_gs_pecec736_vs_etdrk4(self):
        tol = 1e-10
        S = Spinop2("GS")
        dt = 1e-2
        N = 64
        S.tspan = (0.0, 10 * dt)
        # MATLAB spinpref2 default: dealias 'off' (chebfunjax's spin2
        # dealiases by default; the 2/3-rule projection interacts with the
        # two schemes differently at the 5e-9 level).
        u = spin2(S, N, dt, "plot", "off", "dealias", "off", scheme="etdrk4")
        umulti = spin2(S, N, dt, "plot", "off", "dealias", "off",
                       scheme="pecec736")
        ax, bx, ay, by = S.domain
        xx, yy = np.meshgrid(np.linspace(ax, bx, 65), np.linspace(ay, by, 65))
        err = 0.0
        scale = 0.0
        for k in range(len(u.components) if hasattr(u, "components") else 1):
            uk = u.components[k] if hasattr(u, "components") else u
            vk = umulti.components[k] if hasattr(umulti, "components") else umulti
            a = np.asarray(uk(jnp.asarray(xx), jnp.asarray(yy)))
            b = np.asarray(vk(jnp.asarray(xx), jnp.asarray(yy)))
            err = max(err, float(np.max(np.abs(a - b))))
            scale = max(scale, float(np.max(np.abs(a))))
        assert err / scale < tol                                      # pass(2)

    def test_ac_sphere_imexbdf4_vs_lirk4(self):
        tol = 1e-10
        S = Spinopsphere("AC")
        dt = 5e-4
        N = 128
        S.tspan = (0.0, 10 * dt)
        u = spinsphere(S, N, dt, "plot", "off", scheme="lirk4")
        umulti = spinsphere(S, N, dt, "plot", "off", scheme="imexbdf4")
        lam = jnp.linspace(-np.pi, np.pi, 81)
        th = jnp.linspace(0.0, np.pi, 41)
        LL, TT = jnp.meshgrid(lam, th)
        a = np.asarray(u(LL, TT))
        b = np.asarray(umulti(LL, TT))
        assert np.max(np.abs(a - b)) / np.max(np.abs(a)) < tol        # pass(3)
