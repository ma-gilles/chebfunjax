"""The 40-function battery shared by tests/chebfun3/test_battery.m and
test_chebfun3f.m (Fable 5), with MATLAB's reference values of sum3 on
[0, 1]^3."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

pi = np.pi
s2, s3 = np.sqrt(2), np.sqrt(3)

BATTERY = [
    lambda x, y, z: jnp.cos(pi * x * y * z),
    lambda x, y, z: jnp.cos(2 * pi * x * y * z),
    lambda x, y, z: jnp.cos(3 * pi * x * y * z),
    lambda x, y, z: jnp.cos(4 * pi * x * y * z),
    lambda x, y, z: jnp.cos(5 * pi * x * y * z),
    lambda x, y, z: jnp.cos(6 * pi * x * y * z),
    lambda x, y, z: jnp.cos(7 * pi * x * y * z),
    lambda x, y, z: jnp.sin(pi * x * y * z),
    lambda x, y, z: jnp.sin(8 * pi * x * (1 - x) * y * (1 - y) * z * (1 - z)),
    lambda x, y, z: jnp.sin(8 * pi * x * (1 - x) * y * (1 - y) * z * (1 - z) * (x - y - z) ** 2),
    lambda x, y, z: jnp.cos(0 * pi * (x - y - z) ** 2),
    lambda x, y, z: jnp.cos(pi * (x - y - z) ** 2),
    lambda x, y, z: jnp.cos(2 * pi * (x - y - z) ** 2),
    lambda x, y, z: jnp.exp(jnp.sin(4 * pi / (1 + x)) * jnp.sin(4 * pi / (1 + y)) * jnp.sin(4 * pi / (1 + z))),
    lambda x, y, z: jnp.log(1 + x * y * z),
    lambda x, y, z: jnp.cos(pi * x * jnp.sin(pi * y)) + jnp.cos(pi * y * z * jnp.sin(pi * x * z)),
    lambda x, y, z: jnp.cos(2 * pi * x * jnp.sin(pi * y) * jnp.cos(pi * z)) + jnp.cos(2 * pi * y * jnp.sin(pi * x)),
    lambda x, y, z: (1 - x * y * z) / (1 + x ** 2 + y ** 2),
    lambda x, y, z: jnp.cos(pi * x * y ** 2 * z ** 3) * jnp.cos(pi * z * y ** 2 * x ** 3),
    lambda x, y, z: jnp.cos(2 * pi * x * y ** 2 * z ** 3) * jnp.cos(2 * pi * y * x ** 2 * z ** 3),
    lambda x, y, z: jnp.cos(3 * pi * x * y ** 2 * (1 + z)) * jnp.cos(3 * pi * y * x ** 2 * (1 + z)),
    lambda x, y, z: (x - y) / (3 - x ** 2 + z ** 2),
    lambda x, y, z: jnp.exp(-y * x ** 2 * z) + jnp.exp(-x * y ** 2 * z),
    lambda x, y, z: jnp.exp((1 - x ** 2 * z) / (1 + y ** 2 * z)) + jnp.exp((1 - y ** 2 * z) / (1 + x ** 2 * z)),
    lambda x, y, z: 10.0 ** (-x * y * z),
    lambda x, y, z: 10.0 ** (-10 * x * y * z),
    lambda x, y, z: jnp.sin(x + y + z),
    lambda x, y, z: jnp.exp(x + y + z) * jnp.cos(x + y + z),
    lambda x, y, z: jnp.cos(pi / 4 + x + y + z),
    lambda x, y, z: jnp.cos(pi / 3 + 5 * x + 5 * y + 5 * z),
    lambda x, y, z: (x + y + z) ** 12,
    lambda x, y, z: jnp.exp(x ** 2 * y ** 2 * z ** 2) * jnp.cos(x + y + z),
    lambda x, y, z: jnp.sin(x * y * z) + x + y + z,
    lambda x, y, z: jnp.cos(100 * x) - jnp.cos(100 * y) - jnp.cos(100 * z),
    lambda x, y, z: x - y + z,
    lambda x, y, z: -jnp.sin(x) * (jnp.sin(x ** 2 / pi) ** 2 - jnp.sin(y) * jnp.sin(y ** 2 / pi) ** 2) - jnp.sin(z) * jnp.sin(z ** 2 / pi) ** 2,
    lambda x, y, z: -x * jnp.sin(jnp.sqrt(jnp.abs(0.2 + x))) - y * jnp.sin(jnp.sqrt(jnp.abs(0.3 + y))) - z * jnp.sin(jnp.sqrt(jnp.abs(0.1 + z))),
    lambda x, y, z: (x ** 2 + y ** 2 + z ** 2) / 4000 - jnp.cos(x) * jnp.cos(y / s2) * jnp.cos(z / s3) + 1,
    lambda x, y, z: jnp.exp(1.0 / (1 + (((x + 1) / 2) * ((y + 1) / 2) * (z + 1) / 2) ** 2)),
    lambda x, y, z: 2 * jnp.exp(jnp.exp((x + 1) / 2)),
]

EXACT = np.array([
    0.8461106227350253, 0.6052157118483288, 0.4697661749231274,
    0.3886397051915788, 0.3330920107414529, 0.2928314960173565,
    0.2619772790103158, 0.3226674912855009, 0.1153949594968427,
    0.0463743346003271, 1.0, 0.3854484437166375, 0.2595975105701650,
    1.0700418045865364, 0.1103040719136995, 1.1941278093664705,
    0.4916086077355209, 0.5741043338997425, 0.9381453489074456,
    0.8830048694732021, 0.3411136775418946, 0.0101828389788635,
    1.8529116285910792, 4.2715143503929390, 0.7859343211742496,
    0.3352208605577725, 0.8793549306454008, -0.801590008944990,
    -0.577703137905825, -0.008766419956637, 5220.0021978021978,
    0.0366931467095498, 1.6224340287967378, 0.0050636564110975, 0.5,
    -0.022434416365347, -1.181086511303566, 0.2694080277004217,
    2.3340368233541618, 17.816513659679896,
])

DOM = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)


def _sum3_isolated(jj, use_fiber, flag):
    """Run one battery member in a fresh process: the 40 adaptive 3D
    constructions in one interpreter exhaust the XLA JIT code arena
    ("Failed to materialize symbols"), which is a process-level resource
    limit, not a numerical failure."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        return pool.apply(_sum3_worker, (jj, use_fiber, flag))


def _sum3_worker(jj, use_fiber, flag):
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    jax.config.update("jax_enable_x64", True)
    from chebfunjax.chebfun3d.chebfun3 import chebfun3
    from tests.test_matlab_port.chebfun3._battery import BATTERY, DOM
    out = []
    if flag == "chebfun3f":
        g = chebfun3(BATTERY[jj], DOM, chebfun3f=True)
        return [float(g.sum3())]
    g = chebfun3(BATTERY[jj], DOM)
    out.append(float(g.sum3()))
    h = chebfun3(lambda x, y, z: g(x, y, z), DOM)
    out.append(float(h.sum3()))
    for k in (1, 2, 3):
        hk = chebfun3(lambda x, y, z: g(x, y, z), DOM, fiberDim=k)
        out.append(float(hk.sum3()))
    return out
