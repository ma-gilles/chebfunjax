"""JAX polynomial algebra for the public Chebfun residue wrapper.

Algorithm references: MATLAB R2025b residue.m, mpoles.m, resi2.m,
poly.m, polyder.m and polydiv.m. These licensed MathWorks algorithms are
separate from the MIT Chebfun wrapper @chebfun/residue.m (7574c77).
"""

import math

import jax
import jax.numpy as jnp


def _trim(c):
    indices = jnp.flatnonzero(c != 0)
    return c[int(indices[0]) :] if indices.size else c


def _poly_step(c, item):
    index, root = item
    # The source recurrence multiplies before subtracting. Keep their
    # rounding boundary when compiling the fixed-shape scan.
    product, old_tail = jax.lax.optimization_barrier((root * c[:-1], c[1:]))
    difference = old_tail - product
    tail = jnp.where(jnp.arange(c.size - 1) <= index, difference, old_tail)
    return jnp.concatenate((c[:1], tail)), None


@jax.jit
def _poly_finite(roots):
    c = jnp.zeros(roots.size + 1, dtype=jnp.result_type(roots, 1.0)).at[0].set(1)
    c, _ = jax.lax.scan(_poly_step, c, (jnp.arange(roots.size), roots))
    return c


def _poly(roots):
    roots = roots[jnp.isfinite(roots)]
    c = _poly_finite(roots)
    if jnp.iscomplexobj(roots):
        positive = roots[jnp.imag(roots) > 0]
        negative = jnp.conj(roots[jnp.imag(roots) < 0])
        if positive.size == negative.size and bool(
            jnp.array_equal(jnp.sort(positive), jnp.sort(negative))
        ):
            c = jnp.real(c)
    return c


def _divide(b, a):
    """POLYDIV's transposed direct-form II impulse-response division."""
    if bool(a[0] == 0):
        raise ValueError("MATLAB:deconv:ZeroCoef1")
    dtype = jnp.result_type(a, b)
    if a.size > b.size:
        return jnp.zeros(1, dtype=dtype), b.astype(dtype)
    count = b.size - a.size + 1
    size = max(b.size, a.size)
    aa = jnp.pad(a / a[0], (0, size - a.size))
    bb = jnp.pad(b / a[0], (0, size - b.size))
    state = jnp.zeros(size - 1, dtype=dtype)
    q = jnp.zeros(count, dtype=dtype)
    for index in range(count):
        sample = 1 if index == 0 else 0
        value = bb[0] * sample + (state[0] if state.size else 0)
        q = q.at[index].set(value)
        if state.size:
            shifted = jnp.concatenate((state[1:], jnp.zeros(1, dtype=dtype)))
            state = shifted + bb[1:] * sample - aa[1:] * value
    remainder = jnp.zeros(b.size, dtype=dtype)
    remainder = remainder.at[count:].set(a[0] * state[: b.size - count])
    return q, remainder


def _derivative(c):
    return c[:-1] * jnp.arange(c.size - 1, 0, -1) if c.size > 1 else jnp.zeros(1, dtype=c.dtype)


def _ratio_derivative(u, v):
    first = jnp.convolve(_derivative(u), v)
    second = jnp.convolve(u, _derivative(v))
    size = max(first.size, second.size)
    numerator = jnp.pad(first, (size - first.size, 0)) - jnp.pad(second, (size - second.size, 0))
    numerator = _trim(numerator)
    if bool(jnp.all(numerator == 0)):
        numerator = jnp.zeros(1, dtype=numerator.dtype)
    denominator = _trim(jnp.convolve(v, v))
    if bool(jnp.all(denominator == 0)):
        denominator = jnp.zeros(1, dtype=denominator.dtype)
    if numerator.size > max(u.size + v.size - 2, 1):
        numerator = numerator[1:]
    return numerator, denominator


def _mpoles(p, reorder):
    order = list(range(p.size))
    if reorder:
        order = [int(i) for i in jnp.argsort(-jnp.abs(p), stable=True)]
    result, multiplicities = [], []
    while len(order) > 1:
        first = p[order[0]]
        threshold = 0.001 * jnp.abs(first) if all(bool(jnp.abs(p[i]) > 0) for i in order) else 0.001
        group = [i for i in order if bool(jnp.abs(first - p[i]) < threshold)]
        if not group:
            raise ValueError(
                "Residue pole grouping made no progress: nonfinite poles or an "
                "underflowed relative threshold."
            )
        result.extend(group)
        multiplicities.extend(range(1, len(group) + 1))
        selected = set(group)
        order = [i for i in order if i not in selected]
    # MPOLES assigns the last singleton unconditionally. A Python error
    # above replaces a non-progressing multi-pole loop, not this source exit.
    if order:
        result.append(order[0])
        multiplicities.append(1)
    return jnp.asarray(result, dtype=jnp.int64), multiplicities


def _repeated_residue(u, v, pole, multiplicity, power):
    factor = jnp.asarray([1, -pole])
    for _ in range(multiplicity):
        v, _ = _divide(v, factor)
    for _ in range(multiplicity - power):
        u, v = _ratio_derivative(u, v)
    return jnp.polyval(u, pole) / jnp.polyval(v, pole) / math.factorial(multiplicity - power)


def forward(u, v):
    """MATLAB numeric RESIDUE forward algorithm, with eager group metadata."""
    u, v = _trim(jnp.ravel(u, order="F")), _trim(jnp.ravel(v, order="F"))
    u, v = u / v[0], v / v[0]
    k = jnp.empty(0, dtype=jnp.result_type(u, v))
    if u.size >= v.size:
        k, u = _divide(u, v)
    roots = jnp.roots(v, strip_zeros=False)
    if roots.size == 0:
        return jnp.empty(0, dtype=u.dtype), roots, k
    order, _ = _mpoles(roots, True)
    p = roots[order]
    groups = []
    total, count = p[0], 1
    for index in range(1, p.size):
        if bool(jnp.abs(total / count - p[index]) <= 0.001):
            total, count = total + p[index], count + 1
        else:
            groups.append((total / count, count))
            total, count = p[index], 1
    groups.append((total / count, count))
    repeated = any(count > 1 for _, count in groups)
    if repeated:
        p = jnp.concatenate([jnp.full(count, pole) for pole, count in groups])
        v = _poly(p)
        residues = []
        for pole, count in groups:
            for power in range(1, count + 1):
                residues.append(_repeated_residue(u, v, pole, count, power))
    else:
        residues = [
            jnp.polyval(u, p[i]) / jnp.polyval(_poly(jnp.concatenate((p[:i], p[i + 1 :]))), p[i])
            for i in range(p.size)
        ]
    return jnp.stack(residues), p, k


def inverse(r, p, k):
    """MATLAB numeric RESIDUE inverse, grouping nonadjacent repeats."""
    r, p, k = jnp.ravel(r, order="F"), jnp.ravel(p, order="F"), jnp.ravel(k, order="F")
    if r.size != p.size:
        raise ValueError("Residue and pole vectors must have equal lengths.")
    order, powers = _mpoles(p, False)
    p, r = p[order], r[order]
    denominator = _poly(p)
    numerator = jnp.zeros(p.size, dtype=jnp.result_type(r, p, 1.0))
    for index, power in enumerate(powers):
        others = jnp.concatenate((p[: index - power + 1], p[index + 1 :]))
        term = _poly(others)
        numerator = numerator + r[index] * jnp.pad(term, (p.size - term.size, 0))
    if k.size and bool(jnp.any(k != 0)):
        numerator = jnp.pad(numerator, (k.size, 0)) + jnp.convolve(k, denominator)
    return numerator, denominator
