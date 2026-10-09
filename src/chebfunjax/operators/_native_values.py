"""Private native first-kind operator-stack realization.

Provenance
----------
MATLAB source: @valsDiscretization/instantiate.m, @chebcolloc/{diff,sum,feval}.m,
    @chebcolloc1/functionPoints.m, @functionalBlock/functionalBlock.m (promote).
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.

This capability preserves each intermediate grid multiplication. It neither
converts a C2 matrix nor changes differential coefficient assembly.
"""
from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp

from chebfunjax.utils.diffmat import _cheb1_barywts, diffmat
from chebfunjax.utils.interpolation import barymat
from chebfunjax.utils.quadrature import chebpts, chebweights


@dataclass(frozen=True)
class Capability:
    realize: Callable
    domain: tuple

    def __post_init__(self):
        object.__setattr__(self, 'domain', tuple(float(x) for x in self.domain))


class FirstKindDisc:
    """Input grid of one source column, including all subintervals."""

    def __init__(self, dimensions, domain):
        self.sizes = tuple(int(n) for n in dimensions)
        self.domain = tuple(domain)
        if len(self.sizes) != len(self.domain)-1:
            raise ValueError("First-kind dimensions must match domain intervals")
        self.intervals = tuple(zip(self.domain[:-1], self.domain[1:]))
        self.n = sum(self.sizes)
        self.points = jnp.concatenate([
            b*(points+1)/2+a*(1-points)/2
            for n, (a, b) in zip(self.sizes, self.intervals)
            for points in [chebpts(n, kind=1)]])

    def evaluation(self, location, direction):
        # @opDiscretization/whichInterval short-circuits single intervals.
        if len(self.sizes) == 1:
            index = 0
        else:
            index = max(i for i, a in enumerate(self.domain) if location >= a)
            if index == len(self.domain)-1:
                if direction > 0:
                    raise ValueError(
                        "CHEBFUN:OPDISCRETIZATION:whichInterval:undefined -- "
                        "Evaluation direction is undefined at the location.")
                direction = -1
            if direction < 0 and abs(location-self.domain[index]) < (
                    10*jnp.finfo(jnp.float64).eps*(self.domain[-1]-self.domain[0])):
                index -= 1
            if index < 0:
                # MATLAB then fails on interval index0 in chebcolloc/feval.
                # Python's negative index must not silently select the last piece.
                raise IndexError("Evaluation selects an interval before the domain")
        n = self.sizes[index]
        start = sum(self.sizes[:index])
        row = barymat(jnp.asarray([location]), self.points[start:start+n],
                      _cheb1_barywts(n)).reshape(-1)
        return jnp.zeros(self.n, dtype=row.dtype).at[start:start+n].set(row)


def _diagonal(blocks):
    size = sum(a.shape[0] for a in blocks)
    result = jnp.zeros((size, size), dtype=jnp.result_type(*blocks))
    start = 0
    for block in blocks:
        stop = start+block.shape[0]
        result = result.at[start:stop, start:stop].set(block)
        start = stop
    return result


def binary(left, right, operation):
    a, b = left._values_capability, right._values_capability
    if a is None or b is None:
        return None
    domain = tuple(sorted(set(a.domain).union(b.domain)))
    def action(disc):
        if operation == 'add':
            return a.realize(disc)+b.realize(disc)
        if operation == 'sub':
            return a.realize(disc)-b.realize(disc)
        return a.realize(disc) @ b.realize(disc)
    return Capability(action, domain)


def scale(block, scalar):
    a = block._values_capability
    return None if a is None else Capability(
        lambda disc: scalar*a.realize(disc), a.domain)


def promote(block):
    a = block._values_capability
    return None if a is None else Capability(
        lambda disc: jnp.tile(a.realize(disc).reshape(1, -1), (disc.n, 1)),
        a.domain)


def identity(domain):
    return Capability(lambda disc: jnp.eye(disc.n), tuple(domain))


def zero(domain, functional=False):
    return Capability(lambda disc: jnp.zeros(disc.n if functional else
                                            (disc.n, disc.n)), tuple(domain))


def derivative(domain, order):
    return Capability(lambda disc: _diagonal([
        diffmat(n, order, domain=interval, kind=1)
        for n, interval in zip(disc.sizes, disc.intervals)])
        if not hasattr(disc, "native_derivative") else disc.native_derivative(order),
        tuple(domain))


def multiplier(value, domain):
    data_domain = getattr(getattr(value, 'domain', None), 'breakpoints', ())
    domain = tuple(sorted(set(domain).union(data_domain)))
    return Capability(lambda disc: jnp.diag(jnp.broadcast_to(
        jnp.asarray(value(disc.points)).reshape(-1), (disc.n,))), domain)


def evaluation(location, domain, direction):
    return Capability(lambda disc: disc.evaluation(location, direction), tuple(domain))


def integral(domain):
    return Capability(lambda disc: jnp.concatenate([
        .5*(b-a)*chebweights(n, kind=1)
        for n, (a, b) in zip(disc.sizes, disc.intervals)])
        if not hasattr(disc, "native_integral") else disc.native_integral(),
        tuple(domain))
