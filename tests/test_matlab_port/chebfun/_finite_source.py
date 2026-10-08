"""Literal inputs shared by Chebfun7574c77 isfinite/isinf source tests."""

import jax.numpy as jnp

import chebfunjax as cj


def source_case(case):
    if case == 1:
        return cj.chebfun(jnp.sin)
    if case in (2, 3):
        f = cj.chebfun(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
        if case == 3:
            f = f.set_point_values(f.point_values.at[0, 0].set(jnp.inf))
        return f
    if case == 4:
        return cj.chebfun(
            lambda x: jnp.sin(100 * x) * (x + 2) ** -1.64,
            domain=(-2, 7),
            exps=(-1.64, 0),
            splitting=True,
        )
    if case == 5:
        return cj.chebfun(
            lambda x: 0.75 + jnp.sin(10 * x) / jnp.exp(x), domain=(0, jnp.inf), splitting=True
        )
    end = -3 * jnp.pi
    return cj.chebfun(
        lambda x: x * (5 + jnp.exp(x**3)) / (end - x), domain=(-jnp.inf, end), exps=(0, -1)
    )
