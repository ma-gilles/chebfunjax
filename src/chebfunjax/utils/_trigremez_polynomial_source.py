"""Polynomial trig Remez trial functions and source full exchange.

Provenance
----------
MATLAB source: @chebfun/trigremez.m, lines 80–200,265–278,351–440.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
The Python output prefix is (p, err, status); status is a dictionary.
"""

import warnings

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.utils.trigutils import trigBary, trigBaryWeights


def _exchange(f, p, xk, h, method, count):
    derivative = (f - p).diff()
    # Native roots(e_num, "nobreaks") retains true jump roots.
    # Python scalar roots() inserts piece and jump roots, without inserting
    # roots solely from stored breakpoint values, so it adapts that flag.
    roots = derivative.roots()
    rr = jnp.concatenate((jnp.asarray([f.domain.a]), roots))
    errors = f(rr) - p(rr)
    if method == 1:
        selected = jnp.asarray([jnp.argmax(jnp.abs(errors))])
    else:
        selected = jnp.where(jnp.abs(errors) >= jnp.abs(h))[0]
    sigma = jnp.where(jnp.arange(count) % 2, -1.0, 1.0)
    locations = jnp.concatenate((rr[selected], xk))
    values = jnp.concatenate((errors[selected], sigma * h))
    order = jnp.argsort(locations, stable=True)
    locations, values = locations[order], values[order]
    # Native repeated = diff(r)==0 deletes the earlier member of each pair.
    keep = jnp.concatenate((jnp.diff(locations) != 0, jnp.asarray([True])))
    locations, values = locations[keep], values[keep]
    kept_locations, kept_values = [locations[0]], [values[0]]
    for i in range(1, locations.size):
        if bool(jnp.sign(values[i]) == jnp.sign(kept_values[-1])):
            if bool(jnp.abs(values[i]) > jnp.abs(kept_values[-1])):
                kept_locations[-1], kept_values[-1] = locations[i], values[i]
        else:
            kept_locations.append(locations[i])
            kept_values.append(values[i])
    locations, values = jnp.stack(kept_locations), jnp.stack(kept_values)
    index = int(jnp.argmax(jnp.abs(values)))
    error = jnp.abs(values[index])
    first = max(index - count + 1, 0)
    return (locations[first : first + count] if count <= locations.size else locations), error


def polynomial_trigremez(f, degree, max_iter, tol, normf, iteration_callback=None):
    """Run native polynomial loop with public representations and root finding."""
    degree = int(degree)
    count = 2 * degree + 2
    original_domain = (float(f.domain.a), float(f.domain.b))
    domain = (-float(jnp.pi), float(jnp.pi))
    target = f.new_domain(domain)
    xk = -jnp.pi + 2 * jnp.pi * jnp.arange(count) / count
    old = xk
    sigma = jnp.where(jnp.arange(count) % 2, -1.0, 1.0)
    delta, best_delta, diffx, iteration = normf, jnp.inf, 1.0, 0
    best = None
    while float(delta / normf) > tol and iteration < max_iter and float(diffx) > 0:
        fk = target(xk)
        weights = trigBaryWeights(xk)
        h = (weights @ fk) / (weights @ sigma)
        pk = fk - h * sigma
        # Native computes the trial function before perturbing exactly-zero h.
        p = chebfun(
            lambda x: trigBary(x, pk, xk, domain), domain=domain, n=2 * degree + 1, trig=True
        )
        if bool(h == 0):
            h = jnp.asarray(1e-19)
        xk, error = _exchange(target, p, xk, h, 2, count)
        if float(error / normf) > 1e5:
            xk, error = _exchange(target, p, old, h, 1, count)
        diffx = jnp.max(jnp.abs(old - xk))
        delta = error - jnp.abs(h)
        if bool(delta < best_delta):
            best = p, error, xk
            best_delta = delta
        if iteration_callback is not None:
            iteration_callback(
                iteration, error, h, delta, normf, diffx, old, xk, target, p, original_domain
            )
        old = xk
        iteration += 1
    if best is None:
        # max_iter=0 is an existing Python extension; source default always runs.
        fk = target(xk)
        weights = trigBaryWeights(xk)
        h = (weights @ fk) / (weights @ sigma)
        p = chebfun(
            lambda x: trigBary(x, fk - h * sigma, xk, domain),
            domain=domain,
            n=2 * degree + 1,
            trig=True,
        )
        best = p, jnp.max(jnp.abs(fk)), xk
        best_delta = delta
    if float(best_delta / normf) > tol:
        warnings.warn(
            "CHEBFUN:CHEBFUN:trigremez:convergence: Remez algorithm did not "
            f"converge after {iteration} iterations to the tolerance {tol:.2g}.",
            UserWarning,
            stacklevel=2,
        )
    p, error, references = best
    a, b = original_domain
    physical = b * (references + jnp.pi) / (2 * jnp.pi) + a * (jnp.pi - references) / (2 * jnp.pi)
    return (
        p.new_domain(original_domain),
        error,
        {"delta": best_delta / normf, "iter": iteration, "diffx": diffx, "xk": physical},
    )
