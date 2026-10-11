"""Observed native CPU reduction for a real row times one real column.

Chebfun 7574c77 @chebtech/extrapolate.m prescribes a matrix product. This
adapter records the operation order independently observed in MATLAB R2025b
Update2 CPU captures, including cancellation basis, length transition, tails,
and nonpower products. The source does not prescribe a universal BLAS tree.
The caller restricts its use to at most 128 good rows, one missing second-kind
endpoint and signed unit/half weights. Other extrapolation routes retain compensated reductions.
"""

import jax
import jax.numpy as jnp


@jax.jit
def _cpu_endpoint_dot(weights, values):
    """Evaluate the qualified single-column product with rounded operations."""
    products = jax.lax.optimization_barrier(weights * values)
    n = products.shape[0]
    if n <= 50:
        # An odd-length product starts with the first entry. Remaining
        # adjacent products form a subtotal before addition to the carry.
        offset = n % 2
        initial = products[0] if offset else jnp.zeros((), products.dtype)

        def pair_step(i, total):
            subtotal = jax.lax.optimization_barrier(
                products[offset + 2 * i] + products[offset + 2 * i + 1]
            )
            return jax.lax.optimization_barrier(total + subtotal)

        return jax.lax.fori_loop(0, n // 2, pair_step, initial)

    # Above 50 entries, the observed provider carries eight lanes; its
    # residual entries feed the first two lanes. Even/odd lane groups are
    # reduced separately in ascending order before their final addition.
    def block_step(i, carry):
        block = jax.lax.dynamic_slice(products, (8 * i,), (8,))
        return jax.lax.optimization_barrier(carry + block)

    carry = jax.lax.fori_loop(
        0, n // 8, block_step, jnp.zeros((8,), products.dtype)
    )
    for i in range(n % 8):
        lane = i % 2
        updated = jax.lax.optimization_barrier(carry[lane] + products[8 * (n // 8) + i])
        carry = carry.at[lane].set(updated)
    even = carry[0]
    odd = carry[1]
    for i in (2, 4, 6):
        even = jax.lax.optimization_barrier(even + carry[i])
        odd = jax.lax.optimization_barrier(odd + carry[i + 1])
    return jax.lax.optimization_barrier(even + odd)
