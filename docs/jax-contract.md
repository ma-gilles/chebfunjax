# JAX Semantic Contract

What is guaranteed to work under JAX transforms, and what is intentionally not.

## The Core Rule

**Evaluation is differentiable. Construction is not.**

- `f(x)` — evaluating a Chebfun at points — is JIT, vmap, and grad safe.
- `cj.chebfun(func)` — constructing a Chebfun adaptively — is NOT JIT-safe
  (uses Python control flow with data-dependent termination).

This split is fundamental and intentional.

## JIT-safe (traced, compiled, fast)

| Operation | Why it works |
|-----------|-------------|
| `f(x)` — evaluate at points | Clenshaw recurrence, fixed shape |
| `f.diff(k)` — differentiate | Coefficient recurrence, fixed shape |
| `f.cumsum()` — antiderivative | Coefficient recurrence, fixed shape |
| `f.sum()` — definite integral | Dot product of coeffs with known weights |
| `f.norm()` — L2 norm | Inner product |
| `f.inner(g)` — inner product | Dot product of coefficients |
| `f + g`, `f * g`, arithmetic | Coefficient operations, fixed output size |
| `f.coeffs`, `f.values` — access data | Direct array access |
| `chebpts(n)`, `legpts(n)` — quadrature | Pure array computation (n must be static) |
| `cheb2leg(c)`, `leg2cheb(c)` — transforms | Matrix-vector or FFT, fixed shape |
| `bary(x, values, points)` — interpolation | Barycentric formula |

These should all be tested with `jax.jit(f)(args)` in their test suites.

## vmap-safe (batched evaluation)

| Operation | Why it works |
|-----------|-------------|
| `jax.vmap(f)(xs)` — batch evaluate | Vectorized Clenshaw |
| `jax.vmap(chebpts, in_axes=(0,))(ns)` | Only if `n` is static per-call (use partial) |

**Rule:** Any function that is JIT-safe is also vmap-safe if all shapes are static.

## grad-safe (differentiable)

| Operation | Gradient of what w.r.t. what |
|-----------|------------------------------|
| `jax.grad(lambda x: f(x))(x0)` | df/dx at a point |
| `jax.grad(lambda c: Chebtech2.from_coeffs(c).sum())(coeffs)` | d(integral)/d(coefficients) |
| `jax.grad(lambda c: Chebtech2.from_coeffs(c).norm())(coeffs)` | d(norm)/d(coefficients) |
| `jax.grad(lambda c: Chebtech2.from_coeffs(c)(x0))(coeffs)` | d(f(x0))/d(coefficients) |

**What this enables:** optimization over Chebyshev coefficient space,
sensitivity analysis, physics-informed loss functions.

## Intentionally NOT JIT/grad-safe

| Operation | Why not | What to do instead |
|-----------|---------|--------------------|
| `cj.chebfun(func)` — adaptive construction | Python while loop, dynamic output size | Call outside JIT; pass the resulting Chebfun into JIT |
| `f.roots()` — rootfinding | Eigenvalue of colleague matrix; variable number of roots | Call outside JIT; use roots as static data |
| `f.max()`, `f.min()` — extrema | Depends on roots of derivative | Call outside JIT |
| `abs(f)` — absolute value of a chebfun | May introduce breakpoints (piecewise) | Construction-level op, not pointwise |
| `sign(f)`, `heaviside(f)` | Introduces discontinuities | Construction-level op |
| Piecewise breakpoint detection | Adaptive, variable number of pieces | Construction-level op |
| `Chebfun.simplify()` — degree reduction | Adaptive chopping, variable output size | Call outside JIT |

### The pattern for mixing JIT and non-JIT

```python
# Construction outside JIT
f = cj.chebfun(jnp.sin)
g = cj.chebfun(lambda x: jnp.exp(-x**2))

# Fast operations inside JIT
@jax.jit
def compute(f, g, x):
    return f(x) + g.diff()(x) - g.sum()

result = compute(f, g, jnp.linspace(-1, 1, 1000))
```

## CPU/GPU Parity

All JIT-safe operations must produce identical results (at `rtol=1e-12`)
on CPU and GPU. This is tested in CI when GPU runners are available.

Known exceptions:
- Eigenvalue-based rootfinding may have different ordering on GPU
  (roots are sorted, so final output matches, but intermediate order may differ)
- FFT precision may differ by 1-2 ULP between CPU and GPU implementations

## Agent Testing Requirements

When implementing a function, mark it in the test class docstring:

```python
class TestChebpts:
    """Tests for chebpts.

    JAX contract: jit=yes (n must be static), vmap=no, grad=no
    """
```

And include the corresponding tests:

```python
def test_jit(self):
    """JIT with static n."""
    import functools
    jitted = jax.jit(functools.partial(chebpts, 10))
    npt.assert_allclose(np.array(jitted()), np.array(chebpts(10)), rtol=1e-15)
```
