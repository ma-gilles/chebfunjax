# The Mean Value Theorem

*Kuan Xu, October 2012*

*Original: [chebfun.org/examples/calc/MeanValueTheorem](https://www.chebfun.org/examples/calc/MeanValueTheorem.html)*

---

The Mean Value Theorem is one of the central results of differential
calculus. It states: if $f$ is continuous on $[a,b]$ and differentiable
on $(a,b)$, then there exists $c \in (a,b)$ such that

$$f'(c) = \frac{f(b) - f(a)}{b - a}.$$

Geometrically, there exists a point where the tangent to the curve is
parallel to the secant joining the endpoints.

## Finding the MVT point with chebfunjax

Consider $f(x) = (x-1)(x-2)(x-3)$ on $[-6, 6]$.

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

a, b = -6.0, 6.0
f = cj.chebfun(lambda x: (x - 1) * (x - 2) * (x - 3), domain=(a, b))

# Secant slope
sl = (float(f(jnp.array(b))) - float(f(jnp.array(a)))) / (b - a)
print(f"Secant slope = {sl:.6f}")

# Find points where f'(x) = secant slope
fprime = f.diff()
c_vals = (fprime - sl).roots()
print(f"MVT points c: {np.sort(np.array(c_vals))}")
```

```
Secant slope = -11.000000
MVT points c: [-2.309401  2.309401]
```

Two roots are found (both inside $(a,b)$) because this cubic has a
double-well structure.

![MVT visualization: tangent parallel to secant](../../../images/calc/mean_value_theorem.png)

## Verifying the result

We can verify that $f'(c)$ equals the secant slope at each MVT point:

```python
for c in np.sort(np.array(c_vals)):
    if a < c < b:
        fprime_c = float(fprime(jnp.array(c)))
        print(f"  f'({c:.6f}) = {fprime_c:.8f}  (secant slope = {sl:.8f})")
```

```
  f'(-2.309401) = -11.000000  (secant slope = -11.000000)
  f'( 2.309401) = -11.000000  (secant slope = -11.000000)
```

The chebfunjax `roots` function finds these analytically (via eigenvalue
methods), giving machine-precision results.

## References

1. W. Rudin, *Principles of Mathematical Analysis*, 3rd ed., McGraw-Hill, 1976.
