# Local Complexity of a Function

*Nick Trefethen, June 2011*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/approx/Local.html)

## Local vs. global complexity

A globally smooth function may have much more complexity in some regions than
others. The piecewise Chebfun representation adapts to this by using more
polynomial terms where the function oscillates faster.

```python
import chebfunjax as cj
import jax.numpy as jnp

breakpoints = [-1.0, -0.5, 0.0, 0.5, 1.0]

# Increasing frequency from left to right
f = cj.chebfun(lambda x: jnp.sin(x * (20.0 - 15.0*x)),
               domain=breakpoints)

# Each piece has a different length
for k, piece in enumerate(f.funs):
    print(f"Piece {k}: [{breakpoints[k]:.1f}, {breakpoints[k+1]:.1f}], length = {len(piece.tech.coeffs)}")
```

The right-hand pieces have higher local frequency and thus more Chebyshev
coefficients — the chebfun adapts locally.

![Local Complexity of a Function](../../images/approx/Local.png)

## Figures (chebfun.org parity)

![Local figure 1](../../images/approx/Local_01.png)

![Local figure 2](../../images/approx/Local_02.png)

![Local figure 3](../../images/approx/Local_03.png)
