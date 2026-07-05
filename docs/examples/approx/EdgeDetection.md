# Edge Detection in Chebfun

*Nick Trefethen, November 2016*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/approx/EdgeDetection.html)

## Automatic breakpoint detection

Chebfun's `splitting on` mode uses a recursive bisection algorithm (originally by
Rodrigo Platte) to automatically detect where a function has a jump discontinuity
or is merely non-smooth.

```python
import chebfunjax as cj
import jax.numpy as jnp

# With explicit breakpoints at 2 and 5
f = cj.chebfun(lambda x: jnp.sin(x * jnp.where(x < 2.0, 1.0,
               jnp.where(x < 5.0, 2.0, 3.0))),
               domain=[0.0, 2.0, 5.0, 8.0])
print(f"Pieces: {len(f.funs)}, lengths: {[len(p.tech.coeffs) for p in f.funs]}")
```

The accuracy of breakpoint location is related to the smoothness class:
$O(\epsilon^{1/k})$ for $C^k$ functions.

![Edge Detection in Chebfun](../../images/approx/EdgeDetection.png)

## Figures (chebfun.org parity)

![EdgeDetection figure 1](../../images/approx/EdgeDetection_01.png)

![EdgeDetection figure 2](../../images/approx/EdgeDetection_02.png)

![EdgeDetection figure 3](../../images/approx/EdgeDetection_03.png)

![EdgeDetection figure 4](../../images/approx/EdgeDetection_04.png)

![EdgeDetection figure 5](../../images/approx/EdgeDetection_05.png)
