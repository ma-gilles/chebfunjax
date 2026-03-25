# Wikipedia ODE examples

*Mark Richardson, September 2010*

[Chebfun example](https://www.chebfun.org/examples/ode-linear/WikiODE.html)

## Overview

Solves a selection of ODEs taken from the Wikipedia page on ordinary differential
equations, covering a range of linear first- and second-order problems including
constant-coefficient equations, variable-coefficient examples, and systems.

## Method

Each ODE is solved using the Chebop BVP framework, and results are compared against
exact solutions where available.

```python
from chebfunjax.operators.chebop import Chebop

# Example: u' + u = e^x * cos(x), u(0) = 0
dom = (0.0, 5.0)
N = Chebop(lambda x, u: u.diff() + u, domain=dom)
N.lbc = 0.0
u = N.solve(lambda x: jnp.exp(x) * jnp.cos(x))
```

## Results

All solved examples match exact solutions to near machine precision,
demonstrating the spectral accuracy of the Chebyshev collocation method.

![Wikipedia ODEs](../../images/ode-linear/wiki_odes.png)
