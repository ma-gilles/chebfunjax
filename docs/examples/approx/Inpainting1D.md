# Inpainting in one dimension

*Yuji Nakatsukasa and Nick Trefethen, November 2019*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/approx/Inpainting1D.html)

(Chebfun example approx/Inpainting1D.m)

Suppose a smooth function is corrupted on part of its domain — here by
taking the pointwise maximum with a smooth random function — and we try
to recover it by polynomial fitting.  The $L^1$ fit (`polyfitL1`)
recovers the smooth function to high precision, while the $L^2$ and
$L^\infty$ fits are pulled far off by the corruption — the
sparsity-promoting property of $L^1$ fitting.

```python
import jax
import jax.numpy as jnp
import chebfunjax as cj
from chebfunjax.utils.randnfun import randnfun

x = cj.chebfun(lambda t: t)
smooth = 0.3 + x**2 + (0.3*x).exp()
noise = randnfun(0.1, key=jax.random.PRNGKey(1))   # MATLAB randn streams
corrupted = smooth.maximum(noise)                  # are not reproducible
```

![Inpainting1D figure 1](../../images/approx/Inpainting1D_repl_01.png)

The $L^1$ fit recovers the underlying smooth function almost exactly:

```python
p1 = corrupted.polyfitL1(len(smooth) - 3)
```

![Inpainting1D figure 2](../../images/approx/Inpainting1D_repl_02.png)

```
err1 =
     8.876516212234929e-10
```

(The published MATLAB value is `9.8e-13` on its own noise realization;
the recovery-to-negligible-error phenomenon is fully reproduced.)

The $L^2$ fit, by contrast, is thrown off by the corrupted region:

![Inpainting1D figure 3](../../images/approx/Inpainting1D_repl_03.png)

```
err2 =
   0.164201627741589
```

![Inpainting1D figure 4](../../images/approx/Inpainting1D_repl_04.png)

And the $L^\infty$ (minimax) fit splits the corruption error evenly,
which is exactly what one does not want here:

![Inpainting1D figure 5](../../images/approx/Inpainting1D_repl_05.png)

```
errinf =
   0.629201401450443
```

(The corruption realization differs from MATLAB's, so the $L^2$ and
$L^\infty$ error magnitudes differ from the published 0.041/0.276; the
qualitative contrast — near-exact $L^1$ recovery versus order-0.1
failures — is the same.)

## References

1. Y. Nakatsukasa and A. Townsend, Error localization of best $L_1$
   polynomial approximants, arXiv:1902.02664.

---

*Replicated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); original
example copyright The University of Oxford and The Chebfun Developers.*
