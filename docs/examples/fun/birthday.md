# Birthday Cards and Analytic Functions

**Original:** [fun/Birthday](https://www.chebfun.org/examples/fun/Birthday.html)
**Author:** Nick Trefethen, September 2010

---

Chebfun's `scribble` command was introduced for entertainment, but it turns
out to be surprisingly useful for illustrating complex variables. Suppose
for example it is Chebyshev's birthday and you want to send him a card.

## Piecewise-linear complex path

The `scribble` command converts a text string into a piecewise-linear
complex-valued chebfun defined on $[-1,1]$. Each letter is encoded as a
sequence of breakpoints $z_k$ in the complex plane, and the function
interpolates linearly between them:

$$s(t) = \sum_k \mathbf{1}_{[t_k, t_{k+1}]}(t) \left[(1-\tau) z_k + \tau z_{k+1}\right]$$

where $\tau = (t - t_k)/(t_{k+1} - t_k)$.

The resulting chebfun `s` has many smooth pieces with complex endpoint values.
For example, `scribble('Happy Birthday Pafnuty!')` produces roughly 89
piecewise-linear segments.

## Applying analytic functions

Since `s` is a chebfun, we can apply functions to it and produce transformed
greeting cards. For example:

- $\exp(s)$ distorts the text through the exponential map.
- $\exp(3is)$ wraps the letters around the unit circle.
- $\exp\bigl((1+i)s\bigr)$ combines rotation with scaling.
- $\sinh(3s)$ stretches and folds the letters dramatically.

Playing around with different functions is a good way to learn about complex
variables, and a good way to make greeting cards.


![Birthday](../../images/fun/birthday.png)

The left panel shows a heart-shaped card with "PAFNUTY" (Chebyshev's first name)
written in an arc. The right panel shows "CHEBY" encoded as piecewise-linear
strokes in the complex plane.

## Code

```python
from examples.fun.birthday import run
run()
```

