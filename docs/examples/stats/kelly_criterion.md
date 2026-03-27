# Kelly Criterion

**Original:** [stats/KellyCriterion](https://github.com/chebfun/examples/blob/master/stats/KellyCriterion.m)
**Author(s):** Mark Richardson, October 2012

---

This example explores optimal bet-sizing using the Kelly Criterion, a
fundamental result in investment theory and information theory first described
by John Kelly at Bell Labs in 1956 [1].

## Theoretical setup

Suppose you have a fixed amount of money and can enter a wager an unlimited
number of times on identical terms. Each time, you bet a fraction $f$ of your
current balance. With probability $p$, your stake is multiplied by $a$ and
returned; with probability $1-p$, your stake is lost.

After $n$ wagers with $w$ wins and $l = n - w$ losses, your capital is

$$C_n = (1 + af)^w (1 - f)^l\, C_0.$$

The **logarithmic growth rate** of capital is

$$G(f) = \lim_{n \to \infty} \frac{1}{n}\log\frac{C_n}{C_0}
= p\log(1 + af) + (1-p)\log(1 - f).$$

Maximizing $G$ over $f \in [0, 1]$ by calculus gives the **Kelly fraction**:

$$f^* = \frac{ap - (1-p)}{a}.$$

For $a = 2$ (payoff odds 2:1) and $p = 1/2$, the optimal bet fraction is
$f^* = 1/4$. The expected capital multiplier per bet is $e^{G(f^*)}$, which
exceeds 1 for any $f$ in the interval where $G(f) > 0$.

## Numerical approach for complex wagers

Real-world situations often involve multiple outcomes with different payoffs and
probabilities. For a wager with six outcomes having probabilities $p_j$ and
payoffs $a_j$, the growth rate generalizes to

$$G(f) = \sum_{j=1}^6 p_j \log(1 + a_j f).$$

No closed-form solution exists, but the global maximum can be found numerically.
The critical fraction beyond which we expect to lose wealth is determined by the
roots of $G(f)$.

## Practical considerations

In practice, probabilities are estimated from data, so it is common to bet less
than the full Kelly fraction -- "half-Kelly" is typical. This conservative
approach also reduces volatility (the annualized standard deviation of
logarithmic returns).

## References

1. J. L. Kelly, Jr., A new interpretation of information rate, *Bell Systems
   Technical Journal* 35 (1956), 917--926.

```python
from examples.stats.kelly_criterion import run
run()
```

![Kelly Criterion](../../images/stats/kelly_criterion.png)