# A Bayesian Gradebook

**Original:** [stats/BayesianGradebook](https://www.chebfun.org/examples/stats/BayesianGradebook.html)
**Author(s):** Toby Driscoll, November 2013

---

In US educational systems and many others, students receive numerous assessments
(homework, projects, exams) that are marked with individual scores, typically
averaged -- often with weights -- to arrive at a course score. This is not a
very scientific process. One way of injecting more clarity is to appeal to
Bayesian statistics.

## Bayes' theorem

Suppose a student's "true ability" is represented as a number $\theta$ in the
interval $[0,1]$. We start with a prior belief about how $\theta$ is
distributed, represented as a probability density $P(\theta)$. Now let $x$ be an
observation -- a score on an assessment. Bayes' theorem tells us how to update
our beliefs about $\theta$ in light of the new evidence. The posterior
distribution is

$$P(\theta \mid x) = \frac{P(x \mid \theta)\, P(\theta)}{P(x)}.$$

The quantity $P(x \mid \theta)$ is the _likelihood_: for each possible $\theta$,
what is the probability that we would observe $x$? Where the likelihood is
small, the posterior is greatly reduced. The denominator $P(x)$ is simply a
normalization constant ensuring the posterior integrates to one.

## Prior distribution

We choose a prior to initialize our belief about ability $\theta$. Good practice
would use past empirical data, but here we idealize it as a truncated normal
distribution centered at 0.7 with standard deviation 0.3. We also define
expected-value and variance functionals:

$$E[f] = \int_0^1 f(\theta)\,P(\theta)\,d\theta, \qquad
\mathrm{Var}[f] = E\!\bigl[(f - E[f])^2\bigr].$$

## Likelihood function

Given a true ability value $\theta$, we model the probability of observing a
score $x$ with a truncated normal:

$$P(x \mid \theta) = \frac{\phi_{\theta,\sigma}(x)}{q(\theta,\sigma)},$$

where $\phi_{\theta,\sigma}$ is the Gaussian function with mean $\theta$ and
variance $\sigma^2$, and $q$ provides normalization. Because
$\phi_{\theta,\sigma}(x) = \phi_{x,\sigma}(\theta)$ for the Gaussian, the
likelihood function is easy to assemble. The parameter $\sigma$ is treated as
fixed, though one could build a Bayesian estimator for it as well.

## Update process

Each new assessment score triggers a Bayesian update: the posterior becomes the
new prior, incorporating accumulated evidence. Two natural point estimates
summarize the posterior -- the **mode** (maximum of the distribution) and the
**mean** (expected value of $\theta$). Both are compared to the traditional
running average of scores. As scores accumulate, the posterior narrows, and the
standard deviation of $\theta$ quantifies our increasing confidence.

## Poor student

For a student with scores [0.55, 0.67, 0.62, 0.66], the Bayesian and
traditional methods do not differ meaningfully. However, the Bayesian approach
provides extra information: a confidence measure via the standard deviation of
$\theta$.

## Good student

Shifting all scores up by 30 percentage points reveals the Bayesian method's
advantage for high performers. The running average is shortchanged near the upper
boundary because it is impossible to score above 1. The likelihood's
normalization factor $1/q$ amplifies belief in higher values when observed scores
are near the boundary, compensating for the impossibility of observing scores
greater than 1.

## The comeback kid

When a student's first score is anomalously low but subsequent scores are strong,
the Bayesian method provides a modest, principled adjustment. The first score by
itself is not unlikely from someone of considerably higher ability. The standard
deviation shows comparable confidence to the reliable-good-student case.

## Variable assessment reliability

Increasing $\sigma$ to 0.15 models low-stakes assessments (e.g., homework) where
individual scores carry less information. The Bayesian update then overlooks low
scores more than a running average, which can be considered more accurate than an
ad hoc "drop lowest score" policy.

Perhaps the most illustrative numbers are the Bayesian standard deviations: even
with decent assessments and consistent scores, a respectable confidence interval
usually covers more than one letter grade category.

```python
from examples.stats.bayesian_gradebook import run
run()
```

## Output

![Bayesian Gradebook](../../images/stats/bayesian_gradebook.png)
