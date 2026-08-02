# The phaseplot command

*Nick Trefethen, March 2020*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/complex/PhaseplotCommand.html)

(Chebfun example complex/PhaseplotCommand.m)

The `phaseplot` command draws a phase portrait directly from a
function handle, with no chebfun2 construction — handy for functions
with poles or branch cuts.  The identity map shows the color wheel:

![PhaseplotCommand figure 1](../../images/complex/PhaseplotCommand_repl_01.png)

A Mobius transformation $(z-1)/(z+1)$, with a zero at $1$ and pole at
$-1$:

![PhaseplotCommand figure 2](../../images/complex/PhaseplotCommand_repl_02.png)

$z^3$ cycles the colors three times:

![PhaseplotCommand figure 3](../../images/complex/PhaseplotCommand_repl_03.png)

$\sqrt{z-1}\sqrt{z+1}$ has branch cuts emanating from $\pm 1$:

![PhaseplotCommand figure 4](../../images/complex/PhaseplotCommand_repl_04.png)

And $e^{3/z}$ has an essential singularity at the origin:

![PhaseplotCommand figure 5](../../images/complex/PhaseplotCommand_repl_05.png)

Two color conventions are available:

![PhaseplotCommand figure 6](../../images/complex/PhaseplotCommand_repl_06.png)

---

*Replicated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); original
example copyright The University of Oxford and The Chebfun Developers.*
