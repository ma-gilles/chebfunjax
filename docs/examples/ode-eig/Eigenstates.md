# Eigenstates of the Schroedinger equation

*Nick Trefethen, January 2012*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/ode-eig/Eigenstates.html)

(Chebfun example ode-eig/Eigenstates.m)

`quantumstates` computes and plots eigenstates of the time-independent
Schroedinger operator

$$ L u = -h^2 u'' + V(x)\,u, $$

with each eigenfunction drawn at the height of its energy level. The
default is ten states with $h = 0.1$.

## The harmonic oscillator

```python
x = chebfun(lambda x: x, domain=(-3, 3))
V = x**2
lam, funs = quantumstates(V)
```

![Eigenstates figure 1](../../images/ode-eig/Eigenstates_repl_01.png)

The eigenvalues are $h\,(2k+1)$ exactly, and come out that way:

```text
ans =
   0.099999999999974
   0.300000000000026
   0.500000000000047
   0.700000000000080
   0.900000000000085
   1.100000000000099
   1.300000000000129
   1.500000000000139
   1.700000000000173
   1.900000000000208
```

against MATLAB's `0.099999999999985, 0.300000000000002, ...` —
thirteen digits on both sides of the comparison.

More states, a smaller $h$, or both:

![Eigenstates figure 2](../../images/ode-eig/Eigenstates_repl_02.png)
![Eigenstates figure 3](../../images/ode-eig/Eigenstates_repl_03.png)
![Eigenstates figure 4](../../images/ode-eig/Eigenstates_repl_04.png)

## Square wells and other potentials

A deep square well confines the low states almost completely:

![Eigenstates figure 5](../../images/ode-eig/Eigenstates_repl_05.png)

a shallow one lets the higher states spill over the top:

![Eigenstates figure 6](../../images/ode-eig/Eigenstates_repl_06.png)

and non-smooth potentials pose no difficulty:

![Eigenstates figure 7](../../images/ode-eig/Eigenstates_repl_07.png)
![Eigenstates figure 8](../../images/ode-eig/Eigenstates_repl_08.png)

An off-centre barrier splits states into near-degenerate pairs:

![Eigenstates figure 9](../../images/ode-eig/Eigenstates_repl_09.png)

> **Implementation note.** `quantumstates` previously hand-rolled a
> fixed 100-point collocation and reached only five digits on the
> harmonic oscillator. MATLAB's version simply builds a chebop and
> calls `eigs` — and our `Chebop.eigs` already reaches $10^{-14}$ on
> this problem at $n = 96$. `quantumstates` now delegates the same way,
> doubling the grid until the requested eigenvalues stabilise, which is
> where the thirteen digits above come from.

---

*Replica script: [`examples/ode-eig/eigenstates_replica.py`](https://github.com/ma-gilles/chebfunjax/blob/main/examples/ode-eig/eigenstates_replica.py).
Original example copyright by The University of Oxford and The Chebfun
Developers.*
