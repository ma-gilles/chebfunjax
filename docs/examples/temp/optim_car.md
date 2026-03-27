# Optimal Performance of a Car

**Original:** [ode/OptimCar](https://github.com/chebfun/examples/blob/master/temp/OptimCar.m)
**Author(s):** Asgeir Birkisson, November 2010

---

This example solves a simple optimal control problem: maximise the distance
travelled by a car in a fixed time, subject to a bounded acceleration.

## Problem formulation

The state variables are the position $x(t)$ and velocity $v(t)$ of a car on
$t \in [0, 2]$, starting from rest at the origin:

$$\dot{x} = v, \qquad \dot{v} = u(t),$$

with initial conditions $x(0) = 0$, $v(0) = 0$, where $u(t)$ is the
acceleration control and $|u(t)| \le 1$.

The goal is to choose $u(t)$ to maximise $x(2)$, the distance at the final
time. By the **Pontryagin minimum principle**, the optimal control is
**bang-bang**:

$$u^*(t) = \operatorname{sign}(1 - t),$$

i.e., full acceleration for $t < 1$ followed by full braking for $t > 1$.
This brings the car to a stop exactly at $t = 2$.

## Adjoint variables

The co-state (adjoint) variables $\lambda_x(t)$ and $\lambda_v(t)$ satisfy

$$\dot{\lambda}_x = 0, \qquad \dot{\lambda}_v = -\lambda_x,$$

with terminal conditions $\lambda_x(2) = 0$, $\lambda_v(2) = 0$.

The full system of four ODEs (state + co-state) is solved simultaneously as
a `chebop` boundary value problem, and the solution confirms the bang-bang
structure.

## Code

```python
from examples.temp.optim_car import run
run()
```

## Output

![Optimal Performance of a Car](../../images/temp/optim_car.png)
