"""Source text display for finite scalar Newton iteration.

Chebfun 7574c77: @chebop/displayInfoInit.m, displayInfoIter.m,
displayInfoFinal.m, displayInfoExactInitial.m, and solvebvpNonlinear.m.
GUI plotting and pause callbacks are outside this text adapter.
"""

import jax.numpy as jnp

_DASHES = "-" * 62


def _number(value, specification):
    # Numeric MATLAB fprintf conversions display the real component.
    return format(complex(value).real, specification).replace("nan", "NaN").replace("inf", "Inf")


def format_display_info(mode, display, *, step=None, iterations=0,
                        backend="chebcolloc2", error=0, boundary_error=0):
    """Format literal source event fields; never infer iteration metadata."""
    if not isinstance(display, str):
        return ""
    if mode == "exactInitial" and display.lower() == "iter":
        return (f"\n{_DASHES}\nInitial guess appears to be a solution of the BVP. \n"
                f"Initial guess returned as the solution.\n{_DASHES}\n")
    if mode == "init" and display.lower() == "iter":
        return ("Iter.   || du ||   Contraction    stepsize   len(du)   len(u)"
                f"\n{_DASHES}\n")
    if mode == "iter" and display.lower() == "iter":
        head = (f" {step['iteration']:02d} {_number(step['normDelta'], '12.2e')} "
                f"{_number(step['contraction'], '12.2e')} ")
        if step["lambda"] == 1:
            return (head + _number(step["lambda"], "11.4f")
                    + f" {step['lenDelta']:7d} {step['lenU']:9d} \n")
        return (head + _number(step["lambda"], "12.2e")
                + f" {step['lenDelta']:6d} {step['lenU']:9d} \n")
    # Native final helper uses strcmp, while init/iter use strcmpi.
    if mode == "final" and display in ("iter", "final"):
        count = "1 iteration" if iterations == 1 else f"{iterations} iterations"
        basis = "Coefficients" if backend == "ultraS" else "Values"
        return (f"{_DASHES}\nNewton's method converged in {count}.\n"
                f"Discretization basis used: {basis}. \n"
                f"Final error estimate: {_number(error, '.2e')} (differential equation) \n"
                f"{_number(boundary_error, '30.2e')} (boundary conditions). \n\n")
    return ""


def boundary_error(values, groups):
    """Source normBCres: endpoint squares and general-condition 2-norm.

    Values are from the solver's existing final boundary evaluation; callbacks
    are not evaluated again. Endpoint squares deliberately have no conjugation.
    General conditions are vectors in the native scalar source contract.
    """
    total = jnp.asarray(0, dtype=jnp.complex128)
    for kind, start, stop in groups:
        parts = [jnp.asarray(v).reshape(-1) for v in values[start:stop]]
        entries = jnp.concatenate(parts) if parts else jnp.empty(0)
        if kind == "general":
            total = total + jnp.linalg.norm(entries, ord=2) ** 2
        else:
            total = total + jnp.sum(entries ** 2)
    # Native sqrt promotes a negative real sum to its principal complex root.
    return jnp.sqrt(total)
