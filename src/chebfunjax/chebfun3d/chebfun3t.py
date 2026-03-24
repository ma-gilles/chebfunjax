# uses-numpy: Tucker tensor operations use numpy for dense array manipulation (not JIT-safe)
"""Chebfun3T — Tucker tensor wrapper for Chebfun3 decompositions.

A thin data-class that holds the raw Tucker decomposition data produced by
:class:`~chebfunjax.chebfun3d.chebfun3.Chebfun3`, together with helpers for
inspecting and manipulating the decomposition directly.

The Tucker format represents f(x, y, z) as:

    f(x, y, z) ≈ Σ_ijk  core[i, j, k] * X_i(x) * Y_j(y) * Z_k(z)

where ``X_i``, ``Y_j``, ``Z_k`` are Chebyshev basis functions and ``core``
is the Tucker core tensor.

This class mirrors the ``chebfun3t`` concept in MATLAB Chebfun, which wraps
Chebfun3 Tucker data in a convenient container.

Translated from MATLAB Chebfun @chebfun3t (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

Provenance
----------
MATLAB source : @chebfun3t/chebfun3t.m, @chebfun3/core.m,
    @chebfun3/cols.m, @chebfun3/rows.m, @chebfun3/tubes.m
Chebfun commit: 7574c77
Original authors: Copyright 2023 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.tech.chebtech import Chebtech2

__all__ = ["Chebfun3T", "chebfun3t"]


class Chebfun3T(eqx.Module):
    """Tucker tensor wrapper for a :class:`~chebfunjax.chebfun3d.chebfun3.Chebfun3`.

    Stores the Tucker core and the three sets of factor functions (columns,
    rows, tubes), together with the physical domain.  Provides accessors for
    inspection and a round-trip back to :class:`Chebfun3`.

    Attributes
    ----------
    core : jax.Array, shape (rx, ry, rz)
        Tucker core tensor.
    cols : list of Chebtech2
        Column factor functions X_i(x), Chebyshev expansions on [-1, 1].
    rows : list of Chebtech2
        Row factor functions Y_j(y), Chebyshev expansions on [-1, 1].
    tubes : list of Chebtech2
        Tube factor functions Z_k(z), Chebyshev expansions on [-1, 1].
    domain : tuple (xa, xb, ya, yb, za, zb)
        Physical domain.  Static field.

    Notes
    -----
    Evaluation is performed by converting to a :class:`Chebfun3` first,
    or by using the :meth:`__call__` method directly (which mirrors
    :class:`Chebfun3`'s evaluation).

    Construction is NOT JIT-safe.  Evaluation IS JIT-safe.

    Provenance
    ----------
    MATLAB source : @chebfun3t/chebfun3t.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2023 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Chebfun3, chebfun3t
    """

    core: jax.Array       # Tucker core tensor, shape (rx, ry, rz)
    cols: list            # list of Chebtech2 — x-direction factor functions
    rows: list            # list of Chebtech2 — y-direction factor functions
    tubes: list           # list of Chebtech2 — z-direction factor functions
    domain: tuple = eqx.field(static=True)  # (xa, xb, ya, yb, za, zb)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_chebfun3(cls, f: Chebfun3) -> "Chebfun3T":
        """Extract the Tucker decomposition data from a Chebfun3.

        Parameters
        ----------
        f : Chebfun3
            A fully constructed Chebfun3.

        Returns
        -------
        Chebfun3T
            The Tucker data wrapper.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from chebfunjax.chebfun3d.chebfun3 import chebfun3
        >>> from chebfunjax.chebfun3d.chebfun3t import Chebfun3T
        >>> g = chebfun3(lambda x, y, z: jnp.cos(x + y + z))
        >>> t = Chebfun3T.from_chebfun3(g)
        >>> t.rank == g.rank
        True

        Provenance
        ----------
        MATLAB source : @chebfun3/core.m, @chebfun3/cols.m
        Chebfun commit: 7574c77
        """
        return cls(
            core=f.core,
            cols=list(f.cols),
            rows=list(f.rows),
            tubes=list(f.tubes),
            domain=f.domain,
        )

    @classmethod
    def from_arrays(
        cls,
        core: jax.Array,
        cols: list[Chebtech2],
        rows: list[Chebtech2],
        tubes: list[Chebtech2],
        domain: tuple[float, float, float, float, float, float] = (
            -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
        ),
    ) -> "Chebfun3T":
        """Construct from explicit Tucker data.

        Parameters
        ----------
        core : jax.Array, shape (rx, ry, rz)
            Tucker core tensor.
        cols : list of Chebtech2
            Column (x) factor functions, length rx.
        rows : list of Chebtech2
            Row (y) factor functions, length ry.
        tubes : list of Chebtech2
            Tube (z) factor functions, length rz.
        domain : 6-tuple of floats, optional
            (xa, xb, ya, yb, za, zb).

        Returns
        -------
        Chebfun3T

        Provenance
        ----------
        MATLAB source : @chebfun3t/chebfun3t.m (constructor)
        Chebfun commit: 7574c77
        """
        rx, ry, rz = int(core.shape[0]), int(core.shape[1]), int(core.shape[2])
        if len(cols) != rx:
            raise ValueError(
                f"Chebfun3T.from_arrays: len(cols)={len(cols)} != rx={rx}."
            )
        if len(rows) != ry:
            raise ValueError(
                f"Chebfun3T.from_arrays: len(rows)={len(rows)} != ry={ry}."
            )
        if len(tubes) != rz:
            raise ValueError(
                f"Chebfun3T.from_arrays: len(tubes)={len(tubes)} != rz={rz}."
            )
        return cls(
            core=jnp.asarray(core, dtype=jnp.float64),
            cols=list(cols),
            rows=list(rows),
            tubes=list(tubes),
            domain=tuple(float(v) for v in domain),
        )

    # ------------------------------------------------------------------
    # Round-trip conversion
    # ------------------------------------------------------------------

    def to_chebfun3(self) -> Chebfun3:
        """Convert back to a :class:`~chebfunjax.chebfun3d.chebfun3.Chebfun3`.

        Returns
        -------
        Chebfun3
            A Chebfun3 with the same Tucker data.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from chebfunjax.chebfun3d.chebfun3 import chebfun3
        >>> from chebfunjax.chebfun3d.chebfun3t import Chebfun3T
        >>> g = chebfun3(lambda x, y, z: jnp.exp(-x**2 - y**2 - z**2))
        >>> t = Chebfun3T.from_chebfun3(g)
        >>> g2 = t.to_chebfun3()
        >>> abs(float(g(0.5, -0.3, 0.1)) - float(g2(0.5, -0.3, 0.1))) < 1e-14
        True

        Provenance
        ----------
        MATLAB source : @chebfun3t/chebfun3.m (conversion method)
        Chebfun commit: 7574c77
        """
        return Chebfun3(
            cols=self.cols,
            rows=self.rows,
            tubes=self.tubes,
            core=self.core,
            domain=self.domain,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def __call__(
        self,
        x: jax.Array,
        y: jax.Array,
        z: jax.Array,
    ) -> jax.Array:
        """Evaluate the Tucker approximation at (x, y, z).

        Delegates to the equivalent :class:`Chebfun3` evaluation.

        Parameters
        ----------
        x : jax.Array, scalar or shape (m,)
            x-coordinates in [xa, xb].
        y : jax.Array, scalar or shape (m,)
            y-coordinates in [ya, yb].
        z : jax.Array, scalar or shape (m,)
            z-coordinates in [za, zb].

        Returns
        -------
        jax.Array
            Function values; same shape as broadcast(x, y, z).

        Notes
        -----
        JIT-safe, grad-safe, vmap-safe.

        Provenance
        ----------
        MATLAB source : @chebfun3/feval.m
        Chebfun commit: 7574c77
        """
        return self.to_chebfun3()(x, y, z)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def rank(self) -> tuple[int, int, int]:
        """Tucker rank (rx, ry, rz)."""
        return (len(self.cols), len(self.rows), len(self.tubes))

    def core_numpy(self) -> np.ndarray:
        """Return the Tucker core as a NumPy array.

        Useful for inspection, plotting, or interfacing with NumPy/SciPy.

        Returns
        -------
        np.ndarray, shape (rx, ry, rz)

        Provenance
        ----------
        MATLAB source : @chebfun3/core.m
        Chebfun commit: 7574c77
        """
        return np.array(self.core, dtype=np.float64)

    def col_coeffs(self) -> list[np.ndarray]:
        """Chebyshev coefficients of each column (x) factor function.

        Returns
        -------
        list of np.ndarray
            One array of coefficients per column factor, ascending order.

        Provenance
        ----------
        MATLAB source : @chebfun3/cols.m
        Chebfun commit: 7574c77
        """
        return [np.array(c.coeffs, dtype=np.float64) for c in self.cols]

    def row_coeffs(self) -> list[np.ndarray]:
        """Chebyshev coefficients of each row (y) factor function.

        Returns
        -------
        list of np.ndarray

        Provenance
        ----------
        MATLAB source : @chebfun3/rows.m
        Chebfun commit: 7574c77
        """
        return [np.array(r.coeffs, dtype=np.float64) for r in self.rows]

    def tube_coeffs(self) -> list[np.ndarray]:
        """Chebyshev coefficients of each tube (z) factor function.

        Returns
        -------
        list of np.ndarray

        Provenance
        ----------
        MATLAB source : @chebfun3/tubes.m
        Chebfun commit: 7574c77
        """
        return [np.array(t.coeffs, dtype=np.float64) for t in self.tubes]

    # ------------------------------------------------------------------
    # Triple integral via Tucker structure
    # ------------------------------------------------------------------

    def sum3(self) -> jax.Array:
        """Definite triple integral over the domain.

        Delegates to :meth:`~chebfunjax.chebfun3d.chebfun3.Chebfun3.sum3`.

        Returns
        -------
        jax.Array, scalar

        Provenance
        ----------
        MATLAB source : @chebfun3/sum3.m
        Chebfun commit: 7574c77
        """
        return self.to_chebfun3().sum3()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Compact display.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from chebfunjax.chebfun3d.chebfun3 import chebfun3
        >>> from chebfunjax.chebfun3d.chebfun3t import Chebfun3T
        >>> g = chebfun3(lambda x, y, z: jnp.cos(x + y + z))
        >>> t = Chebfun3T.from_chebfun3(g)
        >>> "Chebfun3T" in repr(t)
        True

        Provenance
        ----------
        MATLAB source : @chebfun3t/disp.m
        Chebfun commit: 7574c77
        """
        xa, xb, ya, yb, za, zb = self.domain
        rx, ry, rz = self.rank
        return (
            f"Chebfun3T(rank=({rx}, {ry}, {rz}), "
            f"domain=(({xa}, {xb}), ({ya}, {yb}), ({za}, {zb})))"
        )


# ============================================================================
# Factory function
# ============================================================================


def chebfun3t(f_or_chebfun3, domain=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0), **kwargs):
    """Construct a Chebfun3T from a function or an existing Chebfun3.

    Parameters
    ----------
    f_or_chebfun3 : callable or Chebfun3
        If a :class:`~chebfunjax.chebfun3d.chebfun3.Chebfun3`, wraps it
        directly.  If callable, first constructs a Chebfun3 then wraps it.
    domain : 6-tuple of floats, optional
        (xa, xb, ya, yb, za, zb).  Ignored when ``f_or_chebfun3`` is a
        Chebfun3.
    **kwargs
        Extra keyword arguments forwarded to
        :func:`~chebfunjax.chebfun3d.chebfun3.chebfun3` (e.g. ``tol``,
        ``max_rank``).

    Returns
    -------
    Chebfun3T

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from chebfunjax.chebfun3d.chebfun3t import chebfun3t
    >>> t = chebfun3t(lambda x, y, z: jnp.exp(-x**2 - y**2 - z**2))
    >>> isinstance(t.rank, tuple) and len(t.rank) == 3
    True

    Provenance
    ----------
    MATLAB source : @chebfun3t/chebfun3t.m
    Chebfun commit: 7574c77
    """
    if isinstance(f_or_chebfun3, Chebfun3):
        return Chebfun3T.from_chebfun3(f_or_chebfun3)
    else:
        from chebfunjax.chebfun3d.chebfun3 import chebfun3 as _chebfun3
        g = _chebfun3(f_or_chebfun3, domain=domain, **kwargs)
        return Chebfun3T.from_chebfun3(g)
