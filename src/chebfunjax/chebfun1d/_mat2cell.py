"""Native MAT2CELL grouping with an explicit flat Python cell adapter.

Pin7574c77 @chebfun/mat2cell.m. CellList keeps existing list indexing and
iteration; shape records the MATLAB row/column cell layout, without nesting.
"""

import jax.numpy as jnp


class CellList(list):
    """Flat list of cells; shape is the corresponding native cell layout.

    Provenance
    ----------
    MATLAB source : @chebfun/mat2cell.m
    Chebfun commit: 7574c77
    """

    def __init__(self, values, *, row=False):
        super().__init__(values)
        self._cell_column = bool(row)

    @property
    def shape(self):
        return (len(self), 1) if self._cell_column else (1, len(self))


def source_sizes(count, row, sizes, n):
    """Native fixed dimension, component sum, and size error identifier.

    Provenance
    ----------
    MATLAB source : @chebfun/mat2cell.m
    Chebfun commit: 7574c77
    """

    def parsed(value):
        try:
            array = jnp.asarray(value)
            if array.ndim > 2 or jnp.iscomplexobj(array) or not bool(jnp.all(jnp.isfinite(array))):
                raise ValueError
            flat = array.reshape(-1)
            if not bool(jnp.all((flat >= 0) & (flat == jnp.floor(flat)))):
                raise ValueError
            return [int(t) for t in flat]
        except (TypeError, ValueError):
            raise ValueError(
                "CHEBFUN:CHEBFUN:mat2cell:size: Input sizes must be nonnegative integer component counts."
            ) from None

    if n is not None:
        m_counts, n_counts = parsed(sizes), parsed(n)
        free, fixed = (m_counts, n_counts) if row else (n_counts, m_counts)
        if fixed != [1]:
            raise ValueError(
                "CHEBFUN:CHEBFUN:mat2cell:size: Input arguments must sum to each input dimension."
            )
    else:
        free = [1] * count if sizes is None else parsed(sizes)
    if sum(free) != count:
        raise ValueError(
            "CHEBFUN:CHEBFUN:mat2cell:size: Input arguments must sum to each input dimension."
        )
    return free
