"""Small public utilities for the dense ``Chebfun3T`` representation.

The methods mirror the public utility surface in pinned MATLAB ``@chebfun3t``.
Tensor dimensions accepted by ``unfold``, ``fold`` and ``txm`` are one-based,
as in MATLAB. The Python ``length(outputs=3)`` option represents MATLAB's
three-output form.

Provenance
----------
MATLAB source: ``@chebfun3t/{norm,length,unfold,fold,txm,vals2coeffs,
coeffs2vals}.m``
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and the
Chebfun Developers.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from chebfunjax.chebfun3d._chebfun3t_arithmetic import Chebfun3TArithmeticError


def _dimension_tuple(dimensions):
    try:
        return tuple(int(d) for d in dimensions)
    except TypeError:
        return (int(dimensions),)


def _native_tensor(tensor, pad_to=3):
    """Apply MATLAB size padding, optionally preserving arbitrary rank."""
    tensor = jnp.asarray(tensor)
    if pad_to is not None and tensor.ndim > pad_to:
        raise ValueError("Chebfun3T transforms support tensors up to three dimensions")
    if tensor.ndim == 0:
        target = max(2, pad_to or 0)
        shape = (1,) * target
    else:
        target = max(2, tensor.ndim) if pad_to is None else pad_to
        shape = tensor.shape + (1,) * max(0, target - tensor.ndim)
    return jnp.reshape(tensor, shape)


class Chebfun3TUtilitiesMixin:
    """Public source-shaped norm, length and tensor transform methods."""

    coeffs: object

    def norm(self, p="fro"):
        """Return the native Frobenius-like norm.

        Native ``norm.m`` computes ``sqrt(sum3(f.^2))`` (without a conjugate
        absolute value despite its descriptive comment). This literal rule
        therefore preserves a complex result for a purely imaginary input.

        Provenance
        ----------
        MATLAB source: ``@chebfun3t/norm.m``
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        if self.isempty():
            return jnp.array([])
        if p != "fro":
            raise Chebfun3TArithmeticError(
                "CHEBFUN3T does not support this norm.",
                identifier="CHEBFUN:CHEBFUN3T:norm:norm",
            )
        return jnp.sqrt((self ** 2).sum3())

    def length(self, outputs=1):
        """Return max coefficient dimension or the three dimensions.

        ``outputs=3`` is the Python representation of MATLAB's three-output
        ``[m,n,p] = length(f)`` call. Empty objects return empty arrays, as in
        the source method.

        Provenance
        ----------
        MATLAB source: ``@chebfun3t/length.m``
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        if outputs not in (1, 3):
            raise ValueError("outputs must be 1 or 3")
        if self.isempty():
            empty = jnp.array([])
            return empty if outputs == 1 else (empty, empty, empty)
        dims = tuple(int(n) for n in self.coeffs.shape)
        return max(dims) if outputs == 1 else dims

    @staticmethod
    def vals2coeffs(values):
        """Convert values to coefficients, padding rank below three.

        Provenance
        ----------
        MATLAB source: ``@chebfun3t/vals2coeffs.m``
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        from chebfunjax.chebfun3d.chebfun3t import _vals2coeffs

        return _vals2coeffs(_native_tensor(values))

    @staticmethod
    def coeffs2vals(coeffs):
        """Convert coefficients to tensor-grid values, with native rank padding.

        Provenance
        ----------
        MATLAB source: ``@chebfun3t/coeffs2vals.m``
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        from chebfunjax.chebfun3d.chebfun3t import _coeffs2vals

        return _coeffs2vals(_native_tensor(coeffs))

    @staticmethod
    def unfold(tensor, row_dims, col_dims=None):
        """Matricize a tensor using native one-based dimension groups.

        Rank-one inputs are treated as native ``n x 1`` arrays. Omitting
        ``col_dims`` uses the complement of ``row_dims`` as described in the
        MATLAB help text.

        Provenance
        ----------
        MATLAB source: ``@chebfun3t/unfold.m``
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        tensor = _native_tensor(tensor, pad_to=None)
        ndim = tensor.ndim
        rows = tuple(int(d) - 1 for d in _dimension_tuple(row_dims)
                     if col_dims is None or int(d) <= ndim)
        cols = (tuple(d for d in range(ndim) if d not in rows)
                if col_dims is None else tuple(
                    int(d) - 1 for d in _dimension_tuple(col_dims)
                    if int(d) <= ndim
                ))
        perm = rows + cols
        if sorted(perm) != list(range(ndim)):
            raise ValueError("row_dims and col_dims must partition tensor dimensions")
        ordered = jnp.transpose(tensor, perm)
        nrows = 1
        for d in rows:
            nrows *= tensor.shape[d]
        ncols = 1
        for d in cols:
            ncols *= tensor.shape[d]
        return jnp.reshape(ordered, (nrows, ncols), order="F")

    @staticmethod
    def fold(matrix, size_tensor, row_dims, col_dims=None):
        """Inverse ``unfold`` using native one-based dimension groups.

        Omitting ``col_dims`` uses the documented complement convention. The
        pinned MATLAB implementation's three-argument default references an
        undefined variable, so this optional-argument behavior is a Python
        convenience extension.

        Provenance
        ----------
        MATLAB source: ``@chebfun3t/fold.m``
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        matrix = jnp.asarray(matrix)
        dims = tuple(int(n) for n in size_tensor)
        if len(dims) < 2:
            dims = dims + (1,) * (2 - len(dims))
        ndim = len(dims)
        rows = tuple(int(d) - 1 for d in _dimension_tuple(row_dims)
                     if int(d) <= ndim)
        cols = (tuple(d for d in range(ndim) if d not in rows)
                if col_dims is None else tuple(
                    int(d) - 1 for d in _dimension_tuple(col_dims)
                    if int(d) <= ndim
                ))
        perm = rows + cols
        if sorted(perm) != list(range(ndim)):
            raise ValueError("row_dims and col_dims must partition tensor dimensions")
        ordered_shape = tuple(dims[d] for d in perm)
        if matrix.size != math.prod(ordered_shape):
            raise ValueError("matrix size does not match size_tensor")
        ordered = jnp.reshape(matrix, ordered_shape, order="F")
        inverse = tuple(perm.index(d) for d in range(ndim))
        return jnp.transpose(ordered, inverse)

    @staticmethod
    def txm(tensor, matrix, mode):
        """Multiply tensor by a matrix along a native one-based mode.

        Source extends ``size(T)`` with singleton dimensions when ``mode``
        exceeds ``ndims(T)``. A nonmatching matrix dimension raises the native
        ``CHEBFUN:CHEBFUN3:txm`` identifier.

        Provenance
        ----------
        MATLAB source: ``@chebfun3t/txm.m``
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford and the
            Chebfun Developers.
        """
        from chebfunjax.chebfun3d.chebfun3t import _txm

        mode = int(mode)
        if mode < 1:
            raise ValueError("mode must be a positive one-based tensor dimension")
        tensor = _native_tensor(tensor, pad_to=None)
        if mode > tensor.ndim:
            tensor = jnp.reshape(tensor, tensor.shape + (1,) * (mode - tensor.ndim))
        matrix = jnp.asarray(matrix)
        if matrix.ndim != 2 or tensor.shape[mode - 1] != matrix.shape[1]:
            raise Chebfun3TArithmeticError(
                "Tensor-matrix dimensions do not agree.",
                identifier="CHEBFUN:CHEBFUN3:txm",
            )
        return _txm(tensor, matrix, mode - 1)
