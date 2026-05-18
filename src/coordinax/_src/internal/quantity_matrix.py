"""Heterogeneous unit containers for vectors and matrices.

This module provides two closely related building blocks:

- `UnitsMatrix`, an immutable nested tuple of units with indexing support
- `QMatrix`, a quantity-like wrapper around one array plus a matching
    static `UnitsMatrix`

The numeric payload is a single JAX array of shape ``(..., *shape)`` where the
trailing dimensions are the logical vector or matrix dimensions and any leading
dimensions are batch dimensions. Units are stored separately as a static nested
tuple structure with the same logical shape, allowing every element to carry
its own physical unit.

Currently the public surface supports only 1-D and 2-D structures:

- 1-D: ``(..., N)`` with units ``(u0, u1, ..., uN-1)``
- 2-D: ``(..., N, M)`` with units ``((u00, u01, ...), (u10, u11, ...), ...)``

Quax primitive dispatches (``add_p``, ``dot_general_p``) perform the
necessary per-element unit conversions via `unxt.uconvert_value` — which
correctly handles affine conversions (e.g. °F → °C), not just
multiplicative scale factors.
"""

__all__ = (
    "QMatrix",
    "UnitsMatrix",
    "cdict_units",
    "det",
    "det_p",
    "inv",
    "inv_p",
)


import functools as ft
import operator

from jaxtyping import Array, Shaped
from typing import Any, NoReturn, TypeAlias, TypeVar, cast, final

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import plum
import quax
from jax import lax
from jax.extend import core as jexc
from jax.interpreters import ad as jax_ad, batching as jax_batching, mlir as jax_mlir

import unxt as u
from unxt.quantity import AllowValue

CDict: TypeAlias = dict[str, Any]
_DMLS = u.unit("")


def strict_zip(*args: Any) -> zip:
    """Zip iterables while enforcing equal lengths."""
    return zip(*args, strict=True)


##############################################################################
# Units helpers

PackedUnitOutput: TypeAlias = tuple[u.AbstractUnit | None, ...]


def cdict_units(p: CDict, keys: tuple[str, ...], /) -> PackedUnitOutput:
    """Extract per-key units from a component dictionary.

    Non-quantity entries yield `None`, so the output tuple can be used for
    heterogeneous dictionaries containing both quantity and non-quantity data.

    >>> import unxt as u
    >>> d = {'x': u.Q(1.0, 'm'), 'y': 2.0, 'z': u.Q(3.0, 'kg')}
    >>> cdict_units(d, ('x', 'y', 'z'))
    (Unit("m"), None, Unit("kg"))

    """
    # `unit_of()` returns None for non-quantities, so this works for both cases.
    return cast("PackedUnitOutput", tuple(u.unit_of(p[k]) for k in keys))


##############################################################################
# UnitsMatrix

T = TypeVar("T")

NestedTuple: TypeAlias = T | tuple["NestedTuple[T]", ...]
UnitTree: TypeAlias = NestedTuple[u.AbstractUnit]


def _normalize_unit(x: Any, /) -> u.AbstractUnit:
    """Convert *x* to an ``AbstractUnit``; accept unit strings and AbstractUnit.

    Raises ``TypeError`` for unsupported types.
    """
    if isinstance(x, str):
        return u.unit(x)  # ty: ignore[invalid-return-type]
    if isinstance(x, u.AbstractUnit):
        return x
    msg = f"Expected an AbstractUnit or unit string; got {type(x).__name__!r}"
    raise TypeError(msg)


def _build_object_array(iterable: Any, /) -> np.ndarray:  # noqa: C901
    """Build a 1-D or 2-D numpy object array of ``AbstractUnit`` from *iterable*.

    Accepts:

    - A numpy object array (element-normalize and validate ndim).
    - A plain tuple/list of units or unit strings → 1-D output.
    - A plain tuple/list of tuples of units or unit strings → 2-D output.

    Raises ``TypeError`` if a non-sequence is passed, ``ValueError`` if the
    structure is ragged or has unsupported ndim.
    """
    if isinstance(iterable, np.ndarray) and iterable.dtype == object:
        if iterable.ndim not in (1, 2):
            msg = f"UnitsMatrix only supports 1D or 2D; got ndim={iterable.ndim}"
            raise ValueError(msg)
        flat = [_normalize_unit(v) for v in iterable.flat]
        data: np.ndarray = np.empty(iterable.shape, dtype=object)
        data.flat[:] = flat
        return data

    # Sequence path: tuple, list, or any iterable
    items = list(iterable)  # raises TypeError if not iterable

    if not items:
        raise ValueError("UnitsMatrix requires at least one element")

    first = items[0]
    if isinstance(first, (tuple, list)):
        # 2-D: sequence of rows — validate and fill in one pass
        n, m = len(items), len(first)
        data = np.empty((n, m), dtype=object)
        for i, row in enumerate(items):
            if not isinstance(row, (tuple, list)) or len(row) != m:
                raise ValueError("ragged structure")
            for j, v in enumerate(row):
                if isinstance(v, (tuple, list)):
                    raise ValueError("ragged structure")  # noqa: TRY004
                data[i, j] = _normalize_unit(v)
        return data

    # 1-D: sequence of units / unit strings
    n = len(items)
    data = np.empty(n, dtype=object)
    for i, v in enumerate(items):
        if isinstance(v, (tuple, list)):  # Mixed leaf/nested → ragged
            raise ValueError("ragged structure")  # noqa: TRY004
        data[i] = _normalize_unit(v)
    return data


@final
class UnitsMatrix:
    """Immutable, hashable unit structure for `QMatrix`.

    `UnitsMatrix` wraps a numpy object array (``dtype=object``) of
    `~unxt.AbstractUnit` elements. Only 1-D and 2-D structures are accepted.

    The class supports tuple-style indexing, iteration, `to_tuple()`, and
    `to_string()`. It is **not** a subclass of `astropy.units.StructuredUnit`;
    bidirectional converters to/from ``StructuredUnit`` are provided in
    ``coordinax.interop.astropy``.

    Hashability is achieved via ``hash(self.to_tuple())``, so the underlying
    ``AbstractUnit`` objects must themselves be hashable (they are).

    For 1D: ``UnitsMatrix(("m", "s", "kg"))``
    For 2D: ``UnitsMatrix((("m", "s"), ("kg", "rad")))``

    Examples
    --------
    >>> import unxt as u
    >>> from coordinax.internal import UnitsMatrix

    1D case:

    >>> units_1d = UnitsMatrix(("m", "s", "kg"))
    >>> units_1d.shape
    (3,)
    >>> units_1d[0]
    Unit("m")
    >>> units_1d.to_string()
    '(m, s, kg)'

    2D case:

    >>> units_2d = UnitsMatrix((("m", "s"), ("kg", "rad")))
    >>> units_2d.shape
    (2, 2)
    >>> units_2d[0, 1]
    Unit("s")
    >>> units_2d.to_string()
    '((m, s), (kg, rad))'

    """

    __slots__ = ("_units",)

    def __init__(self, iterable: Any, /) -> None:
        if isinstance(iterable, UnitsMatrix):
            # Copy from another UnitsMatrix — avoids sharing the mutable array.
            data = iterable._units.copy()
        else:
            data = _build_object_array(iterable)
        if data.ndim not in (1, 2):
            msg = f"UnitsMatrix only supports 1D or 2D, but got ndim={data.ndim}"
            raise ValueError(msg)
        self._units = data

    # ── Shape / structure ─────────────────────────────────────────────

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the N-D unit structure."""
        return tuple(self._units.shape)

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return int(self._units.ndim)

    @property
    def T(self) -> "UnitsMatrix":
        """Compute the all-axis units array transpose.

        Examples
        --------
        >>> from coordinax.internal import UnitsMatrix

        >>> units = UnitsMatrix(("m", "s"))
        >>> units.T
        UnitsMatrix("(m, s)")

        >>> units = UnitsMatrix((("m", "s"), ("kg", "rad")))
        >>> units.T
        UnitsMatrix("((m, kg), (s, rad))")

        >>> units = UnitsMatrix((("m", "s", "kg"), ("Hz", "candela", "km")))
        >>> units.T
        UnitsMatrix("((m, Hz), (s, cd), (kg, km))")

        """
        return UnitsMatrix(self._units.T)

    def inverse(self) -> "UnitsMatrix":
        r"""Inverse unit structure — each unit raised to the power -1.

        For a 1-D (diagonal) ``UnitsMatrix`` the inversion is done
        entry-by-entry in *O(n)*, providing a speedup over the general 2-D
        case.  For a 2-D ``UnitsMatrix`` with a uniform unit (all entries
        equal) the reciprocal is computed once and broadcast in *O(1)*;
        mixed-unit 2-D structures fall back to an element-wise *O(nm)* loop.

        Examples
        --------
        >>> from coordinax.internal import UnitsMatrix

        1-D (diagonal) case — element-wise reciprocal:

        >>> UnitsMatrix(("m2", "s2")).inverse()
        UnitsMatrix("(1 / m2, 1 / s2)")

        2-D uniform-unit case:

        >>> UnitsMatrix((("m2", "m2"), ("m2", "m2"))).inverse()
        UnitsMatrix("((1 / m2, 1 / m2), (1 / m2, 1 / m2))")

        2-D mixed-unit case:

        >>> UnitsMatrix((("m2", "s2"), ("s2", "rad2"))).inverse()
        UnitsMatrix("((1 / m2, 1 / s2), (1 / s2, 1 / rad2))")

        """
        inv_data = np.empty(self._units.shape, dtype=object)
        if self._units.ndim == 1:
            # Diagonal speedup: 1-D represents a diagonal metric's units.
            for i in range(self._units.shape[0]):
                inv_data[i] = self._units[i] ** (-1)
        else:
            # 2-D: fast path when all entries share the same unit.
            flat = self._units.ravel()
            first = flat[0]
            if all(u == first for u in flat[1:]):
                inv_unit = first ** (-1)
                inv_data[:] = inv_unit
            else:
                n, m = self._units.shape
                for i in range(n):
                    for j in range(m):
                        inv_data[i, j] = self._units[i, j] ** (-1)
        return UnitsMatrix(inv_data)

    # ── Serialization ─────────────────────────────────────────────────

    def to_tuple(self) -> UnitTree:
        """Convert to a nested tuple of `~unxt.AbstractUnit` objects.

        Examples
        --------
        >>> from coordinax.internal import UnitsMatrix
        >>> import unxt as u
        >>> UnitsMatrix(("m", "s")).to_tuple()
        (Unit("m"), Unit("s"))

        """
        if self._units.ndim == 1:
            return tuple(self._units)
        return tuple(map(tuple, self._units))

    def to_string(self) -> str:
        """Return a human-readable string representation of the unit structure.

        Examples
        --------
        >>> from coordinax.internal import UnitsMatrix
        >>> UnitsMatrix(("m", "s", "kg")).to_string()
        '(m, s, kg)'
        >>> UnitsMatrix((("m", "s"), ("kg", "rad"))).to_string()
        '((m, s), (kg, rad))'

        """
        if self._units.ndim == 1:
            inner = ", ".join(str(x) for x in self._units)
            if len(self._units) == 1:
                return f"({inner},)"
            return f"({inner})"
        # 2D
        row_strs = []
        for row in self._units:
            inner = ", ".join(str(x) for x in row)
            row_strs.append(f"({inner},)" if len(row) == 1 else f"({inner})")
        if len(self._units) == 1:
            return f"({row_strs[0]},)"
        return f"({', '.join(row_strs)})"

    # ── Python data model ─────────────────────────────────────────────

    def __repr__(self) -> str:
        return f'UnitsMatrix("{self.to_string()}")'

    def __eq__(self, other: Any, /) -> bool:
        if isinstance(other, UnitsMatrix):
            if self._units.shape != other._units.shape:
                return False
            return bool(np.all(self._units == other._units))
        if isinstance(other, (tuple, list)):
            try:
                return self == UnitsMatrix(other)
            except (TypeError, ValueError):
                return False
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.to_tuple())

    def __iter__(self) -> Any:
        """Iterate over elements (1D) or row ``UnitsMatrix`` objects (2D).

        Examples
        --------
        >>> from coordinax.internal import UnitsMatrix
        >>> list(UnitsMatrix(("m", "rad", "rad")))
        [Unit("m"), Unit("rad"), Unit("rad")]

        """
        if self._units.ndim == 1:
            yield from self._units
            return
        for row in self._units:
            yield UnitsMatrix(row)

    def __getitem__(self, index: Any, /) -> Any:
        """Index into the UnitsMatrix to retrieve a unit or sub-structure.

        >>> from coordinax.internal import UnitsMatrix
        >>> units = UnitsMatrix((("m", "s"), ("kg", "rad")))

        Indexing a single element returns a unit:

        >>> units[0, 1]
        Unit("s")

        Indexing a row returns a UnitsMatrix:

        >>> units[0]
        UnitsMatrix("(m, s)")

        """
        result = self._units[index]
        if isinstance(result, np.ndarray):
            if result.ndim == 0:  # 0-d array -> extract the contained unit.
                return result.item()
            return UnitsMatrix(result)
        return result


@plum.dispatch
def unit(tuple_of_units: tuple[Any, ...], /) -> UnitsMatrix:
    """Convert a nested tuple of units into a ``UnitsMatrix``.

    This allows users to specify units in a convenient nested tuple format when
    constructing ``QMatrix`` instances, and have them automatically
    converted to the appropriate ``UnitsMatrix``.

    >>> import unxt as u

    1D case:

    >>> u.unit(("m", "s", "kg"))
    UnitsMatrix("(m, s, kg)")

    2D case:

    >>> u.unit((("m", "s"), ("kg", "rad")))
    UnitsMatrix("((m, s), (kg, rad))")

    """
    return UnitsMatrix(tuple_of_units)


@plum.dispatch
def unit(arr: np.ndarray, /) -> UnitsMatrix:
    """Convert a numpy object array of units into a ``UnitsMatrix``.

    >>> import numpy as np
    >>> import unxt as u
    >>> from coordinax.internal import UnitsMatrix
    >>> arr = np.array([u.unit("m"), u.unit("s")], dtype=object)
    >>> u.unit(arr)
    UnitsMatrix("(m, s)")

    """
    return UnitsMatrix(arr)


@plum.dispatch
def unit(obj: UnitsMatrix, /) -> UnitsMatrix:
    """Identity: a UnitsMatrix is returned unchanged by the unit converter."""
    return obj


@plum.dispatch
def unit_of(obj: UnitsMatrix, /) -> UnitsMatrix:
    """Identity conversion for UnitsMatrix to itself.

    >>> import unxt as u
    >>> unit = u.unit(("m", "s", "kg"))
    >>> u.unit_of(unit) is unit
    True

    """
    return obj


##############################################################################
# QMatrix


class QMatrix(u.AbstractQuantity):
    """Quantity container whose elements may each carry different units.

    `QMatrix` stores one numeric array together with a static
    `UnitsMatrix` describing the unit of each logical element. The shape of the
    unit structure determines whether the object behaves as a heterogeneous
    vector or matrix.

    Only 1-D and 2-D logical structures are supported.

    Parameters
    ----------
    value : Array, shape ``(..., *shape)``
        Numeric payload. For 1D: ``(..., N)``. For 2D: ``(..., N, M)``.
        The value of element ``[i]`` (1D) or ``[i, j]`` (2D) is expressed
        in the corresponding unit.
    unit : UnitsMatrix
        Per-element units. For 1D: ``(u0, u1, ...)``.
        For 2D: ``((u00, u01, ...), (u10, u11, ...), ...)``.
        Must be a static (hashable) nested tuple structure whose shape
        matches the trailing dimensions of ``value``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> from coordinax.internal import QMatrix

    1D case (vector):

    >>> qv = QMatrix(jnp.array([1.0, 2.0, 3.0]), unit=("m", "s", "kg"))
    >>> qv.value
    Array([1., 2., 3.], dtype=float64)
    >>> qv.unit.shape
    (3,)

    >>> 2 * qv
    QMatrix([2., 4., 6.], '(m, s, kg)')

    >>> qv2 = QMatrix(jnp.array([0.1, 200.0, 300.0]), unit=("km", "ms", "g"))
    >>> qv + qv2
    QMatrix([101. ,   2.2,   3.3], '(m, s, kg)')

    2D case (matrix):

    >>> qm = QMatrix(jnp.ones((2, 2)), unit=(("m", "s"), ("kg", "rad")))
    >>> qm.value.shape
    (2, 2)
    >>> qm.unit.shape
    (2, 2)

    >>> 2 * qm
    QMatrix([[2., 2.],
                    [2., 2.]], '((m, s), (kg, rad))')

    >>> qm2 = QMatrix(jnp.array([[0.1, 200.0], [300.0, 0.5]]),
    ...                      unit=(("km", "ms"), ("g", "deg")))
    >>> qm + qm2
    QMatrix([[101.        ,   1.2       ],
                    [  1.3       ,   1.00872665]], '((m, s), (kg, rad))')

    Indexing:

    >>> qv[0]
    Q(1., 'm')
    >>> qm[0]
    QMatrix([1., 1.], '(m, s)')
    >>> qm[1, 0]
    Q(1., 'kg')

    """

    value: Shaped[Array, "..."] = eqx.field()
    unit: UnitsMatrix = eqx.field(static=True, converter=u.unit)  # ty: ignore[invalid-assignment]

    @property
    def ndim(self) -> int:
        """Number of real dimensions (1 for vector, 2 for matrix)."""
        return self.unit.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape, including batch dimensions."""
        return self.value.shape

    @classmethod
    def from_cdict(cls, v: CDict, /, keys: tuple[str, ...] | None = None) -> "QMatrix":
        """Pack a component dictionary into a 1-D ``QMatrix``.

        Each value in *v* is stripped to its numeric value and stacked into a
        single JAX array.  Values that carry units (``unxt.Quantity``) retain
        those units in the resulting ``UnitsMatrix``; plain arrays are treated
        as dimensionless.

        Examples
        --------
        >>> import unxt as u
        >>> from coordinax.internal import QMatrix

        From a dictionary of quantities:

        >>> v = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "s"), "z": u.Q(3.0, "kg")}
        >>> qv = QMatrix.from_cdict(v)
        >>> qv.unit.to_string()
        '(m, s, kg)'
        >>> qv.value
        Array([1., 2., 3.], dtype=float64, ...)

        Selecting and reordering a subset of keys:

        >>> qv2 = QMatrix.from_cdict(v, keys=("z", "x"))
        >>> qv2.unit.to_string()
        '(kg, m)'
        >>> qv2.value
        Array([3., 1.], dtype=float64, ...)

        Dimensionless entries (bare arrays) are accepted:

        >>> import jax.numpy as jnp
        >>> v2 = {"a": jnp.array(4.0), "b": u.Q(5.0, "m")}
        >>> qv3 = QMatrix.from_cdict(v2)
        >>> qv3.unit.to_string()
        '(, m)'

        """
        keys = tuple(v) if keys is None else keys
        vs = [v[k] for k in keys]
        us = [u.unit_of(x) or _DMLS for x in vs]
        svs = jnp.stack([u.ustrip(AllowValue, unt, x) for x, unt in strict_zip(vs, us)])
        return cls(svs, unit=UnitsMatrix(us))

    def __getitem__(self, index: Any, /) -> "u.Q | QMatrix":  # ty: ignore[invalid-method-override]
        """Index into the QMatrix to retrieve a specific element.

        Indexing a logical dimension returns a ``Quantity`` when the result is
        a scalar unit, or a ``QMatrix`` when the result still has
        structure.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import unxt as u
        >>> from coordinax.internal import QMatrix

        **1-D vector** — indexing a single element returns a ``Quantity``:

        >>> qv = QMatrix(jnp.array([1.0, 2.0, 3.0]), unit=("m", "s", "kg"))
        >>> qv[0]
        Q(1., 'm')
        >>> qv[2]
        Q(3., 'kg')

        **2-D matrix** — indexing a row returns a ``QMatrix``:

        >>> qm = QMatrix(jnp.ones((2, 3)),
        ...                     unit=(("m", "s", "kg"), ("rad", "deg", "m")))
        >>> qm[0]
        QMatrix([1., 1., 1.], '(m, s, kg)')

        Indexing a specific element returns a ``Quantity``:

        >>> qm[1, 2]
        Q(1., 'm')

        """
        value_item = self.value[index]
        unit_item = self.unit[index]
        if isinstance(unit_item, UnitsMatrix):
            return QMatrix(value=value_item, unit=unit_item)
        return u.Q(value_item, unit_item)

    # ── Quax API ─────────────────────────────────────────────────────

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(self.value.shape, self.value.dtype)

    def materialise(self) -> NoReturn:
        msg = "Refusing to materialise `QMatrix`."
        raise RuntimeError(msg)

    def diag(self) -> "QMatrix":
        """Return a 1-D ``QMatrix`` containing the diagonal of this matrix.

        Unlike ``qnp.diag``, this method operates directly on the static
        ``unit`` structure and the raw value array, so it works correctly under
        ``jax.jit`` and with heterogeneous-unit matrices.

        Only supported for 2-D ``QMatrix`` objects.

        Returns
        -------
        QMatrix
            1-D ``QMatrix`` of length ``min(n_rows, n_cols)`` whose
            ``unit[i]`` is ``self.unit[i, i]`` and whose ``value[..., i]`` is
            ``self.value[..., i, i]``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from coordinax.internal import QMatrix

        Uniform units:

        >>> A = QMatrix(jnp.diag(jnp.array([1.0, 4.0, 9.0])),
        ...                    unit=(("m", "m", "m"), ("m", "m", "m"), ("m", "m", "m")))
        >>> d = A.diag()
        >>> d.unit.shape
        (3,)
        >>> d.value
        Array([1., 4., 9.], dtype=float64)

        Heterogeneous units — works under jit:

        >>> B = QMatrix(jnp.diag(jnp.array([1.0, 2.0, 3.0])),
        ...                    unit=(("m", "s", "kg"),
        ...                          ("m", "s", "kg"),
        ...                          ("m", "s", "kg")))
        >>> db = B.diag()
        >>> db.unit.to_string()
        '(m, s, kg)'
        >>> db.value
        Array([1., 2., 3.], dtype=float64)

        """
        if self.ndim != 2:
            msg = f"QMatrix.diag() requires a 2D matrix, got ndim={self.ndim}"
            raise ValueError(msg)
        n = min(self.shape[-2], self.shape[-1])
        diag_value = jnp.stack([self.value[..., i, i] for i in range(n)], axis=-1)
        diag_unit = UnitsMatrix(self.unit._units.diagonal())
        return QMatrix(value=diag_value, unit=diag_unit)

    @property
    def T(self) -> "QMatrix":
        """Transpose a 2-D ``QMatrix`` (swap rows/columns and units).

        Returns a new ``QMatrix`` whose value array and unit structure
        are both transposed.  Only 2-D matrices are supported.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import quaxed.numpy as qnp
        >>> from coordinax.internal import QMatrix

        >>> a = QMatrix(jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        ...                    unit=(("m", "s"), ("kg", "rad")))
        >>> aT = a.T
        >>> aT.value
        Array([[1., 3.],
               [2., 4.]], dtype=float64)
        >>> aT.unit.to_string()
        '((m, kg), (s, rad))'

        Also accessible via ``jax.numpy.transpose``:

        >>> aT2 = qnp.matrix_transpose(a)
        >>> aT2.value
        Array([[1., 3.],
               [2., 4.]], dtype=float64)
        >>> aT2.unit.to_string()
        '((m, kg), (s, rad))'

        """
        if self.ndim != 2:
            msg = f"QMatrix.T requires a 2-D matrix, got ndim={self.ndim}"
            raise ValueError(msg)
        return QMatrix(value=jnp.swapaxes(self.value, -2, -1), unit=self.unit.T)


def _convert_value_vector(
    value: Shaped[Array, "*batch N"],
    from_units: tuple[u.AbstractUnit, ...],
    to_units: tuple[u.AbstractUnit, ...],
) -> Shaped[Array, "*batch N"]:
    """Convert every element of *value* from *from_units* to *to_units* (1D case).

    Each ``value[..., i]`` is converted individually via
    `u.uconvert_value` so that **all** conversion types are handled
    correctly.
    """
    n = len(to_units)
    return jnp.stack(
        [u.uconvert_value(to_units[i], from_units[i], value[..., i]) for i in range(n)],
        axis=-1,
    )


def _convert_value_matrix(
    value: Shaped[Array, "*batch N M"],
    from_units: tuple[tuple[u.AbstractUnit, ...], ...],
    to_units: tuple[tuple[u.AbstractUnit, ...], ...],
) -> Shaped[Array, "*batch N M"]:
    """Convert every element of *value* from *from_units* to *to_units* (2D case).

    Each ``value[..., i, j]`` is converted individually via
    `u.uconvert_value` so that **all** conversion types are handled
    correctly — including nonlinear ones like dB, mag, and dex (which
    are logarithmic, not affine).
    """
    n = len(to_units)
    m = len(to_units[0])
    return jnp.stack(
        [
            jnp.stack(
                [
                    u.uconvert_value(to_units[i][j], from_units[i][j], value[..., i, j])
                    for j in range(m)
                ],
                axis=-1,
            )
            for i in range(n)
        ],
        axis=-2,
    )


@plum.conversion_method(type_from=QMatrix, type_to=u.Q)
def QMatrix_to_quantity(x: QMatrix, /) -> u.Q:
    """Convert a ``QMatrix`` to a regular ``Quantity``.

    Conversion is only valid when all elements of ``x`` share the same unit.  If
    units are heterogeneous, this conversion is ambiguous and raises
    ``ValueError``.

    >>> import plum
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> from coordinax.internal import QMatrix

    Uniform units convert to a plain quantity:

    >>> qmat = QMatrix(jnp.array([1.0, 2.0, 3.0]), unit=("m", "m", "m"))
    >>> plum.convert(qmat, u.Q)
    Q([1., 2., 3.], 'm')

    Mixed units are rejected:

    >>> bad = QMatrix(jnp.array([1.0, 2.0]), unit=("m", "s"))
    >>> plum.convert(bad, u.Q)
    Traceback (most recent call last):
    ...
    ValueError: Cannot convert QMatrix to Quantity unless all units are
    identical.

    """
    units = jtu.tree_leaves(x.unit.to_tuple())

    if not units:
        msg = "Cannot convert QMatrix with no unit entries."
        raise ValueError(msg)

    first = units[0]
    if any(unit != first for unit in units[1:]):
        msg = "Cannot convert QMatrix to Quantity unless all units are identical."
        raise ValueError(msg)

    return u.Q(x.value, first)


# ── Primitive dispatches ─────────────────────────────────────────────
# All dispatches use `u.uconvert_value(to, from, val)` element-by-element
# so that affine unit conversions (e.g. temperature) are handled.


def _convert_value(
    value: Array,
    from_units: UnitsMatrix,
    to_units: UnitsMatrix,
) -> Array:
    """Convert value with heterogeneous units (works for both 1D and 2D)."""
    from_tup = from_units.to_tuple()
    to_tup = to_units.to_tuple()
    if from_units.ndim == 1:
        return _convert_value_vector(value, from_tup, to_tup)
    if from_units.ndim == 2:
        return _convert_value_matrix(value, from_tup, to_tup)
    msg = f"Unsupported ndim={from_units.ndim}"
    raise NotImplementedError(msg)


@plum.dispatch
def uconvert(to_units: UnitsMatrix, x: QMatrix, /) -> QMatrix:
    """Convert a ``QMatrix`` to different (but compatible) units.

    Unlike the generic astropy ``StructuredUnit.to()`` path, this dispatch uses
    ``_convert_value`` directly so that the regular 2D JAX array in ``x.value``
    is converted element-by-element without requiring a numpy structured array.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> from coordinax.internal import QMatrix

    >>> x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    >>> q = QMatrix(x, (("m", "rad"), ("m", "rad")))
    >>> target = u.unit((("km", "deg"), ("km", "deg")))
    >>> q.uconvert(target).unit.to_string()
    '((km, deg), (km, deg))'

    """
    if x.unit == to_units:
        return x
    value = _convert_value(x.value, x.unit, to_units)
    return QMatrix(value=value, unit=to_units)


@quax.register(lax.add_p)
def add_qm_qm(x: QMatrix, y: QMatrix, /) -> QMatrix:
    """Element-wise addition of two `QMatrix` objects.

    The result adopts the units of *x*.  Each element is converted from
    ``y.unit`` → ``x.unit`` before the numeric add.

    Works for both 1D (vector) and 2D (matrix) cases.

    >>> import jax.numpy as jnp
    >>> import quaxed.numpy as qnp
    >>> import unxt as u
    >>> from coordinax.internal import QMatrix

    2D case:

    >>> a = QMatrix(jnp.ones((2, 2)), unit=(("m", "s"), ("kg", "rad")))
    >>> b = QMatrix(jnp.ones((2, 2)), unit=(("km", "ms"), ("g", "deg")))

    >>> result = qnp.add(a, b)
    >>> result.unit.to_string()
    '((m, s), (kg, rad))'

    >>> result.value
    Array([[1.00100000e+03, 1.00100000e+00],
           [1.00100000e+00, 1.01745329e+00]], dtype=float64)

    1D case:

    >>> a1d = QMatrix(jnp.ones(3), unit=("m", "s", "kg"))
    >>> b1d = QMatrix(jnp.ones(3), unit=("km", "ms", "g"))

    >>> result1d = qnp.add(a1d, b1d)
    >>> result1d.unit.to_string()
    '(m, s, kg)'

    >>> result1d.value
    Array([1.001e+03, 1.001e+00, 1.001e+00], dtype=float64)

    """
    y_converted = _convert_value(y.value, y.unit, x.unit)
    return QMatrix(value=lax.add(x.value, y_converted), unit=x.unit)


@quax.register(lax.dot_general_p)
def dot_general_qm_qm(
    lhs: QMatrix,
    rhs: QMatrix,
    /,
    *,
    dimension_numbers: lax.DotDimensionNumbers,
    precision: Any = None,
    preferred_element_type: Any = None,
    **kw: Any,
) -> QMatrix | u.Q:
    """Dot product / matrix multiply two `QMatrix` objects.

    Delegates to specialized implementations based on the dimensionality:
    - 1D @ 1D → scalar (vector dot product)
    - 2D @ 2D → 2D (matrix-matrix multiply)

    For the standard matmul contraction: contracting_dims = ((-1,), (-2,)),
    with no batch dims (batch is handled by leading dims in QMatrix).

    >>> import jax.numpy as jnp
    >>> import quaxed.numpy as qnp
    >>> import unxt as u
    >>> from coordinax.internal import QMatrix

    1D @ 1D (dot product):

    >>> v1 = QMatrix(jnp.array([1.0, 2.0]), unit=("m", "km"))
    >>> v2 = QMatrix(jnp.array([3.0, 4.0]), unit=("s", "s"))
    >>> result = qnp.dot(v1, v2)
    >>> result.value
    Array(8003., dtype=float64)
    >>> result.unit
    Unit("m s")

    2D @ 2D (matrix multiply):

    >>> a = QMatrix(jnp.array([[1.0, 2.0], [3.0, 4.0]]),
    ...                    unit=(("m", "km"), ("m", "km")))
    >>> b = QMatrix(jnp.array([[1.0, 0.0], [0.0, 1.0]]),
    ...                    unit=(("s", "s"), ("s", "s")))

    >>> c = qnp.matmul(a, b)
    >>> c.unit.to_string()
    '((m s, m s), (m s, m s))'

    >>> c.value
    Array([[1.e+00, 2.e+03],
           [3.e+00, 4.e+03]], dtype=float64)

    """
    # For now, we only handle the standard matmul/dot contraction
    (contract, batch) = dimension_numbers
    assert len(contract[0]) == 1 and len(contract[1]) == 1  # noqa: PT018, S101
    assert len(batch[0]) == 0 and len(batch[1]) == 0  # noqa: PT018, S101

    # Delegate based on dimensionality
    if lhs.ndim == 1 and rhs.ndim == 1:
        return _dot_general_1d_1d(
            lhs,
            rhs,
            dimension_numbers=dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
            **kw,
        )
    if lhs.ndim == 2 and rhs.ndim == 2:
        return _dot_general_2d_2d(
            lhs,
            rhs,
            dimension_numbers=dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
            **kw,
        )
    if lhs.ndim == 2 and rhs.ndim == 1:
        return _dot_general_2d_1d(
            lhs,
            rhs,
            dimension_numbers=dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
            **kw,
        )
    msg = f"Unsupported dimensionality: lhs.ndim={lhs.ndim}, rhs.ndim={rhs.ndim}"
    raise NotImplementedError(msg)


@quax.register(lax.dot_general_p)
def dot_general_qm_arr(
    lhs: QMatrix,
    rhs: jax.Array,
    /,
    *,
    dimension_numbers: lax.DotDimensionNumbers,
    precision: Any = None,
    preferred_element_type: Any = None,
    **kw: Any,
) -> "QMatrix | u.Q":
    """Dot product of a :class:`QMatrix` with a plain JAX array.

    The plain array is treated as dimensionless.  Delegates to
    :func:`dot_general_qm_qm` after wrapping ``rhs`` in a dimensionless
    :class:`QMatrix`.

    >>> import jax.numpy as jnp
    >>> import quaxed.numpy as qnp
    >>> from coordinax.internal import QMatrix, UnitsMatrix

    2D metric x 1D plain vector:

    >>> g = QMatrix(
    ...     jnp.array([[2.0, 0.0], [0.0, 3.0]]),
    ...     unit=UnitsMatrix((("m2", "m2"), ("m2", "m2"))),
    ... )
    >>> v = jnp.array([1.0, 1.0])
    >>> w = qnp.matmul(g, v)
    >>> w.unit.to_string()
    '(m2, m2)'
    >>> w.value
    Array([2., 3.], dtype=float64)

    """
    if rhs.ndim == 1:
        n = rhs.shape[0]
        rhs_qm = QMatrix(rhs, unit=UnitsMatrix(tuple(_DMLS for _ in range(n))))
    else:
        nr, nc = rhs.shape[-2], rhs.shape[-1]
        rhs_qm = QMatrix(
            rhs,
            unit=UnitsMatrix(tuple(tuple(_DMLS for _ in range(nc)) for _ in range(nr))),
        )
    return dot_general_qm_qm(
        lhs,
        rhs_qm,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
        **kw,
    )


@quax.register(lax.dot_general_p)
def dot_general_qm_qty(
    lhs: QMatrix,
    rhs: u.AbstractQuantity,
    /,
    *,
    dimension_numbers: lax.DotDimensionNumbers,
    precision: Any = None,
    preferred_element_type: Any = None,
    **kw: Any,
) -> "QMatrix | u.Q":
    """Dot product of a :class:`QMatrix` with a :class:`~unxt.AbstractQuantity`.

    The Quantity carries a single scalar unit that applies uniformly to all
    elements.  The ``rhs`` is wrapped as a uniform-unit
    :class:`QMatrix` and delegated to :func:`dot_general_qm_qm`.

    Note that :class:`QMatrix` is itself a subtype of
    :class:`~unxt.AbstractQuantity`, so :func:`dot_general_qm_qm` takes
    precedence when both sides are :class:`QMatrix`.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import quaxed.numpy as qnp
    >>> from coordinax.internal import QMatrix, UnitsMatrix

    2D metric with units @ uniform-unit Quantity vector:

    >>> g = QMatrix(
    ...     jnp.array([[2.0, 0.0], [0.0, 3.0]]),
    ...     unit=UnitsMatrix((("m2", "m2"), ("m2", "m2"))),
    ... )
    >>> v = u.Q(jnp.array([1.0, 1.0]), "m")
    >>> w = qnp.matmul(g, v)
    >>> w.unit.to_string()
    '(m3, m3)'
    >>> w.value
    Array([2., 3.], dtype=float64)

    """
    rhs_unit = u.unit_of(rhs)
    rhs_val = cast("Array", u.ustrip(AllowValue, rhs_unit, rhs))
    if rhs_val.ndim == 1:
        n = rhs_val.shape[0]
        rhs_qm = QMatrix(rhs_val, unit=UnitsMatrix(tuple(rhs_unit for _ in range(n))))
    else:
        nr, nc = rhs_val.shape[-2], rhs_val.shape[-1]
        rhs_qm = QMatrix(
            rhs_val,
            unit=UnitsMatrix(
                tuple(tuple(rhs_unit for _ in range(nc)) for _ in range(nr))
            ),
        )
    return dot_general_qm_qm(
        lhs,
        rhs_qm,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
        **kw,
    )


@quax.register(lax.dot_general_p)
def dot_general_arr_qm(
    lhs: jax.Array,
    rhs: QMatrix,
    /,
    *,
    dimension_numbers: lax.DotDimensionNumbers,
    precision: Any = None,
    preferred_element_type: Any = None,
    **kw: Any,
) -> "QMatrix | u.Q":
    """Dot product of a plain JAX array with a :class:`QMatrix`.

    The plain array is treated as dimensionless.  Delegates to
    :func:`dot_general_qm_qm` after wrapping ``lhs`` in a dimensionless
    :class:`QMatrix`.

    >>> import jax.numpy as jnp
    >>> import quaxed.numpy as qnp
    >>> from coordinax.internal import QMatrix

    Dimensionless identity @ QMatrix vector:

    >>> A = jnp.eye(2, dtype=jnp.float64)
    >>> v = QMatrix(jnp.array([2.0, 3.0]), unit=("m / s", "m / s"))
    >>> w = qnp.matmul(A, v)
    >>> w.unit.to_string()
    '(m / s, m / s)'
    >>> w.value
    Array([2., 3.], dtype=float64)

    """
    if lhs.ndim == 1:
        n = lhs.shape[0]
        lhs_qm = QMatrix(lhs, unit=UnitsMatrix(tuple(_DMLS for _ in range(n))))
    else:
        nr, nc = lhs.shape[-2], lhs.shape[-1]
        lhs_qm = QMatrix(
            lhs,
            unit=UnitsMatrix(tuple(tuple(_DMLS for _ in range(nc)) for _ in range(nr))),
        )
    return dot_general_qm_qm(
        lhs_qm,
        rhs,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
        **kw,
    )


vec_uconvert_value = np.vectorize(u.uconvert_value)


def _dot_general_1d_1d(
    lhs: QMatrix,
    rhs: QMatrix,
    /,
    *,
    dimension_numbers: lax.DotDimensionNumbers,
    precision: Any = None,
    preferred_element_type: Any = None,
    **kw: Any,
) -> u.Q:
    """Vector dot product: (N,) @ (N,) → scalar.

    Result = Σ_i  lhs[i] * rhs[i]

    All terms must be unit-compatible. We convert to the unit of the first term.
    """
    n = lhs.shape[-1]
    assert n == rhs.shape[-1]  # noqa: S101

    # Reference unit: lhs.unit[0] * rhs.unit[0]
    ref_unit = lhs.unit[0] * rhs.unit[0]

    # Compute scale factors
    scales = jnp.array(
        [u.uconvert_value(ref_unit, lhs.unit[i] * rhs.unit[i], 1.0) for i in range(n)]
    )

    # Compute dot product with rescaling
    result_value = jnp.sum(scales * lhs.value * rhs.value, axis=-1)

    return u.Q(result_value, ref_unit)


def _dot_general_2d_1d(
    lhs: QMatrix,
    rhs: QMatrix,
    /,
    *,
    dimension_numbers: lax.DotDimensionNumbers,
    precision: Any = None,
    preferred_element_type: Any = None,
    **kw: Any,
) -> QMatrix:
    """Matrix-vector multiply: (N, K) @ (K,) → (N,).

    For ``w = A @ v`` where ``A`` is ``(N, K)`` and ``v`` is ``(K,)``:

    ``w[i] = Σ_j  A[i, j] * v[j]``

    Each product ``A[i,j] * v[j]`` has unit ``A.unit[i][j] * v.unit[j]``.  All
    ``K`` terms in the sum for output row ``i`` must be unit-compatible.  We
    convert every term to the unit of the *first* term (``j = 0``) for each
    output row ``i``: ``ref[i] = A.unit[i][0] * v.unit[0]``.

    >>> import jax.numpy as jnp
    >>> import quaxed.numpy as qnp
    >>> import unxt as u
    >>> from coordinax.internal import QMatrix

    Identity matrix times a vector:

    >>> A = QMatrix(jnp.eye(3, dtype=jnp.float64),
    ...                    unit=(("", "", ""), ("", "", ""), ("", "", "")))
    >>> v = QMatrix(jnp.array([1.0, 2.0, 3.0]), unit=("m", "m", "m"))
    >>> w = qnp.matmul(A, v)
    >>> w.value
    Array([1., 2., 3.], dtype=float64)

    Mixed units on contraction axis (km column converted to m):

    >>> A2 = QMatrix(jnp.array([[1.0, 2.0], [3.0, 4.0]]),
    ...                     unit=(("m", "km"), ("m", "km")))
    >>> v2 = QMatrix(jnp.array([1.0, 1.0]), unit=("s", "s"))
    >>> w2 = qnp.matmul(A2, v2)
    >>> w2.value
    Array([2001., 4003.], dtype=float64)
    >>> w2.unit.to_string()
    '(m s, m s)'

    """
    assert rhs.shape[-1] == lhs.shape[-1]  # noqa: S101

    # 1) Output units: ref[i] = lhs.unit[i][0] * rhs.unit[0]
    out_unit = UnitsMatrix(np.multiply(lhs.unit._units[:, 0], rhs.unit._units[0]))

    # 2) Precompute scale factors: scale[i, j] converts
    #    lhs.unit[i][j]*rhs.unit[j] → ref[i]
    scale_2d = jnp.array(
        vec_uconvert_value(
            out_unit._units[:, None],  # (N, 1) — broadcast over K
            np.multiply(lhs.unit._units, rhs.unit._units[None, :]),  # (N, K)
            1.0,
        )
    )

    # 3) Vectorised contraction:
    #    w[..., i] = Σ_j  scale[i, j] * A[..., i, j] * v[..., j]
    accum = jnp.einsum("ij,...ij,...j->...i", scale_2d, lhs.value, rhs.value)

    return QMatrix(value=accum, unit=out_unit)


vec_uconvert_value = np.vectorize(u.uconvert_value)


def _dot_general_2d_2d(
    lhs: QMatrix,
    rhs: QMatrix,
    /,
    *,
    dimension_numbers: lax.DotDimensionNumbers,
    precision: Any = None,
    preferred_element_type: Any = None,
    **kw: Any,
) -> QMatrix:
    """Matrix multiply: (N, K) @ (K, M) → (N, M).

    For ``C = A @ B`` where ``A`` is ``(N, K)`` and ``B`` is ``(K, M)``:

    ``C[i, k] = Σ_j  A[i, j] * B[j, k]``

    Each product ``A[i,j] * B[j,k]`` has unit ``A.unit[i][j] * B.unit[j][k]``.
    All ``K`` terms in the sum **must** be unit-compatible.  We convert every
    term to the unit of the *first* term (``j = 0``) using `u.uconvert_value`,
    then sum with a plain matmul.

    The strategy:
    1. Pick a reference unit for each ``(i, k)`` output element:
       ``ref[i][k] = A.unit[i][0] * B.unit[0][k]``.
    2. For each contraction index ``j``, compute per-element conversion
       factors from ``A.unit[i][j] * B.unit[j][k]`` to ``ref[i][k]``.
       Because the products are *multiplicative* compositions, the
       conversion from ``u_A * u_B`` to ``ref`` is multiplicative even
       when the individual units are affine — the product of two
       absolute quantities is always absolute.
       So we can safely compute a scale factor:
       ``scale[i][j][k] = uconvert_value(ref[i][k], A.unit[i][j] * B.unit[j][k], 1.0)``
    3. Build the rescaled sum as:
       ``C_val[i, k] = Σ_j  scale[i][j][k] * A_val[i, j] * B_val[j, k]``
       Done via ``C_val = (A_val * S_ij) @ B_val`` per output column, or
       equivalently with a loop + accumulate.
    """
    # Check contraction axis
    assert lhs.shape[-1] == rhs.shape[-2]  # noqa: S101

    # 1) Compute output units: ref[i][k] = lhs.unit[i][0] * rhs.unit[0][k]
    out_unit = np.multiply(lhs.unit._units[:, 0:1], rhs.unit._units[0:1, :])

    # 2) Precompute all scale factors as a (N, K, M) constant array.
    #    scale[i, j, k] converts lhs.unit[i][j]*rhs.unit[j][k] → out_unit[i][k].
    #
    #    CORRECTNESS NOTE — why a multiplicative scale factor is exact:
    #    Affine units (°C, °F) are the only units where a bare
    #    multiplicative scale would be wrong (they have an additive
    #    offset).  But astropy rejects product conversions involving
    #    affine units — e.g. ``(deg_C * s).to(deg_F * s)`` raises
    #    ``UnitConversionError``.  Every product unit that astropy
    #    *does* accept (including logarithmic units like dex, mag) is
    #    a plain ``CompositeUnit`` whose conversion is purely
    #    multiplicative.  So ``uconvert_value(to, from, 1.0)`` yields
    #    an exact scale factor for all valid product units.
    #
    #    The tests in ``TestAffineProductUnitsRejected`` assert that
    #    astropy keeps rejecting affine product conversions.  If that
    #    ever changes, those tests will fail, alerting us that this
    #    assumption needs revisiting.
    scale_3d = jnp.array(
        vec_uconvert_value(
            out_unit[:, None, :],  # (N, 1, M)
            np.multiply(lhs.unit._units[:, :, None], rhs.unit._units[None, :, :]),
            1.0,  # ꜛ (N, K, M)
        )
    )

    # 3) Vectorised contraction — no Python loop, no accumulator.
    #    C[..., i, k] = Σ_j  scale[i, j, k] * A[..., i, j] * B[..., j, k]
    accum = jnp.sum(  # (N, K, M) * (..., N, K, 1) * (..., 1, K, M)
        scale_3d * lhs.value[..., :, :, None] * rhs.value[..., None, :, :], axis=-2
    )

    return QMatrix(value=accum, unit=out_unit)


@quax.register(lax.sub_p)
def sub_qm_qm(x: QMatrix, y: QMatrix, /) -> QMatrix:
    """Element-wise subtraction of two `QMatrix` objects.

    The result adopts the units of *x*.  Each element is converted from
    ``y.unit`` → ``x.unit`` before the numeric subtract.

    Works for both 1D (vector) and 2D (matrix) cases.

    >>> import jax.numpy as jnp
    >>> import quaxed.numpy as qnp
    >>> import unxt as u
    >>> from coordinax.internal import QMatrix

    2D case:

    >>> a = QMatrix(jnp.ones((2, 2)), unit=(("m", "s"), ("kg", "rad")))
    >>> b = QMatrix(
    ...     value=jnp.ones((2, 2)),
    ...     unit=(("km", u.unit("ms")), (u.unit("g"), u.unit("deg"))))

    >>> result = qnp.subtract(a, b)
    >>> result.unit.to_string()
    '((m, s), (kg, rad))'

    >>> result.value
    Array([[-9.99000000e+02,  9.99000000e-01],
           [ 9.99000000e-01,  9.82546707e-01]], dtype=float64)

    1D case:

    >>> a1d = QMatrix(value=jnp.ones(3),
    ...                      unit=("m", "s", "kg"))
    >>> b1d = QMatrix(value=jnp.ones(3),
    ...                      unit=("km", u.unit("ms"), u.unit("g")))

    >>> result1d = qnp.subtract(a1d, b1d)
    >>> result1d.unit.to_string()
    '(m, s, kg)'

    >>> result1d.value
    Array([-999.   ,    0.999,    0.999], dtype=float64)

    """
    y_converted = _convert_value(y.value, y.unit, x.unit)
    return QMatrix(value=lax.sub(x.value, y_converted), unit=x.unit)


@quax.register(lax.transpose_p)
def transpose_qm(x: QMatrix, /, *, permutation: tuple[int, ...]) -> QMatrix:
    """Transpose a ``QMatrix``, swapping only the last two (matrix) axes.

    Leading batch dimensions must be preserved unchanged.  Only permutations
    that swap the last two axes while keeping all batch axes in place are
    supported, because the unit structure is purely 2-D and cannot represent
    arbitrary axis re-orderings.

    >>> import jax.numpy as jnp
    >>> import quaxed.numpy as qnp
    >>> from coordinax.internal import QMatrix

    2-D (no batch):

    >>> a = QMatrix(jnp.array([[1.0, 2.0], [3.0, 4.0]]),
    ...                    unit=(("m", "s"), ("kg", "rad")))
    >>> aT = qnp.matrix_transpose(a)
    >>> aT.value
    Array([[1., 3.],
           [2., 4.]], dtype=float64)
    >>> aT.unit.to_string()
    '((m, kg), (s, rad))'

    Batched ``(B, N, M)`` — batch axis is preserved:

    >>> import jax
    >>> b = QMatrix(jnp.ones((3, 2, 2)),
    ...                    unit=(("m", "s"), ("kg", "rad")))
    >>> bT = qnp.matrix_transpose(b)
    >>> bT.shape
    (3, 2, 2)

    """
    ndim_val = len(permutation)  # full ndim of the value array (includes batch dims)
    if ndim_val < 2:
        msg = f"transpose_qm requires ndim >= 2, got ndim={ndim_val}"
        raise NotImplementedError(msg)
    # Validate: batch axes must be unchanged, last two must be swapped.
    expected = (*range(ndim_val - 2), ndim_val - 1, ndim_val - 2)
    if tuple(permutation) != expected:
        msg = (
            f"transpose_qm only supports matrix transpose of the last two axes "
            f"(expected permutation {expected}), got {tuple(permutation)}"
        )
        raise NotImplementedError(msg)
    transposed_value = lax.transpose(x.value, permutation)
    return QMatrix(value=transposed_value, unit=x.unit.T)


def _jit_fallback_uniform_unit(units: UnitsMatrix, out_size: int) -> UnitsMatrix:
    """Return a 1-D ``UnitsMatrix`` of length *out_size* if all units are equal.

    Used as a JIT-mode fallback inside ``gather_qm`` when the concrete gather
    indices are not available.  Raises ``ValueError`` for heterogeneous inputs.
    """
    all_units = jtu.tree_leaves(units.to_tuple())
    first = all_units[0]
    if any(u_i != first for u_i in all_units[1:]):
        msg = (
            "QMatrix gather (e.g. jnp.diag) under jit requires all units "
            "to be equal when indices cannot be concretized. "
            "Call eagerly (outside jit) for heterogeneous-unit QMatrix."
        )
        raise ValueError(msg)
    return UnitsMatrix(np.full((out_size,), first, dtype=object))


@quax.register(lax.gather_p)
def gather_qm(
    x: QMatrix,
    start_indices: jax.Array,
    /,
    *,
    dimension_numbers: lax.GatherDimensionNumbers,
    slice_sizes: tuple[int, ...],
    indices_are_sorted: bool = False,
    mode: Any = None,
    fill_value: Any = None,
    unique_indices: bool = False,
    **kwargs: Any,
) -> QMatrix:
    """Handle element-selection gathers (e.g. ``jnp.diag``) for ``QMatrix``.

    Supports only *element-selection* gathers where every input dimension is
    collapsed (``offset_dims == ()`` and all ``slice_sizes == 1``).  This
    covers ``jnp.diag``, ``jnp.diagonal``, and integer-array fancy indexing on
    ``QMatrix`` objects.

    Unit extraction:

    ``QMatrix.unit`` is declared ``static=True`` and is therefore always
    a concrete Python object, even inside ``jax.jit``.  The *indices*, however,
    are traced under JIT and cannot be read concretely.  Because JAX's
    ``jnp.diag`` uses ``platform_dependent`` internally, quax always traces
    both branches via ``make_jaxpr``, so the JIT fallback path is taken for
    unit resolution.  Consequently, all units in the input must be equal;
    heterogeneous-unit inputs raise ``ValueError``.

    >>> import jax.numpy as jnp
    >>> from coordinax.internal import QMatrix

    Diagonal of a 3x3 dimensionless matrix:

    >>> A = QMatrix(jnp.diag(jnp.array([1.0, 4.0, 9.0])),
    ...                    unit=(("", "", ""), ("", "", ""), ("", "", "")))
    >>> d = A.diag()
    >>> d.unit.shape
    (3,)
    >>> d.unit.ndim
    1
    >>> d.value
    Array([1., 4., 9.], dtype=float64)

    ```{note}
    ``jnp.diag`` uses JAX's ``platform_dependent`` internally, which causes
    quax to trace both branches via ``make_jaxpr`` even in eager mode.  This
    means the JIT fallback path is always taken for the unit computation, so
    **heterogeneous-unit matrices are not supported** with ``qnp.diag``.
    All units in the input must be equal; otherwise a ``ValueError`` is raised.
    ```

    """
    result_value = lax.gather(
        x.value,
        start_indices,
        dimension_numbers,
        slice_sizes,
        indices_are_sorted=indices_are_sorted,
        mode=mode,
        fill_value=fill_value,
        unique_indices=unique_indices,
    )

    # Only element-selection gathers are supported: all input dimensions must
    # be collapsed and every slice_size must be 1.
    n_input_dims = x.value.ndim
    normalized_collapsed = {
        d % n_input_dims for d in dimension_numbers.collapsed_slice_dims
    }
    is_element_selection = (
        dimension_numbers.offset_dims == ()
        and normalized_collapsed == set(range(n_input_dims))
        and all(s == 1 for s in slice_sizes)
    )
    if not is_element_selection:
        msg = (
            "QMatrix: only element-selection gathers (all input dims "
            "collapsed, all slice_sizes == 1) are supported. "
            f"Got offset_dims={dimension_numbers.offset_dims}, "
            f"collapsed_slice_dims={dimension_numbers.collapsed_slice_dims}, "
            f"slice_sizes={slice_sizes}."
        )
        raise NotImplementedError(msg)

    # Number of output elements — start_indices.shape is always concrete in JAX.
    out_size = start_indices.shape[0]

    if isinstance(start_indices, jax.core.Tracer):
        # JIT path: indices are traced — fall back to uniform-unit check.
        out_unit = _jit_fallback_uniform_unit(x.unit, out_size)
    else:
        # Eager path: indices are concrete — look up units directly.
        idx_np = np.asarray(start_indices)
        if x.unit.ndim == 1:
            out_unit = UnitsMatrix(x.unit._units[idx_np[:, 0]])
        else:  # x.unit.ndim == 2
            out_unit = UnitsMatrix(x.unit._units[idx_np[:, 0], idx_np[:, 1]])

    return QMatrix(value=result_value, unit=out_unit)


@quax.register(lax.reduce_sum_p)
def reduce_sum_p_qm(operand: QMatrix, /, *, axes: Any, **kwargs: Any) -> QMatrix:
    """Handle ``lax.reduce_sum`` for ``QMatrix``.

    ``jnp.diag`` on a square 2-D matrix uses ``platform_dependent`` which traces
    *both* the default (gather-based) and Mosaic implementation.  The Mosaic
    path computes ``reduce(mul(eye, A), axis=0)`` — JAX's JIT optimises
    ``lax.reduce(x, 0, lax.add, (0,))`` to the simpler ``reduce_sum_p``
    primitive.  This handler ensures the output carries the correct 1-D unit
    structure so that both branches produce the *same* pytree — required by
    ``platform_dependent`` / ``lax.switch``.

    Unit reduction rule:

    When reducing a 2-D ``QMatrix`` along ``axes=(0,)`` (rows): the
    output unit for column *j* is taken from ``operand.unit[0, j]`` (the first
    row).  All elements being summed along a column must be unit-compatible for
    the sum to be physically meaningful.

    Analogously for ``axes=(1,)`` (column reduction), the output unit for row
    *i* is ``operand.unit[i, 0]``.

    >>> import jax.numpy as jnp
    >>> from coordinax.internal import QMatrix

    ``QMatrix.diag()`` on a 3x3 uniform-unit matrix:

    >>> A = QMatrix(jnp.diag(jnp.array([1.0, 4.0, 9.0])),
    ...                    unit=(("m", "m", "m"), ("m", "m", "m"), ("m", "m", "m")))
    >>> d = A.diag()
    >>> d.unit.shape
    (3,)
    >>> d.unit.ndim
    1

    """
    result_value = lax.reduce_sum_p.bind(operand.value, axes=axes, **kwargs)

    # Reduce the unit structure by dropping the summed axes.
    axset = set(axes)
    if operand.ndim == 2:
        if axset == {0}:
            # Row reduction → 1-D output; unit = first row's units.
            out_unit = UnitsMatrix(operand.unit._units[0])
        elif axset == {1}:
            # Column reduction → 1-D output; unit = first column's units.
            out_unit = UnitsMatrix(operand.unit._units[:, 0])
        else:
            msg = f"reduce_sum_p_qm: unsupported axes={axes} for 2-D QMatrix."
            raise NotImplementedError(msg)
    else:
        msg = (
            f"reduce_sum_p_qm: only 2-D QMatrix is supported, got ndim={operand.ndim}."
        )
        raise NotImplementedError(msg)

    return QMatrix(value=result_value, unit=out_unit)


##############################################################################
# Custom det_p JAX primitive
#
# JAX has no built-in `det` primitive.  `jnp.linalg.det` decomposes into
# arithmetic (2×2 / 3×3) or `lu_p` + log/exp (larger matrices).  We define
# `det_p` here so that Quax can intercept determinant calls on `QMatrix`
# objects.
#
# Supported transforms:
#   - JIT via MLIR lowering that delegates to `jnp.linalg.det`
#   - Forward-mode autodiff (JVP): d(det A)(dA) = det(A) · tr(A⁻¹ dA)
#   - Reverse-mode autodiff (VJP): derived automatically from JVP via
#     transposition — no explicit transpose rule needed because the JVP tangent
#     only uses existing primitives (linalg.solve, trace, mul) that already
#     carry transpose rules.
#   - Batching (vmap): move the batch axis to the front and call det_p; the MLIR
#     lowering handles any (*batch, n, n) shape natively.

det_p = jexc.Primitive("det")
det_p.multiple_results = False


def det(x: Array, /) -> Array:
    """Compute the determinant of a square matrix via the ``det_p`` primitive.

    Delegates to ``det_p``, a custom JAX primitive that supports JIT,
    forward and reverse differentiation, and batching (vmap).

    For plain arrays the result is a bare :class:`~jaxtyping.Array`.
    For :class:`~coordinax.internal.QMatrix` inputs the Quax
    dispatch intercepts the call (see ``_det_p_QMatrix``) and
    returns a :class:`~unxt.AbstractQuantity`.

    Parameters
    ----------
    x : Array, shape ``(*batch, n, n)``
        Square matrix or batch of square matrices.

    Returns
    -------
    Array, shape ``(*batch,)``
        Determinant of each matrix.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from coordinax._src.internal.quantity_matrix import det

    Plain 2x2 diagonal matrix:

    >>> det(jnp.array([[2.0, 0.0], [0.0, 3.0]]))
    Array(6., dtype=float64)

    Under JIT:

    >>> import jax
    >>> jax.jit(det)(jnp.array([[2.0, 0.0], [0.0, 3.0]]))
    Array(6., dtype=float64)

    Gradient (via reverse-mode autodiff):

    >>> jax.grad(det)(jnp.array([[2.0, 0.0], [0.0, 3.0]]))
    Array([[3., 0.],
           [0., 2.]], dtype=float64)

    Batched (vmap):

    >>> A = jnp.stack([jnp.diag(jnp.array([2.0, 3.0])),
    ...                jnp.diag(jnp.array([4.0, 5.0]))])
    >>> jax.vmap(det)(A)
    Array([ 6., 20.], dtype=float64)

    """
    return det_p.bind(x)


# ── 1. Primal evaluation rule (eager / concrete values) ──────────────────


def _det_impl(x: Array, /) -> Array:
    return jnp.linalg.det(x)


det_p.def_impl(_det_impl)


# ── 2. Abstract evaluation rule (shape / dtype inference for JIT) ────────


def _det_abstract_eval(x: "jax.core.ShapedArray", /) -> "jax.core.ShapedArray":
    if x.ndim < 2:
        msg = f"det_p requires at least 2-D input, got ndim={x.ndim}"
        raise ValueError(msg)
    if x.shape[-1] != x.shape[-2]:
        msg = (
            f"det_p requires a square matrix "
            f"(shape[-2] == shape[-1]), got shape={x.shape}"
        )
        raise ValueError(msg)
    # (*batch, n, n) → (*batch,)
    return x.update(shape=x.shape[:-2])


det_p.def_abstract_eval(_det_abstract_eval)


# ── 3. MLIR / XLA lowering (JIT compilation) ─────────────────────────────

jax_mlir.register_lowering(
    det_p,
    jax_mlir.lower_fun(_det_impl, multiple_results=False),
)


# ── 4. Forward-mode differentiation (JVP) ────────────────────────────────
#
# Jacobi's formula: d(det A)(dA) = det(A) · tr(A⁻¹ dA)
#
# We use `jnp.linalg.solve(A, dA)` to compute A⁻¹ dA without explicitly
# forming the inverse — more numerically stable.  The tangent uses only
# existing JAX primitives (solve, trace, scalar mul), so JAX derives the
# reverse-mode (VJP) rule automatically via transposition; no explicit
# `ad.primitive_transposes` registration is needed.


def _det_jvp(primals: tuple, tangents: tuple) -> tuple:
    (x,) = primals
    (dx,) = tangents
    primal_out = det_p.bind(x)
    if type(dx) is jax_ad.Zero:
        tangent_out = lax.full_like(primal_out, 0.0)
    else:
        # tr(A⁻¹ dA) via solve — avoids explicit matrix inversion
        tangent_out = primal_out * jnp.trace(jnp.linalg.solve(x, dx))
    return primal_out, tangent_out


jax_ad.primitive_jvps[det_p] = _det_jvp


# ── 5. Batching rule (vmap) ───────────────────────────────────────────────
#
# `jnp.linalg.det` (used in the MLIR lowering) already operates correctly
# on batched (*batch, n, n) arrays.  The batching rule moves the vmap batch
# axis to the front (just before the matrix dims) and calls det_p; the
# result carries the batch axis at position 0.


def _det_batch(args: tuple, batch_axes: tuple) -> tuple:
    (x,) = args
    (ax,) = batch_axes
    x = jnp.moveaxis(x, ax, 0)
    return det_p.bind(x), 0


jax_batching.primitive_batchers[det_p] = _det_batch


# ── 6. Quax dispatch for QMatrix ──────────────────────────────────


@quax.register(det_p)
def _det_p_QMatrix(x: QMatrix, /) -> "u.AbstractQuantity":
    """Compute the determinant of a 2-D :class:`~coordinax.internal.QMatrix`.

    The numeric value is computed via ``det_p.bind(x.value)``.  The unit
    is the product of the main-diagonal units — valid for diagonal metrics
    and any matrix where all cofactor products share the same physical
    dimension (e.g. coordinate metric tensors).

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import quax
    >>> import unxt as u
    >>> from coordinax.internal import QMatrix
    >>> from coordinax._src.internal.quantity_matrix import det

    >>> A = QMatrix(
    ...     jnp.array([[2.0, 0.0], [0.0, 3.0]]),
    ...     unit=(("m2", "m2"), ("m2", "m2")),
    ... )
    >>> quax.quaxify(det)(A)
    Q(6., 'm4')

    """
    if x.ndim != 2:
        msg = f"det_p QMatrix dispatch requires a 2-D unit structure, got ndim={x.ndim}"
        raise ValueError(msg)
    det_val = det_p.bind(x.value)
    n = x.unit.shape[0]
    det_unit = ft.reduce(operator.mul, (x.unit[i, i] for i in range(n)))
    return u.Q(det_val, det_unit)


##############################################################################
# Custom inv_p JAX primitive
#
# JAX has no standalone `inv` primitive.  `jnp.linalg.inv` decomposes into
# `lu_p` + `triangular_solve_p`, and Quax has no dispatch for those on
# `QMatrix` (which raises on materialise).  We define `inv_p` to give
# Quax a single interception point with full unit tracking.
#
# Supported transforms:
#   - JIT via MLIR lowering that delegates to `jnp.linalg.inv`
#   - Forward-mode autodiff (JVP): d(A^{-1}) = -A^{-1} dA A^{-1}
#   - Reverse-mode autodiff (VJP): derived automatically from JVP
#   - Batching (vmap): move batch axis to front, call inv_p

inv_p = jexc.Primitive("inv")
inv_p.multiple_results = False


def inv(x: Array, /) -> Array:
    """Compute the matrix inverse of a square matrix via the ``inv_p`` primitive.

    Delegates to ``inv_p``, a custom JAX primitive that supports JIT,
    forward and reverse differentiation, and batching (vmap).

    For plain arrays the result is a bare :class:`~jaxtyping.Array`.
    For :class:`~coordinax.internal.QMatrix` inputs the Quax
    dispatch intercepts the call (see ``_inv_p_QMatrix``) and
    returns a :class:`~coordinax.internal.QMatrix` with
    reciprocal units.

    Parameters
    ----------
    x : Array, shape ``(*batch, n, n)``
        Square matrix or batch of square matrices.

    Returns
    -------
    Array, shape ``(*batch, n, n)``
        Matrix inverse of each square matrix.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from coordinax._src.internal.quantity_matrix import inv

    Plain 2x2 diagonal matrix:

    >>> inv(jnp.array([[2.0, 0.0], [0.0, 4.0]]))
    Array([[0.5 , 0.  ],
           [0.  , 0.25]], dtype=float64)

    Under JIT:

    >>> import jax
    >>> jax.jit(inv)(jnp.array([[2.0, 0.0], [0.0, 4.0]]))
    Array([[0.5 , 0.  ],
           [0.  , 0.25]], dtype=float64)

    Gradient (via reverse-mode autodiff) — returns a rank-4 Jacobian:

    >>> jac = jax.jacobian(inv)(jnp.array([[2.0, 0.0], [0.0, 4.0]]))
    >>> jac.shape
    (2, 2, 2, 2)

    Batched (vmap):

    >>> A = jnp.stack([jnp.diag(jnp.array([2.0, 4.0])),
    ...                jnp.diag(jnp.array([1.0, 2.0]))])
    >>> jax.vmap(inv)(A)
    Array([[[0.5 , 0.  ],
            [0.  , 0.25]],
    <BLANKLINE>
           [[1.  , 0.  ],
            [0.  , 0.5 ]]], dtype=float64)

    """
    return inv_p.bind(x)


# ── 1. Primal evaluation rule ─────────────────────────────────────────────


def _inv_impl(x: Array, /) -> Array:
    return jnp.linalg.inv(x)


inv_p.def_impl(_inv_impl)


# ── 2. Abstract evaluation rule ───────────────────────────────────────────


def _inv_abstract_eval(x: "jax.core.ShapedArray", /) -> "jax.core.ShapedArray":
    if x.ndim < 2:
        msg = f"inv_p requires at least 2-D input, got ndim={x.ndim}"
        raise ValueError(msg)
    if x.shape[-1] != x.shape[-2]:
        msg = (
            f"inv_p requires a square matrix "
            f"(shape[-2] == shape[-1]), got shape={x.shape}"
        )
        raise ValueError(msg)
    return x.update(shape=x.shape)  # same shape as input


inv_p.def_abstract_eval(_inv_abstract_eval)


# ── 3. MLIR / XLA lowering ────────────────────────────────────────────────

jax_mlir.register_lowering(
    inv_p,
    jax_mlir.lower_fun(_inv_impl, multiple_results=False),
)


# ── 4. Forward-mode differentiation (JVP) ────────────────────────────────
#
# d(A^{-1})(dA) = -A^{-1} dA A^{-1}


def _inv_jvp(primals: tuple, tangents: tuple) -> tuple[Array, Array]:
    (x,) = primals
    (dx,) = tangents
    primal_out = inv_p.bind(x)
    if type(dx) is jax_ad.Zero:
        tangent_out = jax_ad.Zero.from_primal_value(primal_out)  # ty: ignore[unresolved-attribute]
    else:
        tangent_out = -primal_out @ dx @ primal_out
    return primal_out, tangent_out


jax_ad.primitive_jvps[inv_p] = _inv_jvp


# ── 5. Batching rule (vmap) ───────────────────────────────────────────────


def _inv_batch(args: tuple, batch_axes: tuple) -> tuple:
    (x,) = args
    (ax,) = batch_axes
    x = jnp.moveaxis(x, ax, 0)
    return inv_p.bind(x), 0


jax_batching.primitive_batchers[inv_p] = _inv_batch


# ── 6. Quax dispatch for QMatrix ──────────────────────────────────


@quax.register(inv_p)
def _inv_p_QMatrix(x: QMatrix, /) -> QMatrix:
    """Compute the inverse of a 2-D :class:`~coordinax.internal.QMatrix`.

    The numeric value is computed via ``inv_p.bind(x.value)``.
    Units are assumed uniform (all entries share the same physical unit,
    as is the case for metrics produced by the Cartesian-Jacobian pullback);
    the inverse carries the reciprocal unit throughout.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import quax
    >>> import unxt as u
    >>> from coordinax.internal import QMatrix, UnitsMatrix
    >>> from coordinax._src.internal.quantity_matrix import inv

    >>> A = QMatrix(
    ...     jnp.array([[4.0, 0.0], [0.0, 1.0]]),
    ...     unit=UnitsMatrix((('m2', 'm2'), ('m2', 'm2'))),
    ... )
    >>> quax.quaxify(inv)(A)
    QMatrix([[0.25, 0.  ],
                   [0.  , 1.  ]], '((1 / m2, 1 / m2), (1 / m2, 1 / m2))')

    """
    if x.ndim != 2:
        msg = f"inv_p QMatrix dispatch requires a 2-D unit structure, got ndim={x.ndim}"
        raise ValueError(msg)
    inv_val = inv_p.bind(x.value)
    return QMatrix(inv_val, unit=x.unit.inverse())
