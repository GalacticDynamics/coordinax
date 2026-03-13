"""QuantityMatrix — an N-D quantity whose every element can carry its own unit.

The numeric payload is a single JAX array of shape ``(..., *shape)`` where
``*shape`` represents the "real" dimensions (not batch dimensions).
Units are stored as a static nested tuple structure matching the shape
so that every element has its own physical unit.

For 1D: shape ``(..., N)`` with units ``(u0, u1, ..., uN-1)``
For 2D: shape ``(..., N, M)`` with units ``((u00, u01, ...), (u10, u11, ...), ...)``

Quax primitive dispatches (``add_p``, ``dot_general_p``) perform the
necessary per-element unit conversions via `unxt.uconvert_value` — which
correctly handles affine conversions (e.g. °F → °C), not just
multiplicative scale factors.
"""

__all__ = ("QuantityMatrix", "UnitsMatrix")

from collections.abc import Iterable
from jaxtyping import Array, Shaped
from typing import Any, ClassVar, NoReturn, Union, final

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import plum
import quax
import wadler_lindig as wl
from jax import lax

import unxt as u

# ── Helpers ──────────────────────────────────────────────────────────


def _normalize_units(obj: Any) -> tuple[Any, ...]:
    """Recursively normalize a nested structure into nested tuples of units."""
    if isinstance(obj, (u.AbstractUnit, str)):
        return u.unit(obj)
    # It's an iterable - recurse
    items = obj if isinstance(obj, tuple) else tuple(obj)

    if not items:
        return ()

    # Check if first element is a unit or string (leaf level)
    first = items[0]
    if isinstance(first, (u.AbstractUnit, str)):
        return tuple(u.unit(item) for item in items)

    # It's nested - recurse
    return tuple(_normalize_units(item) for item in items)


def _compute_shape(units: tuple[Any, ...]) -> tuple[int, ...]:
    """Compute the shape of a nested tuple structure."""
    if not units:
        return (0,)
    if isinstance(units[0], u.AbstractUnit):
        # 1D case
        return (len(units),)
    # Nested case - recurse
    first_shape = _compute_shape(units[0])
    return (len(units), *first_shape)


@final
class UnitsMatrix(tuple[Any, ...]):
    """Immutable N-D nested tuple of units with indexing and string methods.

    A thin ``tuple`` subclass that handles both 1-D and 2-D (and N-D) unit
    structures. Supports tuple indexing via ``__getitem__``.

    For 1D: ``UnitsMatrix((u.unit("m"), u.unit("s"), u.unit("kg")))``
    For 2D: ``UnitsMatrix(((u.unit("m"), u.unit("s")), (u.unit("kg"), u.unit("rad"))))``

    Examples
    --------
    >>> import unxt as u
    >>> from coordinax.internal._quantity_matrix import UnitsMatrix

    1D case:

    >>> units_1d = UnitsMatrix((u.unit("m"), u.unit("s"), u.unit("kg")))
    >>> units_1d.shape
    (3,)
    >>> units_1d[0]
    Unit("m")
    >>> units_1d.to_string()
    '(m, s, kg)'

    2D case:

    >>> units_2d = UnitsMatrix(((u.unit("m"), u.unit("s")),
    ...                         (u.unit("kg"), u.unit("rad"))))
    >>> units_2d.shape
    (2, 2)
    >>> units_2d[0, 1]
    Unit("s")
    >>> units_2d.to_string()
    '((m, s), (kg, rad))'

    """

    __slots__: ClassVar[tuple[()]] = ()

    def __new__(cls, iterable: Iterable[Any] = (), /) -> "UnitsMatrix":
        normalized = _normalize_units(iterable)
        return super().__new__(cls, normalized)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the N-D unit structure."""
        return _compute_shape(self)

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    def __getitem__(self, key: Any) -> Union[u.AbstractUnit, "UnitsMatrix"]:
        """Support both single indexing and tuple indexing.

        For 1D: units[i] returns a unit
        For 2D: units[i, j] returns a unit, units[i] returns a row
        """
        if isinstance(key, tuple):
            # Nested indexing
            result: Any = self
            for idx in key:
                result = tuple.__getitem__(result, idx)
            # If result is still a tuple of units, wrap it
            if (
                isinstance(result, tuple)
                and result
                and isinstance(result[0], u.AbstractUnit)
            ):
                return UnitsMatrix(result)
            return result

        # Single index
        result = super().__getitem__(key)
        # If result is a tuple of units, wrap it
        if (
            isinstance(result, tuple)
            and result
            and isinstance(result[0], u.AbstractUnit)
        ):
            return UnitsMatrix(result)
        return result

    # ── string protocol ──────────────────────────────────────────────

    def __pdoc__(self, **kwargs: Any) -> wl.AbstractDoc:
        """Return a Wadler-Lindig document for pretty-printing the units."""
        if not self:
            return wl.TextDoc("()")

        # For 1D, just format as a simple tuple
        if self.ndim == 1:
            return wl.bracketed(
                begin=wl.TextDoc("("),
                docs=[wl.TextDoc(u_elem.to_string()) for u_elem in self],
                sep=wl.comma,
                end=wl.TextDoc(")"),
                indent=1,
            )

        # For 2D, format each row as a bracketed list
        row_docs = [
            wl.bracketed(
                begin=wl.TextDoc("("),
                docs=[wl.TextDoc(u_cell.to_string()) for u_cell in row],
                sep=wl.comma,
                end=wl.TextDoc(")"),
                indent=2,
            )
            for row in self
        ]

        # Format the whole matrix as a bracketed list of rows
        return wl.bracketed(
            begin=wl.TextDoc("("),
            docs=row_docs,
            sep=wl.comma,
            end=wl.TextDoc(")"),
            indent=1,
        )

    def __str__(self) -> str:
        """Render units.

        Returns
        -------
        str
            A formatted representation of the units.

        Examples
        --------
        >>> import unxt as u
        >>> from coordinax.internal._quantity_matrix import UnitsMatrix

        >>> units_1d = UnitsMatrix((u.unit("m"), u.unit("s")))
        >>> print(units_1d)
        (m, s)

        >>> units_2d = UnitsMatrix(((u.unit("m"), u.unit("s")),
        ...                         (u.unit("kg"), u.unit("rad"))))
        >>> print(units_2d)
        ((m, s), (kg, rad))

        """
        return wl.pformat(self)

    def to_string(self, format_: str = "generic", /) -> str:  # noqa: ARG002
        """Render units as a concise nested-tuple string.

        Parameters
        ----------
        format_ : str, optional
            Ignored; accepted for compatibility with
            ``AbstractUnit.to_string(format)``.

        Returns
        -------
        str

        Examples
        --------
        >>> import unxt as u
        >>> from coordinax.internal._quantity_matrix import UnitsMatrix

        1D:
        >>> UnitsMatrix((u.unit("m"), u.unit("s"), u.unit("kg"))).to_string()
        '(m, s, kg)'

        >>> UnitsMatrix((u.unit("m"),)).to_string()
        '(m,)'

        2D:
        >>> UnitsMatrix(((u.unit("m"), u.unit("s")),)).to_string()
        '((m, s),)'

        >>> UnitsMatrix(((u.unit("m"),), (u.unit("s"),))).to_string()
        '((m,), (s,))'

        """
        if self.ndim == 1:
            # 1D case
            inner = ", ".join(ui.to_string() for ui in self)
            if len(self) == 1:
                inner += ","
            return f"({inner})"

        # 2D+ case
        def _row_str(row: tuple[Any, ...]) -> str:
            if isinstance(row[0], u.AbstractUnit):
                inner = ", ".join(ui.to_string() for ui in row)
                if len(row) == 1:
                    inner += ","
                return f"({inner})"
            # Nested deeper - recurse
            return UnitsMatrix(row).to_string()

        rows = ", ".join(_row_str(row) for row in self)
        if len(self) == 1:
            rows += ","
        return f"({rows})"


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


# ── Class ────────────────────────────────────────────────────────────


class QuantityMatrix(u.AbstractQuantity):
    """N-D quantity matrix/vector where every element carries its own unit.

    Supports both 1-D (vector) and 2-D (matrix) cases, determined by the
    shape of the units structure.

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
    >>> from coordinax.internal import QuantityMatrix

    1D case (vector):

    >>> qv = QuantityMatrix(
    ...     value=jnp.array([1.0, 2.0, 3.0]),
    ...     unit=(u.unit("m"), u.unit("s"), u.unit("kg")),
    ... )
    >>> qv.value
    Array([1., 2., 3.], dtype=float64)
    >>> qv.unit.shape
    (3,)

    2D case (matrix):

    >>> qm = QuantityMatrix(
    ...     value=jnp.ones((2, 2)),
    ...     unit=((u.unit("m"), u.unit("s")), (u.unit("kg"), u.unit("rad"))),
    ... )
    >>> qm.value.shape
    (2, 2)
    >>> qm.unit.shape
    (2, 2)

    """

    value: Shaped[Array, "..."] = eqx.field()
    unit: UnitsMatrix = eqx.field(static=True, converter=UnitsMatrix)

    @property
    def ndim(self) -> int:
        """Number of real dimensions (1 for vector, 2 for matrix)."""
        return self.unit.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape, including batch dimensions."""
        return self.value.shape

    # Convenience properties for 2D case
    @property
    def n_rows(self) -> int:
        """Number of rows (for 2D case). Raises error for 1D."""
        if self.ndim != 2:
            msg = f"n_rows only available for 2D, but ndim={self.ndim}"
            raise ValueError(msg)
        return self.value.shape[-2]

    @property
    def n_cols(self) -> int:
        """Number of columns (for 2D case). Raises error for 1D."""
        if self.ndim != 2:
            msg = f"n_cols only available for 2D, but ndim={self.ndim}"
            raise ValueError(msg)
        return self.value.shape[-1]

    @property
    def n_elems(self) -> int:
        """Number of elements (for 1D case). Raises error for 2D."""
        if self.ndim != 1:
            msg = f"n_elems only available for 1D, but ndim={self.ndim}"
            raise ValueError(msg)
        return self.value.shape[-1]

    # ── Quax API ─────────────────────────────────────────────────────

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(self.value.shape, self.value.dtype)

    def materialise(self) -> NoReturn:
        msg = "Refusing to materialise `QuantityMatrix`."
        raise RuntimeError(msg)


def _iter_units(units: Any) -> Iterable[u.AbstractUnit]:
    """Yield every leaf unit from a (possibly nested) units structure."""
    for item in units:
        if isinstance(item, u.AbstractUnit):
            yield item
        else:
            yield from _iter_units(item)


@plum.conversion_method(type_from=QuantityMatrix, type_to=u.Quantity)  # type: ignore[arg-type]
def quantitymatrix_to_quantity(x: QuantityMatrix, /) -> u.Quantity:
    """Convert a ``QuantityMatrix`` to a regular ``Quantity``.

    Conversion is only valid when all elements of ``x`` share the same unit.
    If units are heterogeneous, this conversion is ambiguous and raises
    ``ValueError``.

    Examples
    --------
    >>> import plum
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> from coordinax.internal import QuantityMatrix

    Uniform units convert to a plain quantity:

    >>> qmat = QuantityMatrix(jnp.array([1.0, 2.0, 3.0]), unit=("m", "m", "m"))
    >>> plum.convert(qmat, u.Quantity)
    Q([1., 2., 3.], 'm')

    Mixed units are rejected:

    >>> bad = QuantityMatrix(jnp.array([1.0, 2.0]), unit=("m", "s"))
    >>> plum.convert(bad, u.Quantity)
    Traceback (most recent call last):
    ...
    ValueError: Cannot convert QuantityMatrix to Quantity unless all units are
    identical.

    """
    units = tuple(_iter_units(x.unit))
    if not units:
        msg = "Cannot convert QuantityMatrix with no unit entries."
        raise ValueError(msg)

    first = units[0]
    if any(unit != first for unit in units[1:]):
        msg = (
            "Cannot convert QuantityMatrix to Quantity unless all units are identical."
        )
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
    if from_units.ndim == 1:
        return _convert_value_vector(value, from_units, to_units)
    if from_units.ndim == 2:
        return _convert_value_matrix(value, from_units, to_units)
    msg = f"Unsupported ndim={from_units.ndim}"
    raise NotImplementedError(msg)


@quax.register(lax.add_p)
def add_qm_qm(x: QuantityMatrix, y: QuantityMatrix, /) -> QuantityMatrix:
    """Element-wise addition of two `QuantityMatrix` objects.

    The result adopts the units of *x*.  Each element is converted
    from ``y.unit`` → ``x.unit`` before the numeric add.

    Works for both 1D (vector) and 2D (matrix) cases.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import quaxed.numpy as qnp
    >>> import unxt as u
    >>> from coordinax.internal import QuantityMatrix

    2D case:

    >>> a = QuantityMatrix(
    ...     value=jnp.ones((2, 2)),
    ...     unit=((u.unit("m"), u.unit("s")), (u.unit("kg"), u.unit("rad"))))
    >>> b = QuantityMatrix(
    ...     value=jnp.ones((2, 2)),
    ...     unit=((u.unit("km"), u.unit("ms")), (u.unit("g"), u.unit("deg"))))

    >>> result = qnp.add(a, b)
    >>> result.unit.to_string()
    '((m, s), (kg, rad))'

    >>> result.value
    Array([[1.00100000e+03, 1.00100000e+00],
           [1.00100000e+00, 1.01745329e+00]], dtype=float64)

    1D case:

    >>> a1d = QuantityMatrix(
    ...     value=jnp.ones(3),
    ...     unit=(u.unit("m"), u.unit("s"), u.unit("kg")))
    >>> b1d = QuantityMatrix(
    ...     value=jnp.ones(3),
    ...     unit=(u.unit("km"), u.unit("ms"), u.unit("g")))

    >>> result1d = qnp.add(a1d, b1d)
    >>> result1d.unit.to_string()
    '(m, s, kg)'

    >>> result1d.value
    Array([1.001e+03, 1.001e+00, 1.001e+00], dtype=float64)

    """
    y_converted = _convert_value(y.value, y.unit, x.unit)
    return QuantityMatrix(value=lax.add(x.value, y_converted), unit=x.unit)


@quax.register(lax.dot_general_p)
def dot_general_qm_qm(
    lhs: QuantityMatrix,
    rhs: QuantityMatrix,
    /,
    *,
    dimension_numbers: lax.DotDimensionNumbers,
    precision: Any = None,
    preferred_element_type: Any = None,
    **kw: Any,
) -> QuantityMatrix | u.Quantity:
    """Dot product / matrix multiply two `QuantityMatrix` objects.

    Delegates to specialized implementations based on the dimensionality:
    - 1D @ 1D → scalar (vector dot product)
    - 2D @ 2D → 2D (matrix-matrix multiply)

    For the standard matmul contraction: contracting_dims = ((-1,), (-2,)),
    with no batch dims (batch is handled by leading dims in QuantityMatrix).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import quaxed.numpy as qnp
    >>> import unxt as u
    >>> from coordinax.internal import QuantityMatrix

    1D @ 1D (dot product):

    >>> v1 = QuantityMatrix(
    ...     value=jnp.array([1.0, 2.0]),
    ...     unit=(u.unit("m"), u.unit("km")),
    ... )
    >>> v2 = QuantityMatrix(
    ...     value=jnp.array([3.0, 4.0]),
    ...     unit=(u.unit("s"), u.unit("s")),
    ... )
    >>> result = qnp.dot(v1, v2)
    >>> result.value
    Array(8003., dtype=float64)
    >>> result.unit
    Unit("m s")

    2D @ 2D (matrix multiply):

    >>> a = QuantityMatrix(
    ...     value=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
    ...     unit=((u.unit("m"), u.unit("km")),
    ...           (u.unit("m"), u.unit("km"))),
    ... )
    >>> b = QuantityMatrix(
    ...     value=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
    ...     unit=((u.unit("s"), u.unit("s")),
    ...           (u.unit("s"), u.unit("s"))),
    ... )

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
    msg = f"Unsupported dimensionality: lhs.ndim={lhs.ndim}, rhs.ndim={rhs.ndim}"
    raise NotImplementedError(msg)


def _dot_general_1d_1d(
    lhs: QuantityMatrix,
    rhs: QuantityMatrix,
    /,
    *,
    dimension_numbers: lax.DotDimensionNumbers,
    precision: Any = None,
    preferred_element_type: Any = None,
    **kw: Any,
) -> u.Quantity:
    """Vector dot product: (N,) @ (N,) → scalar.

    Result = Σ_i  lhs[i] * rhs[i]

    All terms must be unit-compatible. We convert to the unit of the first term.
    """
    n = lhs.n_elems
    assert rhs.n_elems == n  # noqa: S101

    # Reference unit: lhs.unit[0] * rhs.unit[0]
    ref_unit = lhs.unit[0] * rhs.unit[0]

    # Compute scale factors
    scales = jnp.array(
        [u.uconvert_value(ref_unit, lhs.unit[i] * rhs.unit[i], 1.0) for i in range(n)]
    )

    # Compute dot product with rescaling
    result_value = jnp.sum(scales * lhs.value * rhs.value, axis=-1)

    return u.Quantity(result_value, ref_unit)


def _dot_general_2d_2d(
    lhs: QuantityMatrix,
    rhs: QuantityMatrix,
    /,
    *,
    dimension_numbers: lax.DotDimensionNumbers,
    precision: Any = None,
    preferred_element_type: Any = None,
    **kw: Any,
) -> QuantityMatrix:
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
    n = lhs.n_rows  # N
    k_dim = lhs.n_cols  # K  (contraction axis)
    m = rhs.n_cols  # M
    assert rhs.n_rows == k_dim  # noqa: S101

    # 1) Compute output units: ref[i][k] = lhs.unit[i][0] * rhs.unit[0][k]
    out_unit = UnitsMatrix(
        tuple(
            tuple(lhs.unit[i, 0] * rhs.unit[0, k] for k in range(m)) for i in range(n)
        )
    )

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
        [
            [
                [
                    u.uconvert_value(
                        out_unit[i, k], lhs.unit[i, j] * rhs.unit[j, k], 1.0
                    )
                    for k in range(m)
                ]
                for j in range(k_dim)
            ]
            for i in range(n)
        ]
    )

    # 3) Vectorised contraction — no Python loop, no accumulator.
    #    C[..., i, k] = Σ_j  scale[i, j, k] * A[..., i, j] * B[..., j, k]
    accum = jnp.sum(  # (N, K, M) * (..., N, K, 1) * (..., 1, K, M)
        scale_3d * lhs.value[..., :, :, None] * rhs.value[..., None, :, :],
        axis=-2,
    )

    return QuantityMatrix(value=accum, unit=out_unit)


@quax.register(lax.sub_p)
def sub_qm_qm(x: QuantityMatrix, y: QuantityMatrix, /) -> QuantityMatrix:
    """Element-wise subtraction of two `QuantityMatrix` objects.

    The result adopts the units of *x*.  Each element is converted
    from ``y.unit`` → ``x.unit`` before the numeric subtract.

    Works for both 1D (vector) and 2D (matrix) cases.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import quaxed.numpy as qnp
    >>> import unxt as u
    >>> from coordinax.internal import QuantityMatrix

    2D case:

    >>> a = QuantityMatrix(
    ...     value=jnp.ones((2, 2)),
    ...     unit=((u.unit("m"), u.unit("s")), (u.unit("kg"), u.unit("rad"))))
    >>> b = QuantityMatrix(
    ...     value=jnp.ones((2, 2)),
    ...     unit=((u.unit("km"), u.unit("ms")), (u.unit("g"), u.unit("deg"))))

    >>> result = qnp.subtract(a, b)
    >>> result.unit.to_string()
    '((m, s), (kg, rad))'

    >>> result.value
    Array([[-9.99000000e+02,  9.99000000e-01],
           [ 9.99000000e-01,  9.82546707e-01]], dtype=float64)

    1D case:

    >>> a1d = QuantityMatrix(value=jnp.ones(3),
    ...                      unit=(u.unit("m"), u.unit("s"), u.unit("kg")))
    >>> b1d = QuantityMatrix(value=jnp.ones(3),
    ...                      unit=(u.unit("km"), u.unit("ms"), u.unit("g")))

    >>> result1d = qnp.subtract(a1d, b1d)
    >>> result1d.unit.to_string()
    '(m, s, kg)'

    >>> result1d.value
    Array([-999.   ,    0.999,    0.999], dtype=float64)

    """
    y_converted = _convert_value(y.value, y.unit, x.unit)
    return QuantityMatrix(value=lax.sub(x.value, y_converted), unit=x.unit)
