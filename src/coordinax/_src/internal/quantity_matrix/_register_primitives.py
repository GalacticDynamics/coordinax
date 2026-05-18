"""Quax primitive registrations for QMatrix arithmetic.

Registers handlers for the following JAX primitives:
- ``lax.add_p`` — element-wise addition
- ``lax.sub_p`` — element-wise subtraction
- ``lax.dot_general_p`` — dot product / matrix multiply
- ``lax.transpose_p`` — matrix transpose
- ``lax.gather_p`` — element-selection gather (e.g. jnp.diag)
- ``lax.reduce_sum_p`` — summation reduction
"""

from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import quax
from jax import lax

import unxt as u
from unxt.quantity import AllowValue

from ._quantity_matrix import QMatrix, _convert_value
from ._units_matrix import UnitsMatrix

# Vectorised uconvert_value — used by dot-product helpers.
vec_uconvert_value = np.vectorize(u.uconvert_value)

_DMLS = u.unit("")


# ── add / sub ────────────────────────────────────────────────────────────


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


# ── dot_general helpers ───────────────────────────────────────────────────


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
        scale_3d * lhs.value[..., :, :, None] * rhs.value[..., None, :, :],
        axis=-2,
    )

    return QMatrix(value=accum, unit=out_unit)


# ── dot_general dispatch ──────────────────────────────────────────────────


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
    ...     unit=UnitsMatrix((("m2 / rad2", "m2 / rad2"), ("m2 / rad2", "m2 / rad2"))),
    ... )
    >>> v = u.Q(jnp.array([1.0, 1.0]), "rad")
    >>> w = qnp.matmul(g, v)
    >>> w.unit.to_string()
    '(m2 / rad, m2 / rad)'
    >>> w.value
    Array([2., 3.], dtype=float64)

    """
    rhs_unit = u.unit_of(rhs)
    rhs_val = cast("jax.Array", u.ustrip(AllowValue, rhs_unit, rhs))
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


# ── transpose ────────────────────────────────────────────────────────────


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


# ── gather ───────────────────────────────────────────────────────────────


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

    if isinstance(start_indices, jax.core.Tracer):  # ty: ignore[possibly-missing-submodule]
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


# ── reduce_sum ───────────────────────────────────────────────────────────


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
