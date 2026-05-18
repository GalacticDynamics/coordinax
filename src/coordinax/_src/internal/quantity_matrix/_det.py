"""Custom ``det_p`` JAX primitive with full JAX transform support.

JAX has no built-in ``det`` primitive.  ``jnp.linalg.det`` decomposes into
arithmetic (2x2 / 3x3) or ``lu_p`` + log/exp (larger matrices).  We define
``det_p`` here so that Quax can intercept determinant calls on ``QMatrix``
objects.

Supported transforms:
  - JIT via MLIR lowering that delegates to ``jnp.linalg.det``
  - Forward-mode autodiff (JVP): d(det A)(dA) = det(A) · tr(A⁻¹ dA)
  - Reverse-mode autodiff (VJP): derived automatically from JVP via
    transposition — no explicit transpose rule needed because the JVP tangent
    only uses existing primitives (linalg.solve, trace, mul) that already
    carry transpose rules.
  - Batching (vmap): move the batch axis to the front and call det_p; the MLIR
    lowering handles any (*batch, n, n) shape natively.
"""

import functools as ft
import operator

from jaxtyping import Array

import jax
import jax.core
import jax.numpy as jnp
import quax
from jax import lax
from jax.extend import core as jexc
from jax.interpreters import ad as jax_ad, batching as jax_batching, mlir as jax_mlir

import unxt as u

from ._quantity_matrix import QMatrix

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


def _det_abstract_eval(x: jax.core.ShapedArray, /) -> jax.core.ShapedArray:
    if x.ndim < 2:
        raise ValueError(f"det_p requires at least 2-D input, got ndim={x.ndim}")
    if x.shape[-1] != x.shape[-2]:
        raise ValueError(
            f"det_p requires a square matrix "
            f"(shape[-2] == shape[-1]), got shape={x.shape}"
        )
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
        raise ValueError(
            f"det_p QMatrix dispatch requires a 2-D unit structure, got ndim={x.ndim}"
        )
    det_val = det_p.bind(x.value)
    n = x.unit.shape[0]
    det_unit = ft.reduce(operator.mul, (x.unit[i, i] for i in range(n)))
    return u.Q(det_val, det_unit)
