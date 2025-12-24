"""Base classes for operators."""

__all__ = ("operate", "simplify")

from typing import Any

from plum import dispatch


@dispatch.abstract
def operate(op: type, params: dict[str, Any], x: Any, /) -> Any:
    """Apply an :class:`coordinax.ops.AbstractOperator` to the input.

    The way to think about this function is that it takes:

    1. An operator type subclass of :class:`coordinax.ops.AbstractOperator`.
       This determines how the operation is applied to the input `x` given the
       parameters, `params`, of the operator.
    2. The parameters needed to construct an instance of that operator type. No
       actual instance is constructed. The operator subclass defines how to
       apply the operation given these parameters.
    3. The input, `x`, to which the operation is applied.

    Many of the operators can work on time-dependent coordinates, in which case
    the `operate` function may also take a time argument `tau` preceding `x`.

    ```{note}
    This function uses multiple dispatch. Dispatches made in other modules may
    not be included in the rendered docs. To see the full range of options,
    execute ``coordinax.ops.operate.methods`` in an interactive Python session.
    ```

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    ### `Identity` operator

    An :class:`coordinax.ops.Identity` operator leaves the input unchanged:

    >>> q = u.Quantity([1, 2, 3], "km")
    >>> cx.ops.operate(cx.ops.Identity, {}, q) is q
    True

    ### `Add` operator

    An :class:`coordinax.ops.Add` operator translates the input by a fixed
    amount:

    >>> shift = u.Quantity([1, 1, 1], "km")
    >>> cx.ops.operate(cx.ops.Add, {"delta": shift}, q)
    Quantity(Array([2, 3, 4], dtype=int32), unit='km')

    ### `Rotate` operator

    A :class:`coordinax.ops.Rotate` operator rotates the input by a rotation
    matrix:

    >>> R = jnp.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    >>> cx.ops.operate(cx.ops.Rotate, {"R": R}, q)
    Quantity(Array([-2,  1,  3], dtype=int32), unit='km')

    ### Galilean Operator

    A :class:`coordinax.ops.Galilean` operator shift, rotates, and boosts the input.

    >>> pass

    """
    raise NotImplementedError  # pragma: no cover


# ===================================================================


@dispatch.abstract
def simplify(op: Any, /) -> Any:
    """Simplify an operator (:class:`coordinax.ops.AbstractOperator`).

    This function takes an operator and attempts to simplify it, returning a
    new, potentially simpler operator.

    ```{note}
    This function uses multiple dispatch. Dispatches made in other modules may
    not be included in the rendered docs. To see the full range of options,
    execute ``coordinax.ops.simplify.methods`` in an interactive Python session.
    ```

    Examples
    --------
    In these examples, we demonstrate simplifying various operators.  There are
    more worked examples available in the docstrings of the dispatch methods.

    >>> import coordinax as cx

    1. :class:`coordinax.ops.Identity`

    The :class:`coordinax.ops.Identity` operator is the simplest operator and
    cannot be simplified further:

    >>> op = cx.ops.Identity()
    >>> cx.ops.simplify(op) is op
    True

    2. :class:`coordinax.ops.Add`

    An :class:`coordinax.ops.Add` operator with a non-zero delta cannot be simplified:

    >>> import unxt as u

    >>> qshift = cx.CartesianPos3D.from_([1, 1, 1], "km")
    >>> op = cx.ops.Add(qshift)
    >>> # cx.ops.simplify(op) == op  # TODO

    When the delta is zero, the operator simplifies to an
    :class:`coordinax.ops.Identity`:

    >>> qshift = cx.CartesianPos3D.from_([0, 0, 0], "km")
    >>> op = cx.ops.Add(qshift)
    >>> # cx.ops.simplify(op) == cx.ops.Identity()  # TODO

    3. :class:`coordinax.ops.Rotate`

    A :class:`coordinax.ops.Rotate` operator with a non-trivial rotation matrix
    cannot be simplified:

    >>> op = cx.ops.Rotate.from_euler("z", u.Quantity(45, "deg"))
    >>> simplified = cx.ops.simplify(op)
    >>> simplified
    Rotate(R=f32[3,3])

    A trivial rotation (identity matrix) simplifies to an
    :class:`coordinax.ops.Identity`:

    >>> op = cx.ops.Rotate.from_euler("z", u.Quantity(0, "deg"))
    >>> simplified = cx.ops.simplify(op)
    >>> simplified == cx.ops.Identity()
    True

    """
    raise NotImplementedError  # pragma: no cover
