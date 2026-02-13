"""Vector API for coordinax."""

__all__ = ("apply_op", "simplify")

from typing import TYPE_CHECKING, Any

import plum

if TYPE_CHECKING:
    import coordinax.roles  # noqa: ICN001


@plum.dispatch.abstract
def apply_op(op: "coordinax.ops.AbstractOperator", tau: Any, x: Any, /) -> Any:
    r"""Apply an operator to an input at a given time.

    This is the core dispatch function for operator application. Each operator
    type registers its own implementation via multiple dispatch. Operators act
    on various input types (Quantity, Vector, CsDict) according to their
    semantics.

    Mathematical Definition:

    For an operator $\mathcal{O}$ parameterized by time $\tau$, this computes:

    $$ x' = \mathcal{O}(\tau)(x) $$

    For time-independent operators, $\tau$ is ignored. For composite operators
    (e.g., ``Pipe``, ``GalileanOp``), the component operators are applied
    sequentially.

    Parameters
    ----------
    op : coordinax.ops.AbstractOperator
        The operator to apply. This can be any operator type:

        - ``Translate``: Spatial translation (Point role)
        - ``Boost``: Velocity offset (Vel role)
        - ``Rotate``: Spatial rotation
        - ``Identity``: No-op
        - ``Pipe``: Sequential composition
        - ``GalileanOp``: Full Galilean transformation

    tau : Any
        Time parameter for time-dependent operators. Pass ``None`` for
        time-independent operators. For time-dependent operators, this is
        used to evaluate callable parameters (e.g., ``Translate(lambda t: ...)``)
        via ``eval_op``.

    x : Any
        The input to transform. Supported types depend on the operator:

        - ``Quantity``: Direct arithmetic application
        - ``Vector``: Role-aware transformation with chart preservation
        - ``CsDict``: Low-level component dict (requires ``role=`` kwarg)

    **kwargs : Any
        Additional keyword arguments passed to the dispatch:

        - ``role``: Required for CsDict inputs to specify geometric role
        - ``at``: Base point for non-Euclidean transformations (future)

    Returns
    -------
    Any
        The transformed input, same type as ``x``. For role-specialized
        operators (``Translate``, ``Boost``), the role of the output matches the
        input.

    Raises
    ------
    TypeError
        If a role-specialized operator is applied to an incompatible role.
        For example, applying ``Translate`` to a ``PhysVel``-role vector raises
        ``TypeError``.
    NotImplementedError
        If no dispatch is registered for the given ``(operator, input)`` types.

    Notes
    -----
    - **Role enforcement**: ``Translate`` only acts on ``Point`` role,
      ``Boost`` only on ``PhysVel`` role.
      This ensures geometric correctness (points translate, velocities boost).

    - **Operator.__call__**: The ``__call__`` method of operators delegates
      to this function: ``op(tau, x)`` is equivalent to ``apply_op(op, tau, x)``.

    - **Time evaluation**: For operators with callable parameters, ``eval_op``
      is called internally to materialize the time-dependent values.

    - **Composite operators**: For ``Pipe`` and ``GalileanOp``, the component
      operators are applied in sequence (left-to-right for Pipe).

    See Also
    --------
    coordinax.ops.eval_op : Evaluate time-dependent operator parameters
    coordinax.ops.simplify : Simplify operators to canonical form
    coordinax.ops.Translate : Point translation operator
    coordinax.ops.Boost : Velocity boost operator

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.ops as cxop

    **Apply a translation to a Quantity:**

    >>> shift = cxop.Translate.from_([1, 2, 3], "km")
    >>> q = u.Q([0, 0, 0], "km")
    >>> cxop.apply_op(shift, None, q)
    Quantity(Array([1, 2, 3], dtype=int64), unit='km')

    **Apply a boost to a velocity Quantity:**

    >>> boost = cxop.Boost.from_([100, 0, 0], "km/s")
    >>> v = u.Q([0, 50, 0], "km/s")
    >>> cxop.apply_op(boost, None, v)
    Quantity(Array([100,  50,   0], dtype=int64), unit='km / s')

    **Apply a rotation:**

    >>> import jax.numpy as jnp
    >>> Rz = jnp.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    >>> rot = cxop.Rotate(Rz)
    >>> q = u.Q([1, 0, 0], "m")
    >>> cxop.apply_op(rot, None, q)
    Quantity(Array([0, 1, 0], dtype=int64), unit='m')

    **Composite operator (GalileanOp):**

    >>> op = cxop.GalileanOp(
    ...     translation=cxop.Translate.from_([1, 0, 0], "km"),
    ...     velocity=cxop.Boost.from_([0, 0, 0], "km/s"),
    ... )
    >>> q = u.Q([0, 0, 0], "km")
    >>> cxop.apply_op(op, None, q)
    Quantity(Array([1., 0., 0.], dtype=float64), unit='km')

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def simplify(op: Any, /) -> Any:
    """Simplify an operator to a canonical form.

    This function takes an operator and attempts to simplify it, returning a
    new, potentially simpler operator. For example, a ``Translate`` with zero
    delta simplifies to ``Identity``.

    Parameters
    ----------
    op : AbstractOperator
        The operator to simplify.

    Returns
    -------
    AbstractOperator
        A simplified operator. May be a different type (e.g., ``Identity``)
        if the original operator has no effect.

    Notes
    -----
    This function uses multiple dispatch. Each operator type registers its
    own simplification rules.

    To see all available dispatches::

        >>> import coordinax.ops as cxop
        >>> cxop.simplify.methods  # doctest: +ELLIPSIS
        List of ... method(s):
        ...

    Examples
    --------
    >>> import coordinax.ops as cxop

    **Identity (already simple):**

    >>> op = cxop.Identity()
    >>> cxop.simplify(op) is op
    True

    **Translate with zero delta:**

    >>> op = cxop.Translate.from_([0, 0, 0], "m")
    >>> cxop.simplify(op)
    Identity()

    **Translate with non-zero delta (no simplification):**

    >>> op = cxop.Translate.from_([1, 2, 3], "m")
    >>> simplified = cxop.simplify(op)
    >>> type(simplified).__name__
    'Translate'

    **Rotate with identity matrix:**

    >>> import unxt as u
    >>> op = cxop.Rotate.from_euler("z", u.Q(0, "deg"))
    >>> cxop.simplify(op)
    Identity()

    """
    raise NotImplementedError  # pragma: no cover
