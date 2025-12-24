"""Base classes for operators on coordinates."""

__all__ = ("AbstractOperator", "eval_op")

from abc import abstractmethod
from dataclasses import dataclass, fields as dc_fields

from collections.abc import Mapping
from jaxtyping import ArrayLike
from typing import TYPE_CHECKING, Any, Final, TypeVar, final

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jtu
import plum
import wadler_lindig as wl

import unxt as u
from dataclassish import field_items, flags

from coordinax._src import api

if TYPE_CHECKING:
    import coordinax.ops  # noqa: ICN001

_sentinel: Final = object()


@final
@dataclass(slots=True)
class Neg:
    """A parameter that negates another parameter."""

    param: Any
    """The parameter to negate."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the parameter and negate the result."""
        return -self.param(*args, **kwargs)

    def __neg__(self) -> Any:
        """Return the original parameter."""
        return self.param


class AbstractOperator(eqx.Module):
    """Abstract base class for operators on coordinates.

    An operator is an object that defines a transformation on coordinates. It
    can be applied to a set of coordinates to produce a new set of coordinates.
    Operators can be composed together to form a sequence of transformations.

    """

    # ---------------------------------------------------------------
    # Constructors

    @classmethod
    @plum.dispatch.abstract
    def from_(
        cls: "type[AbstractOperator]", *args: object, **kwargs: object
    ) -> "AbstractOperator":
        """Construct from a set of arguments."""
        raise NotImplementedError  # pragma: no cover

    # ===============================================================
    # Operator API

    @plum.dispatch
    def __call__(self: "AbstractOperator", tau: Any, x: Any, /, **kwargs: Any) -> Any:
        """Apply the operator to the arguments.

        This method calls `coordinax.ops.apply_op` to apply the operator.

        """
        return api.apply_op(self, tau, x, **kwargs)

    @plum.dispatch
    def __call__(self: "AbstractOperator", x: Any, /, **kwargs: Any) -> Any:
        """Apply the operator to the arguments with tau=None.

        This method calls `coordinax.ops.apply_op` to apply the operator.

        """
        return api.apply_op(self, None, x, **kwargs)

    # -------------------------------------------

    @property
    @abstractmethod
    def inverse(self) -> "AbstractOperator":
        """The inverse of the operator."""
        ...

    def simplify(self) -> "AbstractOperator":
        """Simplify the operator.

        This method calls `coordinax.ops.simplify` to simplify the
        operator.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax.ops as cxo

        >>> op = cxo.Identity()
        >>> op.simplify() is op
        True

        >>> pipe = cxo.Pipe((cxo.Identity(), cxo.Identity()))
        >>> pipe
        Pipe((Identity(), Identity()))
        >>> pipe.simplify()
        Identity()

        >>> op = cxo.GalileanOp(translation=u.Q([0., 2., 3., 4.], "km"))
        >>> op.simplify()
        Add(delta=Q(f32[4], 'km'))

        """
        return api.simplify(self)

    # ===============================================================
    # Wadler-Lindig API

    def __pdoc__(self, **kwargs: Any) -> wl.AbstractDoc:
        """Return the documentation for the operator.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments passed to the documentation
            generation functions.

            - compact_arrays : bool, optional
            - indent : int, optional
            - positional_fields : bool, optional
                If True, introspect the dataclass to display non-keyword-only
                fields without their parameter names (positionally).

        """
        # Prefer to use short names (e.g. Quantity -> Q) and compact unit forms
        kwargs.setdefault("use_short_name", True)
        kwargs.setdefault("named_unit", False)

        # Get the field items, excluding those that should not be shown and
        # those that are equal to the default value.
        fitems = list(field_items(flags.FilterRepr, self))
        fitems = [
            (k, v)
            for k, v in fitems
            if not jnp.all(v == getattr(self.__class__, k, _sentinel))
        ]

        # Handle positional fields (fields without names)
        use_positional = kwargs.pop("positional_fields", False)

        # Make the field docs list
        if len(fitems) == 1 and kwargs.get("short_arrays") == "compact":
            docs = [wl.TextDoc(str(fitems[0][1]))]
        elif use_positional:
            # Introspect to find which fields are keyword-only
            kw_only_fields = {f.name for f in dc_fields(self) if f.kw_only}

            # Separate positional and named fields
            pos_items = [(k, v) for k, v in fitems if k not in kw_only_fields]
            named_items = [(k, v) for k, v in fitems if k in kw_only_fields]

            # Create docs for positional fields (without names)
            pos_docs = [wl.pdoc(v, **kwargs) for k, v in pos_items]
            # Create docs for named fields
            named_docs = wl.named_objs(named_items, **kwargs) if named_items else []

            docs = pos_docs + named_docs
        else:
            docs = wl.named_objs(fitems, **kwargs)

        return wl.bracketed(
            begin=wl.TextDoc(f"{self.__class__.__name__}("),
            docs=docs,
            sep=wl.comma,
            end=wl.TextDoc(")"),
            indent=kwargs.get("indent", 4),
        )

    # ===============================================================
    # Python API

    def __repr__(self) -> str:
        return wl.pformat(self, use_short_name=True, positional_fields=True)

    def __str__(self) -> str:
        """Return a string representation of the operator.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import coordinax as cx

        >>> op = cx.ops.Rotate([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> print(op)
        Rotate([[1 0 0]
                          [0 1 0]
                          [0 0 1]])

        >>> op = cx.ops.GalileanOp(
        ...     translation=u.Q([0., 2, 3, 4], "km"),
        ...     velocity=u.Q([1., 2, 3], "km/s"),
        ...     rotation=jnp.eye(3).at[0, 2].set(1),
        ... )
        >>> print(op)
        GalileanOp(
            rotation=Rotate([[1. 0. 1.]
                            [0. 1. 0.]
                            [0. 0. 1.]]),
            translation=Add( delta_t=..., delta_q=... ),
            velocity=Boost(CartVel3D( ... ))
        )

        """
        return wl.pformat(self, width=88, vector_form=True, short_arrays="compact")

    # -------------------------------------------
    # Operator Composition

    def __or__(self, other: "AbstractOperator", /) -> "coordinax.ops.Pipe":
        """Compose with another operator.

        Examples
        --------
        >>> import coordinax.ops as cxo

        >>> op1 = cxo.Rotate([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> op2 = cxo.Identity()
        >>> op3 = op1 | op2
        >>> op3
        Pipe((Rotate(R=i32[3,3]), Identity()))

        >>> op4 = cxo.Identity() | op3
        >>> op4
        Pipe((Identity(), Rotate(R=i32[3,3]), Identity()))

        """
        from .pipe import Pipe  # noqa: PLC0415

        # Defer to other's __ror__ if it's a Pipe
        if isinstance(other, Pipe):
            return other.__ror__(self)

        # Otherwise, create a new Pipe
        return Pipe((self, other))


# ============================================================
# Constructors


@AbstractOperator.from_.dispatch(precedence=-1)
def from_(
    cls: type[AbstractOperator], *args: object, **kwargs: object
) -> AbstractOperator:
    """Construct from a set of arguments.

    This is a low-priority dispatch that will be called if no other
    dispatch is found. It just tries to pass the arguments to the
    constructor.

    """
    return cls(*args, **kwargs)


@AbstractOperator.from_.dispatch
def from_(cls: type[AbstractOperator], obj: Mapping[str, Any], /) -> AbstractOperator:
    """Construct from a mapping.

    Examples
    --------
    >>> import coordinax.ops as cxo
    >>> cxo.Pipe.from_({"operators": (cxo.Identity(), cxo.Identity())})
    Pipe((Identity(), Identity()))

    """
    return cls(**obj)


@AbstractOperator.from_.dispatch
def from_(
    cls: type[AbstractOperator],
    x: ArrayLike | list[float | int],
    unit: str,  # TODO: support unit object
    /,
) -> AbstractOperator:
    """Construct from a Quantity's value and unit.

    Examples
    --------
    >>> import coordinax.ops as cxo

    >>> op = cxo.GalileanOp.from_([1, 1, 1], "km")
    >>> print(op)
    GalileanOp(<Cart3D: (x, y, z) [km]
        [1 1 1]>)

    >>> op = cxo.Translate.from_([1, 1, 1], "km")
    >>> print(op)
    Translate(Cart3D( ... ))

    >>> op = cxo.Boost.from_([1, 1, 1], "km/s")
    >>> print(op)
    Boost(CartVel3D( ... ))

    """
    return cls.from_(u.Q(x, unit))


@AbstractOperator.from_.dispatch
def from_(cls: type[AbstractOperator], obj: AbstractOperator, /) -> AbstractOperator:
    """Construct an operator from another operator.

    Raises
    ------
    TypeError
        If the input object is not a subclass of the target class.

    Examples
    --------
    >>> import coordinax as cx

    If the object is the same type, it should return the object itself.

    >>> op = cx.ops.Identity()
    >>> cx.ops.Identity.from_(op) is op
    True

    If the object is a different type, it will error.

    >>> try:
    ...     cx.ops.Translate.from_(op)
    ... except TypeError as e:
    ...     print(e)
    Cannot construct <class '...Translate'> from <class '...Identity'>.

    Unless the object is a subclass of the target class.

    >>> class MyOperator(cx.ops.Identity):
    ...     pass

    >>> op = MyOperator()
    >>> op
    MyOperator()

    >>> newop = cx.ops.Identity.from_(op)
    >>> newop is op, isinstance(newop, cx.ops.Identity)
    (False, True)

    """
    if not isinstance(obj, cls):
        msg = f"Cannot construct {cls} from {type(obj)}."
        raise TypeError(msg)

    # avoid copying if the types are the same. `isinstance` is not strict
    # enough, so we use type() instead.
    if type(obj) is cls:  # pylint: disable=unidiomatic-typecheck
        return obj

    return cls(**dict(field_items(obj)))


# =============================================================================
# eval_op: Materialization of time-dependent parameters

OpT = TypeVar("OpT", bound=AbstractOperator)


def eval_op(op: OpT, tau: Any, /) -> OpT:
    r"""Evaluate time-dependent parameters of an operator at a given time.

    This function materializes an operator by evaluating all callable
    (time-dependent) parameters at the specified time ``tau``, returning
    a new operator instance of the same type with purely numeric parameters.

    Mathematically, if an operator $\mathrm{Op}$ has parameters that depend on
    an affine parameter $\tau$, then:

    $$
    \mathrm{eval\_op}(\mathrm{Op}, \tau) \to \mathrm{Op}_\tau
    $$

    where $\mathrm{Op}_\tau$ is the operator with all time-dependent parameters
    evaluated at $\tau$.

    Parameters
    ----------
    op : AbstractOperator
        The operator to evaluate. May contain time-dependent parameters
        (callables that take ``tau`` as argument).
    tau : Any
        The time/affine parameter at which to evaluate time-dependent
        parameters. Typically a ``unxt.Quantity`` with time units.

    Returns
    -------
    AbstractOperator
        A new operator of the same type with all time-dependent parameters
        evaluated at ``tau``. If the operator has no time-dependent parameters,
        returns a copy with equivalent parameters.

    Notes
    -----
    This function is:

    - **Pure**: No side effects, safe for JAX tracing
    - **Structure-preserving**: Returns same operator type
    - **Pytree-compatible**: Uses ``equinox.partition`` / ``equinox.combine``

    The implementation:

    1. Partitions operator fields into callable (dynamic) and static
    2. Evaluates dynamic fields at ``tau`` via ``jax.tree_util.tree_map``
    3. Recombines into a new operator instance

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.ops as cxo

    **Time-dependent operator:**

    >>> op = cxo.Translate(lambda t: u.Q(t.ustrip("s"), "km"))
    >>> tau = u.Q(5.0, "s")
    >>> op_eval = cxo.eval_op(op, tau)
    >>> op_eval.delta
    Quantity['length'](Array(5., dtype=float64, weak_type=True), unit='km')

    **Static operator (no change):**

    >>> op_static = cxo.Translate(u.Q([1, 2, 3], "km"))
    >>> op_eval_static = cxo.eval_op(op_static, tau)
    >>> op_eval_static.delta
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='km')

    **Identity operator:**

    >>> identity = cxo.Identity()
    >>> cxo.eval_op(identity, tau)
    Identity()

    See Also
    --------
    apply_op : Apply an operator to an input (calls ``eval_op`` internally)

    """
    # Get all parameter values from the operator
    params = {k: getattr(op, k) for k in op.__dataclass_fields__}

    # Partition into callable (time-dependent) and static parameters
    dynamic, static = eqx.partition(params, filter_spec=callable)

    # Evaluate the dynamic parameters at the given time
    eval_dynamic = jtu.map(lambda p: p(tau), dynamic)

    # Recombine the static and evaluated dynamic parameters
    evaluated_params = eqx.combine(static, eval_dynamic)

    # Create a new operator of the same type with the evaluated parameters
    return type(op)(**evaluated_params)
