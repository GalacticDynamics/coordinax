"""Sequence of Operators."""

__all__ = ("Composed",)

from dataclasses import replace

from jaxtyping import Array, ArrayLike
from typing import Any, Generic, TypeVarTuple, cast, final

import equinox as eqx
import plum
import wadler_lindig as wl

import unxt as u

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinaxs.api.transforms as cxfmapi
from .base import AbstractTransform
from .composite import AbstractCompositeTransform
from .custom_types import CDict
from .identity import Identity, identity
from coordinax.transforms._src import groups

Ts = TypeVarTuple("Ts")


def convert_to_transforms_tuple(inp: Any, /) -> tuple[AbstractTransform, ...]:
    """Convert to a tuple of transforms for `Pipe`.

    >>> import coordinax.transforms as cxfm

    >>> op1 = cxfm.Rotate([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> op2 = cxfm.Identity()
    >>> convert_to_transforms_tuple((op1, op2))
    (Rotate(i64[3,3](jax)), Identity())

    >>> op1 = cxfm.Rotate([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> convert_to_transforms_tuple(op1)
    (Rotate(i64[3,3](jax)),)

    >>> op1 = cxfm.Identity()
    >>> op2 = cxfm.Identity()
    >>> pipe = cxfm.Composed((op1, op2))
    >>> convert_to_transforms_tuple(pipe)
    (Identity(), Identity())

    """
    if isinstance(inp, (tuple, list)):
        return tuple(inp)
    if isinstance(inp, Composed):
        return inp.transforms
    if isinstance(inp, AbstractTransform):
        return (inp,)

    msg = f"Cannot convert object of type {type(inp)} to a tuple of transforms."
    raise ValueError(msg)


# =============================================================


@final
class Composed(AbstractCompositeTransform, Generic[*Ts]):
    r"""Composition of Transforms.

    Piping refers to a process in which the output of one operation is directly
    passed as the input to another. This is a composite operator that represents
    a sequence of operations to be applied in order.

    `Composed` transformations can be created using the 'pipe' syntax `op1 |
    op2`. A `Composed` transformation created as ``FG = F | G``, when evaluated,
    is equivalent to evaluating $g \circ f = g(f(x))$. Note the order of the
    transformations!

    .. note::

        The ``|`` operator works differently from the functional composition
        operator $\circ$, which is sadly not supported in Python. The ``|``
        operator is like the Unix Shell pipe operator, where output is passed
        left-to-right. This order can be seen in the indexing of the
        transformations in the `Composed` object.

    Parameters
    ----------
    transforms
        The sequence of transformations to apply.

    Examples
    --------
    >>> import coordinax.transforms as cxfm

    >>> shift = cxfm.Translate.from_([1, 2, 3], "km")
    >>> rotate = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    >>> pipe = cxfm.Composed((shift, rotate))
    >>> pipe
    Composed(( Translate(...), Rotate(...) ))

    A pipe can also be constructed by ``|``:

    >>> pipe2 = shift | rotate
    >>> pipe2
    Composed(( Translate(...), Rotate(...) ))

    The pipe can be simplified. For this example, we add an identity operator to
    the sequence and simplify, which will remove the identity operator.

    >>> pipe3 = pipe2 | cxfm.Identity()
    >>> pipe3
    Composed(( Translate(...), Rotate(...), Identity() ))

    >>> cxfm.simplify(pipe3)
    Composed(( Translate(...), Rotate(...) ))

    """

    transforms: tuple[*Ts] = eqx.field(converter=convert_to_transforms_tuple)

    def groups(self) -> frozenset[type]:
        """Return the least common supergroup of the component transforms."""
        component_groups = tuple(
            groups.most_specific_group(op.groups()) for op in self.transforms
        )
        group = groups.least_common_supergroup(component_groups)
        return frozenset((group, groups.DiffeomorphismGroup))

    # ---------------------------------------------------------------

    def __or__(self, other: AbstractTransform) -> "Composed":
        """Compose with another transform.

        Examples
        --------
        >>> import coordinax.transforms as cxfm

        >>> shift = cxfm.Translate.from_([1, 2, 3], "km")
        >>> rotate = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
        >>> pipe = cxfm.Composed((shift, rotate))

        >>> pipe | pipe
        Composed(( Translate(...), Rotate(...), Translate(...), Rotate(...) ))

        """
        # Concatenate sequences
        if isinstance(other, type(self)):
            return replace(self, transforms=self.transforms + other.transforms)
        # Append single transform
        return replace(self, transforms=(*self.transforms, other))

    def __ror__(self, other: AbstractTransform) -> "Composed":
        """Compose with another transform."""
        # Append single transform
        return replace(self, transforms=(other, *self.transforms))

    def __pdoc__(self, **kw: Any) -> wl.AbstractDoc:
        """Return the Wadler-Lindig representation.

        This is used to generate the documentation for the operator.

        Examples
        --------
        >>> import wadler_lindig as wl
        >>> import coordinax.transforms as cxfm

        >>> shift = cxfm.Translate.from_([1, 2, 3], "km")
        >>> rotate = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
        >>> pipe = cxfm.Composed((shift, rotate))
        >>> wl.pprint(pipe)
        Composed(( Translate(...), Rotate(...) ))

        """
        # Prefer to use short names (e.g. Quantity -> Q) and compact unit forms
        kw.setdefault("short_arrays", "compact")
        kw.setdefault("use_short_name", True)
        kw.setdefault("named_unit", False)
        kw.setdefault("include_params", False)

        # Build docs for each operator
        docs = [wl.pdoc(op, **kw) for op in self.transforms]
        # Wrap in ((...)) if more than one operator
        begin = wl.TextDoc("((" if len(docs) > 1 else "(")
        end = wl.TextDoc("))" if len(docs) > 1 else ")")
        # Assemble in Composed(...)
        return wl.bracketed(
            begin=wl.TextDoc(f"{self.__class__.__name__}") + begin,
            docs=docs,
            sep=wl.comma,
            end=end,
            indent=kw.get("indent", 4),
        )


# ===================================================================
# Compose


@plum.dispatch
def compose(*transforms: AbstractTransform) -> Composed:
    """Compose multiple transforms into a `Composed` transform.

    >>> import coordinax.transforms as cxfm

    >>> shift = cxfm.Translate.from_([1, 2, 3], "km")
    >>> rotate = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))

    >>> pipe = cxfm.compose(shift, rotate)
    >>> pipe
    Composed(( Translate(...), Rotate(...) ))

    """
    # Flatten nested Composed objects
    flattened = tuple(
        item
        for t in transforms
        for item in (t.transforms if isinstance(t, Composed) else (t,))
    )
    return Composed(transforms=flattened)


# ===================================================================
# `act` for Composed


@plum.dispatch
def act(
    op: Composed,
    tau: Any,
    x: ArrayLike,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    **kw: object,
) -> Array:
    """Apply Composed to an ArrayLike by sequentially applying each transform.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxfm
    >>> import coordinax.representations as cxr

    >>> shift = cxfm.Translate.from_([1, 2, 3], "km")
    >>> rot = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    >>> pipe = cxfm.Composed((shift,))
    >>> x = jnp.array([0.0, 0.0, 0.0])
    >>> usys = u.unitsystems.si
    >>> cxfm.act(pipe, None, x, cxc.cart3d, cxr.point, usys=usys)
    Array([1000., 2000., 3000.], dtype=float64)

    """
    result: Any = x
    for sub_op in op.transforms:
        result = cxfmapi.act(sub_op, tau, result, chart, rep, **kw)
    return cast("Array", result)


@plum.dispatch
def act(
    op: Composed,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    **kw: object,
) -> CDict:
    """Apply Composed to a CDict by sequentially applying each transform.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxfm
    >>> import coordinax.representations as cxr

    >>> shift = cxfm.Translate.from_([1, 2, 3], "km")
    >>> pipe = cxfm.Composed((shift,))
    >>> data = {"x": u.Q(0, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")}
    >>> cxfm.act(pipe, None, data, cxc.cart3d, cxr.point)
    {'x': Q(1, 'km'), 'y': Q(2, 'km'), 'z': Q(3, 'km')}

    """
    # 'at' / 'at_vel' track the evolving jet slots (base point and velocity)
    # for tangent transformations: advance them after each sub-op so the next
    # step evaluates at the correct anchor. Whether each anchor is provided is
    # loop-invariant, so the step kwargs dict is built once and updated.
    current_at = kw.get("at")
    current_at_vel = kw.get("at_vel")
    kw_rest = {k: v for k, v in kw.items() if k not in ("at", "at_vel")}
    step_kw = dict(kw_rest)
    at_kw = {"at": current_at} if current_at is not None else {}
    result = x
    last = len(op.transforms) - 1
    for i, sub_op in enumerate(op.transforms):
        if current_at is not None:
            step_kw["at"] = current_at
            at_kw["at"] = current_at
        if current_at_vel is not None:
            step_kw["at_vel"] = current_at_vel
        result = cxfmapi.act(sub_op, tau, result, chart, rep, **step_kw)
        if i == last:
            # The advanced anchors would never be read — skip the (eager-mode)
            # dead work of transforming them through the final sub-op.
            break
        # Advance the anchor jet (velocity first: it needs the old base point).
        if current_at_vel is not None:
            current_at_vel = cxfmapi.act(
                sub_op, tau, current_at_vel, chart, cxr.coord_vel, **kw_rest, **at_kw
            )
        if current_at is not None:
            current_at = cxfmapi.act(
                sub_op, tau, current_at, chart, cxr.point, **kw_rest
            )
    return cast("CDict", result)


@plum.dispatch
def act(
    op: Composed,
    tau: Any,
    x: u.AbstractQuantity,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    **kw: object,
) -> u.AbstractQuantity:
    """Apply Composed to a Quantity by sequentially applying each transform.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxfm
    >>> import coordinax.representations as cxr

    >>> shift = cxfm.Translate.from_([1, 2, 3], "km")
    >>> pipe = cxfm.Composed((shift,))
    >>> q = u.Q([0, 0, 0], "km")
    >>> cxfm.act(pipe, None, q, cxc.cart3d, cxr.point)
    Q([1, 2, 3], 'km')

    """
    result = x
    for xfm in op.transforms:
        result = cxfmapi.act(xfm, tau, result, chart, rep, **kw)
    return result  # ty: ignore[invalid-return-type]


@plum.dispatch
def act(
    op: Composed, tau: Any, x: u.AbstractQuantity, /, **kw: object
) -> u.AbstractQuantity:
    """Apply Composed to a Quantity by sequentially applying each transform.

    >>> import unxt as u
    >>> import coordinax.transforms as cxfm

    >>> shift = cxfm.Translate.from_([1, 2, 3], "km")
    >>> pipe = cxfm.Composed((shift,))
    >>> q = u.Q([0, 0, 0], "km")
    >>> cxfm.act(pipe, None, q)
    Q([1, 2, 3], 'km')

    """
    result = x
    for xfm in op.transforms:
        result = cxfmapi.act(xfm, tau, result, **kw)
    return result  # ty: ignore[invalid-return-type]


# ===================================================================
# `prolong` / `pushforward` for Composed


@plum.dispatch
def prolong(
    op: Composed,
    tau: Any,
    jet: dict,
    chart: cxc.AbstractChart,
    /,
    **kw: object,
) -> dict:
    """Prolong a Composed transform: fold the prolongations left-to-right.

    Jets compose by the chain rule, so the prolongation of a composition is
    the composition of the prolongations. Each sub-transform materializes its
    time-dependent parameters exactly once per step.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxfm

    >>> dv = {"x": u.Q(1.0, "km/s"), "y": u.Q(0.0, "km/s"), "z": u.Q(0.0, "km/s")}
    >>> pipe = cxfm.Composed((cxfm.Boost(dv, chart=cxc.cart3d), cxfm.Identity()))
    >>> jet = {0: {"x": u.Q(0.0, "km"), "y": u.Q(0.0, "km"), "z": u.Q(0.0, "km")},
    ...        1: {"x": u.Q(0.0, "km/s"), "y": u.Q(0.0, "km/s"),
    ...            "z": u.Q(0.0, "km/s")}}
    >>> out = cxfm.prolong(pipe, u.Q(2.0, "s"), jet, cxc.cart3d)
    >>> out[0]["x"], out[1]["x"]
    (Q(2., 'km'), Q(1., 'km / s'))

    """
    result = jet
    for sub_op in op.transforms:
        result = cast("dict", cxfmapi.prolong(sub_op, tau, result, chart, **kw))
    return result


@plum.dispatch
def pushforward(
    op: Composed,
    tau: Any,
    v: CDict,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    *,
    at: CDict | None = None,
    **kw: object,
) -> CDict:
    """Pushforward under a Composed transform: fold, advancing the base point."""
    result = v
    current_at = at
    for sub_op in op.transforms:
        result = cxfmapi.pushforward(
            sub_op, tau, result, chart, rep, at=current_at, **kw
        )
        if current_at is not None:
            current_at = cxfmapi.act(sub_op, tau, current_at, chart, cxr.point, **kw)
    return cast("CDict", result)


# ===================================================================
# Simplification


@plum.dispatch
def simplify(op: Composed, /, *, approx: bool = True, **kw: Any) -> AbstractTransform:
    """Simplify a Composed transform.

    Recursively simplifies each sub-operator, strips `Identity` elements, and
    merges adjacent same-type operators in a single left-to-right pass: two
    rotations collapse to one (``R1 @ R2``), two same-role static translations
    to one, and — via the merge plus per-operator identity detection — an
    inverse pair cancels to `Identity`.

    With ``approx=True`` (default) the value-inspecting per-operator collapses
    run too (an identity rotation matrix, a zero translation). With
    ``approx=False`` only value-free structural simplification is done, which is
    always safe to call inside ``jax.jit``.

    >>> import unxt as u
    >>> import coordinax.transforms as cxfm

    Identity elements are stripped:

    >>> shift = cxfm.Translate.from_([1, 2, 3], "km")
    >>> cxfm.simplify(cxfm.Composed((shift, cxfm.Identity())))
    Translate(...)

    Adjacent rotations merge into one:

    >>> R1 = cxfm.Rotate.from_euler("z", u.Q(30, "deg"))
    >>> R2 = cxfm.Rotate.from_euler("z", u.Q(60, "deg"))
    >>> cxfm.simplify(cxfm.Composed((R1, R2)))
    Rotate(...)

    An inverse pair cancels to Identity:

    >>> cxfm.simplify(cxfm.Composed((R1, R1.inverse)))
    Identity()

    """
    # Recursively simplify each sub-op, flattening Composed / dropping Identity.
    ops: list[AbstractTransform] = []
    for sub in op.transforms:
        _extend(ops, simplify(sub, approx=approx, **kw))

    # Merge adjacent pairs left-to-right; re-simplify each merge so a merged
    # identity (e.g. an inverse pair) collapses under approx=True.
    merged: list[AbstractTransform] = []
    for o in ops:
        combined = _merge(merged[-1], o) if merged else None
        if combined is None:
            merged.append(o)
        else:
            merged.pop()
            _extend(merged, simplify(combined, approx=approx, **kw))

    if not merged:
        return identity
    if len(merged) == 1:
        return merged[0]
    return replace(op, transforms=tuple(merged))


def _extend(acc: list[AbstractTransform], op: AbstractTransform, /) -> None:
    """Append ``op`` to ``acc``, flattening `Composed` and dropping `Identity`."""
    if isinstance(op, Composed):
        acc.extend(op.transforms)
    elif not isinstance(op, Identity):
        acc.append(op)


@plum.dispatch
def _merge(a: AbstractTransform, b: AbstractTransform, /) -> AbstractTransform | None:
    """Merge two adjacent operators of a `Composed` into one.

    Returns the combined operator, or `None` if the pair does not combine. Per
    operator-type rules are registered alongside the operators (e.g. Rotate,
    AbstractAdd). ``a`` is applied before ``b``.
    """
    return None
