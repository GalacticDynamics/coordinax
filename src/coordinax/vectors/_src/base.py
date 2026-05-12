"""Abstract Vector."""

__all__ = ("AbstractVector",)

import abc
import dataclasses

from typing import TYPE_CHECKING, Any, Generic, NoReturn, cast
from typing_extensions import TypeIs, TypeVar, override

import equinox as eqx
import jax
import jax.tree as jtu
import plum
import quax_blocks
import wadler_lindig as wl
from quax import ArrayValue

import dataclassish
import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items

import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.manifolds as cxm
import coordinax.representations as cxr
import coordinax.transforms as cxfm
from .custom_types import HasShape

if TYPE_CHECKING:
    from typing import Self

Ks = TypeVar("Ks", bound=tuple[str, ...])
Ds = TypeVar("Ds", bound=tuple[str | None, ...])
ChartT = TypeVar(
    "ChartT", bound=cxc.AbstractChart[Any, Any], default=cxc.AbstractChart[Any, Any]
)
GeomT = TypeVar("GeomT", bound=cxr.AbstractGeometry, default=cxr.AbstractGeometry)
BasisT = TypeVar("BasisT", bound=cxr.AbstractBasis, default=cxr.AbstractBasis)
SemanticT = TypeVar(
    "SemanticT", bound=cxr.AbstractSemanticKind, default=cxr.AbstractSemanticKind
)
V = TypeVar("V", bound=HasShape, default=u.Q)


class AbstractVector(
    ArrayValue,
    quax_blocks.LaxBinaryOpsMixin[Any, Any],  # TODO: type annotation
    quax_blocks.LaxRoundMixin["AbstractVector"],
    quax_blocks.LaxUnaryMixin[Any],
    Generic[ChartT, GeomT, BasisT, SemanticT, V],
):
    """Abstract base class for all vector-like objects in coordinax.

    ``AbstractVector`` binds three pieces of geometric information — data,
    chart, and representation — into a single JAX-compatible, immutable object
    that can represent points, tangent vectors, or higher-order tensors on a
    smooth manifold.

    Concretely, a vector stores:

    - **data** — a mapping from component name to scalar leaves (typically
      `unxt.Quantity`), one entry per coordinate axis.
    - **chart** — an `~coordinax.charts.AbstractChart` that names the
      coordinates, records their physical dimensions, and knows how to
      transition to every other chart in the same atlas.
    - **rep** — a `~coordinax.representations.Representation` that encodes
      the *geometric kind* of the vector (e.g. base-manifold point, tangent
      displacement, physical-basis velocity) and therefore the correct
      transformation law under chart changes.

    All concrete subclasses are immutable Equinox PyTrees and
    `quax.ArrayValue` subclasses.  Arithmetic operations (``+``, ``-``,
    ``*``, …) are handled via Quax dispatch over JAX primitives so that
    ``jit``, ``vmap``, and ``grad`` all work transparently.

    Attributes
    ----------
    data : Any
        Mapping from chart component name to scalar value.  Each leaf is
        typically a `unxt.Quantity`; components are expected to be scalar
        leaves so that batching is achieved through JAX broadcasting.
    chart : ChartT
        The chart instance (e.g. `coordinax.charts.cart3d`) that defines the
        component schema and coordinate-system geometry.
    rep : coordinax.representations.Representation
        The representation (e.g. `coordinax.representations.point`) that
        selects the transformation semantics for chart conversions.
    M : coordinax.manifolds.AbstractManifold
        The manifold on which the vector lives.
    shape : tuple[int, ...]
        The batch shape of the vector (abstract; implemented by subclasses).

    Methods
    -------
    from_(*args, **kwargs) -> AbstractVector
        Multiple-dispatch constructor.  Dispatches are registered externally
        via ``plum``; call ``.methods`` to inspect all overloads.
    cconvert(*args, **kwargs) -> AbstractVector
        Convert to another chart or representation, forwarding to
        `coordinax.representations.cconvert`.
    to_cartesian() -> AbstractVector
        Shorthand for ``self.cconvert(self.chart.cartesian)``.
    uconvert(*args, **kwargs) -> AbstractVector
        Convert component units by forwarding to `unxt.uconvert`.
    astype(dtype, **kwargs) -> AbstractVector
        Cast all component leaves to a new dtype.
    copy() -> Self
        Shallow copy via `dataclasses.replace`.
    flatten() -> Self
        Flatten all component leaves.
    ravel() -> Self
        Return a flattened copy (alias of ``flatten``).
    reshape(*shape) -> Self
        Return a reshaped copy.
    round(decimals=0) -> Self
        Return a rounded copy.
    to_device(device=None) -> Self
        Move all leaves to the specified JAX device.
    is_like(obj) -> TypeIs[Self]
        Class method; return ``True`` if *obj* is an instance of this class.

    See Also
    --------
    coordinax.vectors.Point : Concrete default implementation.
    coordinax.vectors.AbstractCoordinate : Vectors bound to a reference frame.
    coordinax.representations.cconvert : Chart/representation conversion.
    coordinax.charts.AbstractChart : Chart objects defining coordinate systems.

    Notes
    -----
    **Immutability:** ``__setitem__`` raises ``TypeError``; use
    `dataclassish.replace` to derive modified copies.

    **Not supported:** ``materialise()`` (raises ``RuntimeError``),
    ``__complex__``, ``__float__``, ``__int__``, ``__index__`` (all raise
    ``NotImplementedError``), ``__hash__`` (raises ``TypeError`` in practice
    because JAX arrays are not hashable).

    **Dispatch:** ``from_``, ``cconvert``, and ``uconvert`` are all
    ``plum``-dispatched.  To inspect all registered overloads at runtime,
    call ``.methods`` on the function object.

    Examples
    --------
    Concrete instances are created through `coordinax.vectors.Point`:

    >>> import coordinax.vectors as cxv  # AbstractVector not in main
    >>> import coordinax.main as cx

    >>> vec = cxv.Point.from_([1, 2, 3], "m")
    >>> print(vec)
    <Point: chart=Cart3D (x, y, z) [m]
        [1 2 3]>

    >>> cxv.AbstractVector.is_like(vec)
    True

    Convert to spherical coordinates:

    >>> print(vec.cconvert(cx.sph3d))
    <Point: chart=Spherical3D (r[m], theta[rad], phi[rad])
        [3.742 0.641 1.107]>

    Convert units:

    >>> print(vec.uconvert({u.dimension("length"): "km"}))
    <Point: chart=Cart3D (x, y, z) [km]
        [0.001 0.002 0.003]>

    Arithmetic works under ``jit`` and ``vmap``:

    >>> import jax
    >>> print(jax.jit(lambda v: v * 2)(vec))
    <Point: chart=Cart3D (x, y, z) [m]
        [2 4 6]>

    """

    data: eqx.AbstractVar[Any]
    """The data for each component."""

    chart: eqx.AbstractVar[ChartT]
    """The chart of the vector, e.g. `cxc.cart3d`."""

    rep: eqx.AbstractVar[cxr.Representation[GeomT, BasisT, SemanticT]]
    """The `coordinax.representations.Representation`, e.g. `cxr.point`."""

    frame: eqx.AbstractVar[cxf.AbstractReferenceFrame]
    """The reference frame of the point. Defaults to ``cxf.noframe``."""

    # ---------------------------------
    # Constructors

    @classmethod
    @plum.dispatch.abstract
    def from_(
        cls: "type[AbstractVector]", *args: Any, **kwargs: Any
    ) -> "AbstractVector":
        """Create a vector-like object from arguments."""
        raise NotImplementedError  # pragma: no cover

    # ===============================================================
    # Vector API

    @property
    def M(self) -> cxm.AbstractManifold:
        """The manifold of the vector, from the chart."""
        return self.chart.M

    @plum.dispatch
    def cconvert(
        self, *args: Any, **kwargs: Any
    ) -> Any:  # TODO: return type annotation
        """Represent the vector as another type.

        This forwards to `coordinax.representations.cconvert`.

        Examples
        --------
        >>> import coordinax.main as cx

        >>> vec = cx.Point.from_([1, 2, 3], "m")
        >>> print(vec)
        <Point: chart=Cart3D (x, y, z) [m]
            [1 2 3]>

        >>> print(vec.cconvert(cx.sph3d))
        <Point: chart=Spherical3D (r[m], theta[rad], phi[rad])
            [3.742 0.641 1.107]>

        """
        return cast("AbstractVector", cxr.cconvert(self, *args, **kwargs))

    def to_cartesian(self) -> "AbstractVector":
        """Return the vector in a Cartesian chart.

        This just forwards to `coordinax.cartesian_chart` and `cconvert`.

        Examples
        --------
        >>> import coordinax.main as cx

        >>> vec = cx.Point.from_([1, 2, 3], "m").cconvert(cx.sph3d)
        >>> print(vec)
        <Point: chart=Spherical3D (r[m], theta[rad], phi[rad])
            [3.742 0.641 1.107]>

        >>> print(vec.to_cartesian())
        <Point: chart=Cart3D (x, y, z) [m]
            [1. 2. 3.]>

        """
        return self.cconvert(self.chart.cartesian)

    # ===============================================================
    # Quantity API

    @plum.dispatch(precedence=-1)  # ty: ignore[no-matching-overload]
    def uconvert(
        self, *args: Any, **kwargs: Any
    ) -> Any:  # TODO: return type annotation
        """Convert the vector to the given units.

        This just forwards to `unxt.uconvert`, reversing the order of the
        arguments to match the `unxt` API.

        Examples
        --------
        >>> import coordinax.main as cx

        >>> vec = cx.Point.from_([1, 2, 3], "km")
        >>> print(vec.uconvert({"length": "km"}))
        <Point: chart=Cart3D (x, y, z) [km]
            [1 2 3]>

        """
        return cast("AbstractVector", u.uconvert(*args, self, **kwargs))

    @plum.dispatch
    def uconvert(
        self, usys: u.AbstractUnitSystem, /
    ) -> Any:  # TODO: return type annotation
        """Convert the vector to the given units.

        Parameters
        ----------
        usys
            The units to convert to according to the physical type of the
            components. This is passed to [`unxt.unitsystem`][].

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax.main as cx

        >>> usys = u.unitsystem("m", "s", "kg", "rad")

        >>> vec = cx.Point.from_([1, 2, 3], "km")

        >>> print(vec.uconvert(usys))
        <Point: chart=Cart3D (x, y, z) [m]
            [1000. 2000. 3000.]>

        >>> print(vec.uconvert("galactic"))
        <Point: chart=Cart3D (x, y, z) [kpc]
            [3.241e-17 6.482e-17 9.722e-17]>

        """
        return cast("AbstractVector", u.uconvert(usys, self))

    # ===============================================================
    # Quax API

    def materialise(self) -> NoReturn:
        """Materialise the vector for `quax`.

        Examples
        --------
        >>> import coordinax.main as cx
        >>> vec = cx.Point.from_([1, 2, 3], "m")

        >>> try: vec.materialise()
        ... except RuntimeError as e: print(e)
        Refusing to materialise `Point`.

        """
        msg = f"Refusing to materialise `{type(self).__name__}`."
        raise RuntimeError(msg)

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the vector."""
        raise NotImplementedError  # pragma: no cover

    # ===============================================================
    # Array API

    def __array_namespace__(self) -> Any:
        """Return the array API namespace.

        Here we return the `quaxed.numpy` module, which is a drop-in replacement
        for `jax.numpy`, but allows for array-ish objects to be used in place of
        `jax` arrays. See `quax` for more information.

        Examples
        --------
        >>> import coordinax.main as cx
        >>> vec = cx.Point.from_([3, 4], "m")
        >>> ns = vec.__array_namespace__()
        >>> ns
        <module 'quaxed.numpy' from ...>

        >>> _ = ns.multiply(vec, 2)

        """
        return jnp

    # ---------------------------------
    # comparison operators

    # TODO: use quax_blocks.LaxEqMixin
    @override
    def __eq__(self: "AbstractVector", other: object, /) -> Any:
        """Check if the vector is equal to another object."""
        # Check type equality first
        if type(other) is not type(self):
            return NotImplemented

        # Delegate to `quax` primitives
        return jnp.equal(self, cast("AbstractVector", other))

    # ---------------------------------
    # methods

    def __complex__(self) -> NoReturn:
        """Convert a zero-dimensional object to a Python complex object."""
        raise NotImplementedError  # pragma: no cover

    # TODO: .__dlpack__, __dlpack_device__

    def __float__(self) -> NoReturn:
        """Convert a zero-dimensional object to a Python float object."""
        raise NotImplementedError  # pragma: no cover

    @abc.abstractmethod
    def __getitem__(self, index: Any) -> "Self":
        """Return a new object with the given slice applied.

        Parameters
        ----------
        index
            The slice to apply.

        """
        raise NotImplementedError  # pragma: no cover

    def __index__(self) -> NoReturn:
        """Convert the vector to an integer index."""
        raise NotImplementedError  # pragma: no cover

    def __int__(self) -> NoReturn:
        """Convert the vector to an integer."""
        raise NotImplementedError  # pragma: no cover

    def __setitem__(self, k: Any, v: Any) -> NoReturn:
        """Vectors are immutable.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax.main as cx

        We can't set an item in a vector:

        >>> vec = cx.Point.from_(u.Q([[1, 2], [3, 4]], "m"))
        >>> try: vec[0] = u.Q(1, "m")
        ... except TypeError as e: print(e)
        Point is immutable.

        """
        msg = f"{type(self).__name__} is immutable."
        raise TypeError(msg)

    # ===============================================================
    # JAX API

    # TODO: repeat(), round(), sort(), squeeze(), swapaxes(), transpose(),
    # view() addressable_shards, at, committed, globarl_shards,
    # is_fully_addressable, is_fully_replcated, nbytes, sharding

    def astype(self, dtype: Any, /, **kwargs: Any) -> "AbstractVector":
        """Cast the vector to a new dtype.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import unxt as u
        >>> import coordinax.main as cx

        We can cast a vector to a new dtype:

        >>> vec = cx.Point.from_(u.Q([1, 2, 3], "m"))
        >>> print(vec.astype(jnp.float32))
        <Point: chart=Cart3D (x, y, z) [m]
            [1. 2. 3.]>

        >>> print(jnp.astype(vec, jnp.float32))
        <Point: chart=Cart3D (x, y, z) [m]
            [1. 2. 3.]>

        """
        dynamic, static = eqx.partition(self, lambda x: hasattr(x, "astype"))
        dynamic = jtu.map(lambda x: x.astype(dtype, **kwargs), dynamic)
        return eqx.combine(dynamic, static)

    # -------------------------------

    def copy(self) -> "Self":
        """Return a copy of the vector.

        Examples
        --------
        >>> import coordinax.main as cx

        >>> vec = cx.Point.from_([1, 2, 3], "m")
        >>> print(vec.copy())
        <Point: chart=Cart3D (x, y, z) [m]
            [1 2 3]>

        """
        return dataclasses.replace(self)

    def flatten(self) -> "Self":
        """Flatten the vector."""
        dynamic, static = eqx.partition(self, lambda x: hasattr(x, "flatten"))
        dynamic = jtu.map(lambda x: x.flatten(), dynamic)
        return eqx.combine(dynamic, static)

    def ravel(self) -> "Self":
        """Return a flattened vector."""
        dynamic, static = eqx.partition(self, lambda x: hasattr(x, "ravel"))
        dynamic = jtu.map(lambda x: x.ravel(), dynamic)
        return eqx.combine(dynamic, static)

    def reshape(self, *shape: int) -> "Self":
        """Return a reshaped vector."""
        dynamic, static = eqx.partition(self, lambda x: hasattr(x, "reshape"))
        dynamic = jtu.map(lambda x: x.reshape(*shape), dynamic)
        return eqx.combine(dynamic, static)

    def round(self, decimals: int = 0) -> "Self":
        """Return a rounded vector."""
        dynamic, static = eqx.partition(self, lambda x: hasattr(x, "round"))
        dynamic = jtu.map(lambda x: x.round(decimals), dynamic)
        return eqx.combine(dynamic, static)

    def to_device(self, device: None | jax.Device = None) -> "Self":
        """Move the vector to a new device."""
        dynamic, static = eqx.partition(self, lambda x: hasattr(x, "to_device"))
        dynamic = jtu.map(lambda x: x.to_device(device), dynamic)
        return eqx.combine(dynamic, static)

    # ===============================================================
    # Python API

    def __hash__(self) -> int:
        """Return the hash of the vector.

        This is the hash of the fields, however since jax arrays are
         not hashable this will generally raise an exception.
        Defining the `__hash__` method is required for the vector to
        be considered immutable, e.g. by `dataclasses.dataclass`.

        Examples
        --------
        >>> import coordinax.main as cx
        >>> vec = cx.Point.from_([1, 2, 3], "m")
        >>> try:
        ...     hash(vec)
        ... except TypeError as e:
        ...     print(e)
        unhashable type: 'dict'

        """
        return hash(tuple(field_items(self)))

    def __repr__(self) -> str:
        """Return a string representation of the vector.

        This uses the `equinox.tree_pformat` function to format the vector,
        which internally uses the `wadler_lindig` algorithm to format the string
        representation of the vector.

        """
        return wl.pformat(self, vector_form=False, short_arrays="compact")

    def __str__(self) -> str:
        """Return a string representation of the vector-like object."""
        return wl.pformat(self, vector_form=True, precision=3)

    # ===============================================================
    # Convenience methods

    @classmethod
    def is_like(cls, obj: Any, /) -> TypeIs["Self"]:
        """Check if the object is a `AbstractVector` object.

        Examples
        --------
        >>> import coordinax.vectors as cxv

        >>> vec = cxv.Point.from_([1, 2, 3], "m")
        >>> cxv.AbstractVector.is_like(vec)
        True

        >>> cxv.AbstractVector.is_like(42)
        False

        """
        return isinstance(obj, cls)

    def norm(self, *args: "AbstractVector") -> u.AbstractQuantity:
        msg = "TODO"
        raise NotImplementedError(msg)
        # return self.chart.norm(self.data, *args)

    # ===============================================================
    # Frame API

    def to_frame(
        self, toframe: cxf.AbstractReferenceFrame, /, t: u.Q | None = None
    ) -> "AbstractVector":
        """Transform the vector to a specified reference frame.

        Parameters
        ----------
        toframe : AbstractReferenceFrame
            The target reference frame.
        t : Quantity, optional
            The evolution parameter (e.g. time). Defaults to 0 s.

        Returns
        -------
        AbstractVector
            New vector with the data transformed into ``toframe`` and
            ``frame=toframe``.

        Examples
        --------
        >>> import coordinax.main as cx
        >>> import coordinax.frames as cxf

        >>> p = cx.Point.from_([1, 2, 3], "kpc", cxf.alice)
        >>> p.to_frame(cxf.alice) is p
        True

        """
        op = self.frame.frame_transition(toframe)

        # Special case for identity operations (same frame)
        if isinstance(op, cxfm.Identity):
            return self  # ty: ignore[invalid-return-type]

        # Otherwise, apply the transformation and return a new point
        new = cxfm.act(op, t, self)
        return dataclassish.replace(new, frame=toframe)  # ty: ignore[invalid-return-type]
