"""Coordinate: vector bundle anchored at a base point.

A `Coordinate` stores a base `Point` and a named collection of fibre
`Tangent`s (TangentGeometry rep) anchored at that point.  On construction,
every fibre vector is automatically converted into the reference frame of the
base point so the bundle is always internally consistent.
"""

__all__ = ("Coordinate",)


from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView
from typing import Any, cast, final
from typing_extensions import TypeVar, override

import equinox as eqx
import jax.numpy as jnp
import wadler_lindig as wl
from jax.core import ShapedArray

import dataclassish

import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.manifolds as cxm
import coordinax.representations as cxr
import coordinax.transforms as cxfm
from .base import AbstractVector
from .point import Point, _vector_comps_unit_docs, _vector_values_str
from .tangent import Tangent, _vectorform_pdoc as _vec_vectorform_pdoc
from coordinax.internal import OptUSys

ChartT = TypeVar(
    "ChartT", bound=cxc.AbstractChart[Any, Any], default=cxc.AbstractChart[Any, Any]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _broadcast_shapes(shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
    """Return the broadcast shape of a list of shapes."""
    if not shapes:
        return ()
    result = shapes[0]
    for s in shapes[1:]:
        result = jnp.broadcast_shapes(result, s)
    return result


def vectorform_pdoc(pv: "Coordinate", **kwargs: Any) -> wl.AbstractDoc:
    """Return the vector-form Wadler-Lindig document for a `Coordinate`."""
    kwargs.setdefault("canonical", True)
    chart_name = type(pv.point.chart).__name__
    rep_name = wl.pformat(pv.point.rep, **kwargs)
    comps_doc, unit_doc = _vector_comps_unit_docs(pv.point)
    values_str = _vector_values_str(pv.point, **kwargs)

    header = f"<Coordinate: chart={chart_name}, rep={rep_name} {comps_doc}"
    if unit_doc:
        header = f"{header} {unit_doc}"

    if pv._data:

        def _embedded_vectorform(vec: Tangent) -> str:
            rendered = wl.pformat(_vec_vectorform_pdoc(vec, **kwargs))
            if rendered.startswith("<") and rendered.endswith(">"):
                return rendered[1:-1]
            return rendered

        field_lines = [
            f"  {name}={_embedded_vectorform(vec)}" for name, vec in pv._data.items()
        ]
        return wl.TextDoc(header + values_str + "\n" + "\n".join(field_lines) + ">")
    return wl.TextDoc(header + values_str + ">")


# ---------------------------------------------------------------------------
# Coordinate
# ---------------------------------------------------------------------------


@final
class Coordinate(AbstractVector):
    r"""A vector bundle anchored at a base point.

    A `Coordinate` stores:

    - A base **point** $q \in M$ (a `~coordinax.vectors.Point`).
    - A collection of named **fibre vectors** $\{v_i\}$ anchored at $q$
      (each a `~coordinax.vectors.Tangent` with ``TangentGeometry`` rep,
      e.g. velocity, displacement, acceleration).

    On construction every fibre vector is automatically frame-aligned to the
    **reference frame** of the base point:

    1. Frame-alignment via `~coordinax.vectors.AbstractVector.to_frame`
       ensures ``pv["velocity"].frame == pv.point.frame``.

    Fibre vectors are **not** chart-aligned on construction; each fibre
    retains the chart it was supplied with.  Chart conversion is handled
    lazily: `~coordinax.vectors.Coordinate.cconvert` pushes each fibre
    forward using the Jacobian at the base point expressed in the fibre's
    current chart.

    Coordinate conversion (chart change) is handled automatically: the base
    converts as a point map, and each fibre vector converts via the Jacobian
    pushforward at the base.

    Parameters
    ----------
    point : Point
        Base point. Must be an instance of `~coordinax.vectors.Point`.
    **fields : Tangent
        Named fibre vectors anchored at ``point``.  Must have
        ``TangentGeometry`` representation (i.e. `~coordinax.vectors.Tangent`
        instances).  Shapes must be broadcastable with ``point``.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    >>> point = cx.Point.from_([1.0, 0.0, 0.0], "m")
    >>> vel = cx.Tangent.from_(
    ...     {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
    ...     cxc.cart3d, cxr.coord_vel)
    >>> pv = cx.Coordinate(point=point, velocity=vel)
    >>> pv.point.chart
    Cart3D(M=Rn(3))

    Convert to spherical — point converts as a point map, velocity via Jacobian:

    >>> pv_sph = pv.cconvert(cxc.sph3d)
    >>> pv_sph.point.chart
    Spherical3D(M=Rn(3))
    >>> pv_sph["velocity"].chart
    Spherical3D(M=Rn(3))

    """

    point: Point
    """Base point of the bundle. Must be a ``Point`` instance."""

    _data: dict[str, Tangent] = eqx.field(repr=False)
    """Fibre vectors (fields) anchored at the point. Excluded from repr."""

    def __init__(
        self,
        /,
        point: Any,  # Any so our isinstance check fires before beartype
        **fields: Any,
    ) -> None:
        """Initialise a Coordinate.

        Parameters
        ----------
        point : Point
            Base point. Must be a ``Point`` instance.
        **fields : Tangent
            Named fibre vectors. Each must be a ``Tangent`` instance with
            ``TangentGeometry`` representation.  Each field is automatically
            frame-aligned to ``point`` on construction; the chart of each
            fibre is preserved as supplied.

        """
        # --- Validate: point must be a Point instance ---
        if not isinstance(point, Point):
            msg = (
                "Coordinate: point must be a Point instance, "
                f"got {type(point).__name__!r}"
            )
            raise TypeError(msg)

        # --- Validate and (frame-align) fields ---
        field_vecs: dict[str, Tangent] = {}
        target_frame = point.frame
        for name, val in fields.items():
            # Fields must be Tangent (not Point or arbitrary objects)
            if not isinstance(val, Tangent):
                msg = (
                    f"Coordinate: field '{name}' must be a Tangent instance, "
                    f"got {type(val).__name__!r}"
                )
                raise TypeError(msg)
            vec: Tangent = val

            # Convert to point's frame using act() directly so we can supply
            # the base-point anchor (at=) needed by non-Cartesian tangent
            # frame transforms (e.g. Rotate on TangentGeometry requires 'at'
            # to evaluate the Jacobian pushforward).  The Identity fast-path
            # avoids any JAX tracing overhead when frames already match.
            op = vec.frame.frame_transition(target_frame)
            if not isinstance(op, cxfm.Identity):
                # Express the base point in vec's current chart so that act()
                # can evaluate the Jacobian at the correct location.
                at_point = cast(
                    "Point", cxr.cconvert(point.to_frame(vec.frame), vec.chart)
                )
                vec = dataclassish.replace(
                    cast("Tangent", cxfm.act(op, None, vec, at=at_point.data)),
                    frame=target_frame,
                )  # ty: ignore[invalid-assignment]

            field_vecs[name] = vec

        # --- Validate broadcastable shapes ---
        all_shapes = [point.shape, *(v.shape for v in field_vecs.values())]
        if len(all_shapes) > 1:
            try:
                jnp.broadcast_shapes(*all_shapes)
            except Exception as exc:
                msg = f"Coordinate: shapes {all_shapes} are not broadcastable: {exc}"
                raise ValueError(msg) from exc

        # Bypass equinox's immutable __setattr__
        self.__dict__["point"] = point
        self.__dict__["_data"] = field_vecs

    @classmethod
    def _create_unchecked(
        cls,
        point: Point,
        fields: dict[str, Tangent],
    ) -> "Coordinate":
        """Create a ``Coordinate`` bypassing frame/chart alignment and validation.

        For **internal use only**.  Callers must guarantee that ``point`` and
        every value in ``fields`` already have consistent types and shapes.
        """
        obj: Coordinate = object.__new__(cls)
        obj.__dict__["point"] = point
        obj.__dict__["_data"] = fields
        return obj

    # ===================================================================
    # AbstractVector abstract attribute satisfaction (delegate to point)

    @property
    def data(self) -> Any:
        """Component data of the base point."""
        return self.point.data

    @property
    def chart(self) -> cxc.AbstractChart:
        """Chart of the base point."""
        return self.point.chart

    @property
    def rep(self) -> cxr.Representation:
        """Representation of the base point (always PointGeometry)."""
        return self.point.rep  # ty: ignore[invalid-return-type]

    @property
    def manifold(self) -> cxm.AbstractManifold:
        """Manifold of the base point."""
        return self.point.M

    @property
    def frame(self) -> cxf.AbstractReferenceFrame:
        """Reference frame of the bundle — always equal to ``point.frame``."""
        return self.point.frame

    # ===================================================================
    # Mapping interface (over fields only, not base)

    @override
    def __getitem__(self, key: Any) -> "Tangent | Coordinate":  # ty: ignore[invalid-method-override]
        """Get a named field vector or batch-index the bundle.

        Parameters
        ----------
        key : str or index
            If ``str``, return the named field vector.
            Otherwise, batch-index all component arrays (base + all fields).

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax.main as cx
        >>> import coordinax.charts as cxc
        >>> import coordinax.representations as cxr

        >>> base = cx.Point.from_([1.0, 0.0, 0.0], "m")
        >>> vel = cx.Tangent.from_(
        ...     {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
        ...     cxc.cart3d, cxr.coord_vel)
        >>> pv = cx.Coordinate(point=base, velocity=vel)
        >>> isinstance(pv["velocity"], cx.Tangent)
        True

        """
        if isinstance(key, str):
            return self._data[key]

        # Batch-indexing: delegate to Point/Tangent indexing so invalid
        # indices raise consistently instead of silently skipping.
        new_point = self.point[key]
        new_fields = {k: v[key] for k, v in self._data.items()}
        return Coordinate._create_unchecked(new_point, new_fields)

    def keys(self) -> KeysView[str]:
        """Return field names (excluding base point).

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax.main as cx
        >>> import coordinax.charts as cxc
        >>> import coordinax.representations as cxr

        >>> base = cx.Point.from_([1.0, 0.0, 0.0], "m")
        >>> vel = cx.Tangent.from_(
        ...     {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
        ...     cxc.cart3d, cxr.coord_vel)
        >>> pv = cx.Coordinate(point=base, velocity=vel)
        >>> list(pv.keys())
        ['velocity']

        """
        return self._data.keys()

    def values(self) -> ValuesView[Tangent]:
        """Return field vectors (excluding base point)."""
        return self._data.values()

    def items(self) -> ItemsView[str, Tangent]:
        """Return ``(name, vector)`` pairs for fields (excluding base point)."""
        return self._data.items()

    def __len__(self) -> int:
        """Return number of fibre field vectors."""
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        """Iterate over field names."""
        return iter(self._data)

    # ===================================================================
    # Coordinate conversion

    def cconvert(
        self,
        to_chart: cxc.AbstractChart,
        /,
        *,
        field_charts: Mapping[str, cxc.AbstractChart] | None = None,
        usys: OptUSys = None,
    ) -> "Coordinate":
        r"""Convert the bundle to a new coordinate chart.

        Algorithm:

        1. Convert base as a point map: ``new_point = cconvert(point, to_chart)``.
        2. For each field vector, apply the tangent pushforward at ``point``
           via ``cconvert(vec, field_to_chart, at=point)``.

        Parameters
        ----------
        to_chart : AbstractChart
            Target chart for the base and (by default) all fields.
        field_charts : Mapping[str, AbstractChart], optional
            Per-field target chart overrides.
        usys : UnitSystem, optional
            Unit system for the conversion.

        Examples
        --------
        >>> import coordinax.main as cx
        >>> import coordinax.charts as cxc
        >>> import unxt as u
        >>> import coordinax.representations as cxr

        >>> point = cx.Point.from_([1.0, 0.0, 0.0], "m")
        >>> vel = cx.Tangent.from_(
        ...     {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
        ...     cxc.cart3d, cxr.coord_vel)
        >>> pv = cx.Coordinate(point=point, velocity=vel)
        >>> sph = pv.cconvert(cxc.sph3d)
        >>> sph.point.chart
        Spherical3D(M=Rn(3))
        >>> sph["velocity"].chart
        Spherical3D(M=Rn(3))

        """
        if field_charts is None:
            field_charts = {}

        # 1. Convert base point (pure point map — no Jacobian needed)
        new_point = cast("Point", cxr.cconvert(self.point, to_chart, usys=usys))

        # 2. Convert each field via tangent pushforward at self.point.
        # Express self.point in each fibre's current chart for the Jacobian;
        # this handles fibres that are in a different chart than self.point.
        new_fields: dict[str, Tangent] = {}
        for name, vec in self._data.items():
            target = field_charts.get(name, to_chart)
            at = (
                self.point
                if vec.chart == self.point.chart
                else cast("Point", cxr.cconvert(self.point, vec.chart))
            )
            new_fields[name] = cast(
                "Tangent",
                cxr.cconvert(vec, target, at=at, usys=usys),
            )

        # Use _create_unchecked to bypass frame re-alignment in __init__
        # (the results are already in the correct frame).
        return Coordinate._create_unchecked(new_point, new_fields)

    # ===================================================================
    # AbstractVector — shape

    @property
    def shape(self) -> tuple[int, ...]:
        """Broadcast shape of base point and all field vectors.

        Examples
        --------
        >>> import coordinax.main as cx
        >>> pv = cx.Coordinate(point=cx.Point.from_([1.0, 2.0, 3.0], "m"))
        >>> pv.shape
        ()

        """
        all_shapes = [self.point.shape, *(v.shape for v in self._data.values())]
        return _broadcast_shapes(all_shapes)

    # ===================================================================
    # Quax API

    def aval(self) -> ShapedArray:
        """Return abstract JAX array value for tracing.

        The shape is ``(*batch, total_components)`` where ``total_components``
        is the sum of components across the base `Point` and every fibre
        `Tangent`.  This is consistent with `Point.aval` / `Tangent.aval`
        (which return ``(*batch, n_components)``) and reflects the full
        flattened array that a ``Coordinate`` bundle conceptually represents.

        The dtype is the promoted dtype across all held fields.

        Examples
        --------
        >>> import coordinax.main as cx

        >>> point = cx.Point.from_([1.0, 2.0, 3.0], "m")
        >>> pv = cx.Coordinate(point=point)
        >>> pv.aval()  # doctest: +ELLIPSIS
        ShapedArray(float...[3])

        A ``Coordinate`` with one velocity field (3 + 3 = 6 total components):

        >>> import unxt as u
        >>> import coordinax.charts as cxc
        >>> import coordinax.representations as cxr
        >>> vel = cx.Tangent.from_(
        ...     {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
        ...     cxc.cart3d, cxr.coord_vel,
        ... )
        >>> pv2 = cx.Coordinate(point=point, velocity=vel)
        >>> pv2.aval()  # doctest: +ELLIPSIS
        ShapedArray(float...[6])

        """
        all_vecs = [self.point, *self._data.values()]
        avals = [v.aval() for v in all_vecs]
        dtype = jnp.result_type(*[a.dtype for a in avals])
        batch = self.shape
        total_components = sum(a.shape[-1] for a in avals)
        return ShapedArray((*batch, total_components), dtype)

    # ===================================================================
    # Wadler-Lindig API

    def __pdoc__(self, *, vector_form: bool = False, **kwargs: Any) -> wl.AbstractDoc:
        """Return the Wadler-Lindig document for a `Coordinate`.

        Examples
        --------
        >>> import unxt as u
        >>> import wadler_lindig as wl
        >>> import coordinax.main as cx

        >>> point = cx.Point.from_([1.0, 0.0, 0.0], "m")
        >>> vel = cx.Tangent.from_(
        ...     {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
        ...     cx.cart3d, cx.coord_vel,
        ... )
        >>> coord = cx.Coordinate(point=point, velocity=vel)

        The standard document renders as a constructor-style representation:

        >>> wl.pprint(coord)
        Coordinate(
          Point(
            {'x': Q(f64[], 'm'), 'y': Q(f64[], 'm'), 'z': Q(f64[], 'm')},
            chart=Cart3D(M=Rn(3))
          ),
          velocity=Tangent(
            { 'x': Q(weak_f64[], 'm / s'), 'y': Q(weak_f64[], 'm / s'),
              'z': Q(weak_f64[], 'm / s') },
            chart=Cart3D(M=Rn(3)), basis=coord_basis, semantic=vel
          )
        )

        The vector form renders as the compact angle-bracket representation:

        >>> wl.pprint(coord, vector_form=True)
        <Coordinate: chart=Cart3D, rep=point (x, y, z) [m]
                [1. 0. 0.]
            velocity=Tangent: chart=Cart3D (x, y, z) [m / s]
                [1. 0. 0.]>

        """
        if vector_form:
            return vectorform_pdoc(self, **kwargs)

        kwargs.setdefault("use_short_name", True)
        kwargs.setdefault("named_unit", False)
        docs = [
            wl.pdoc(self.point, **kwargs),
            *wl.named_objs(self._data.items(), **kwargs),
        ]
        return wl.bracketed(
            begin=wl.TextDoc("Coordinate("),
            docs=docs,
            sep=wl.comma,
            end=wl.TextDoc(")"),
            indent=kwargs.get("indent", 4),
        )

    def __repr__(self) -> str:
        return wl.pformat(self, vector_form=False, short_arrays="compact")

    def __str__(self) -> str:
        return wl.pformat(self, vector_form=True, precision=3)


# ===========================================================================
# from_ dispatches
# ===========================================================================


@Coordinate.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Coordinate], pv: Coordinate, /) -> Coordinate:
    """Identity: return the same Coordinate unchanged.

    Examples
    --------
    >>> import coordinax.main as cx
    >>> pv = cx.Coordinate(point=cx.Point.from_([1.0, 2.0, 3.0], "m"))
    >>> cx.Coordinate.from_(pv) is pv
    True

    """
    return pv


@Coordinate.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Coordinate], p: Point, /) -> Coordinate:
    """Wrap a single ``Point`` as a point-only bundle (no field vectors).

    Examples
    --------
    >>> import coordinax.main as cx
    >>> p = cx.Point.from_([1.0, 2.0, 3.0], "m")
    >>> pv = cx.Coordinate.from_(p)
    >>> pv.point is p
    True

    """
    return cls(point=p)


@Coordinate.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Coordinate],
    data: Mapping[str, Any],
    /,
    *,
    point: Point | None = None,
) -> Coordinate:
    """Create a ``Coordinate`` from a mapping of named objects.

    The mapping may contain a ``"point"`` key for the base; the explicit
    ``point`` keyword argument takes precedence if both are supplied.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    >>> p = cx.Point.from_([1.0, 2.0, 3.0], "m")
    >>> vel = cx.Tangent.from_(
    ...     {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
    ...     cxc.cart3d, cxr.coord_vel)
    >>> pv = cx.Coordinate.from_({"point": p, "velocity": vel})
    >>> pv.point is p
    True

    """
    data_dict = dict(data)

    if point is None:
        point = data_dict.pop("point", None)
    else:
        data_dict.pop("point", None)

    if point is None:
        msg = (
            "Coordinate.from_: 'point' must be provided in the mapping "
            "or as a keyword argument."
        )
        raise ValueError(msg)

    return cls(point=point, **data_dict)
