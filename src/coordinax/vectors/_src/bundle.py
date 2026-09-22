"""Coordinate: vector bundle anchored at a base point.

A `Coordinate` stores a base `Point` and a named collection of fibre
`Tangent`s (TangentGeometry rep) anchored at that point.  On construction,
every fibre vector is automatically converted into the reference frame of the
base point so the bundle is always internally consistent.
"""

__all__ = ("Coordinate",)


from collections.abc import Callable, ItemsView, Iterator, KeysView, Mapping, ValuesView
from typing import Any, cast, final, override
from typing_extensions import TypeVar

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
import coordinax.transforms._src.actions.utils as cxfm_utils
from .base import (
    AbstractVector,
    broadcast_and_index_data,
    vector_comps_unit_docs,
    vector_values_str,
    vectorform_pdoc as _vec_vectorform_pdoc,
)
from .point import Point
from .tangent import Tangent
from coordinax.internal import OptUSys
from coordinax.transforms._src.actions.prolong import prolong_point_map

ChartT = TypeVar(
    "ChartT",
    bound=cxc.AbstractChart[Any, Any, Any],
    default=cxc.AbstractChart[Any, Any, Any],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def vectorform_pdoc(pv: "Coordinate", **kwargs: Any) -> wl.AbstractDoc:
    """Return the vector-form Wadler-Lindig document for a `Coordinate`."""
    kwargs.setdefault("canonical", True)
    chart_name = type(pv.point.chart).__name__
    rep_name = wl.pformat(pv.point.rep, **kwargs)
    comps_doc, unit_doc = vector_comps_unit_docs(pv.point)
    values_str = vector_values_str(pv.point, **kwargs)

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
    >>> import coordinax as cx
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
            except ValueError as exc:
                # `ValueError` only: `broadcast_shapes` also raises `TypeError`
                # for a malformed shape (a negative, non-integer or `None`
                # entry), which is a broken `.shape` somewhere upstream rather
                # than an incompatibility between these ones. Relabelling it
                # "not broadcastable" would name the wrong fault.
                msg = f"Coordinate: shapes {all_shapes} are not broadcastable: {exc}"
                raise ValueError(msg) from exc

        # Bypass equinox's immutable __setattr__
        self.__dict__["point"] = point
        self.__dict__["_data"] = field_vecs

    @classmethod
    def _create_unchecked(
        cls, point: Point, fields: dict[str, Tangent]
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
        >>> import coordinax as cx
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

        # Broadcast point + fields to the bundle shape (not each vector's
        # own) before indexing.
        shape = self.shape
        new_point = dataclassish.replace(
            self.point, data=broadcast_and_index_data(self.point.data, shape, key)
        )
        new_fields = {
            name: dataclassish.replace(
                vec, data=broadcast_and_index_data(vec.data, shape, key)
            )
            for name, vec in self._data.items()
        }
        return Coordinate._create_unchecked(new_point, new_fields)

    def keys(self) -> KeysView[str]:
        """Return field names (excluding base point).

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
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
        >>> import coordinax as cx
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

        if _cconvert_needs_joint_jet(self, to_chart, field_charts):
            return _cconvert_jointly(self, to_chart, field_charts, usys)

        # 1. Convert base point (pure point map — no Jacobian needed)
        new_point = cast("Point", cxr.cconvert(self.point, to_chart, usys=usys))

        # 2. Convert each field via tangent pushforward at self.point.
        # Express self.point in each fibre's current chart for the Jacobian;
        # this handles fibres that are in a different chart than self.point.
        # Fibres sharing a chart (e.g. velocity and acceleration both stored
        # in the same non-base chart) share the base-point conversion too,
        # rather than recomputing it once per fibre.
        at_in_chart: dict[cxc.AbstractChart, Point] = {}

        def point_in_chart(chart: cxc.AbstractChart, /) -> Point:
            if chart not in at_in_chart:
                at_in_chart[chart] = cast("Point", cxr.cconvert(self.point, chart))
            return at_in_chart[chart]

        new_fields: dict[str, Tangent] = {}
        for name, vec in self._data.items():
            target = field_charts.get(name, to_chart)
            at = (
                self.point
                if vec.chart == self.point.chart
                else point_in_chart(vec.chart)
            )
            new_fields[name] = cast(
                "Tangent", cxr.cconvert(vec, target, at=at, usys=usys)
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
        >>> import coordinax as cx
        >>> pv = cx.Coordinate(point=cx.Point.from_([1.0, 2.0, 3.0], "m"))
        >>> pv.shape
        ()

        """
        all_shapes = [self.point.shape, *(v.shape for v in self._data.values())]
        return jnp.broadcast_shapes(*all_shapes)

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
        >>> import coordinax as cx

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
        >>> import coordinax as cx

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

    >>> import coordinax as cx
    >>> pv = cx.Coordinate(point=cx.Point.from_([1.0, 2.0, 3.0], "m"))
    >>> cx.Coordinate.from_(pv) is pv
    True

    """
    return pv


@Coordinate.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Coordinate], p: Point, /) -> Coordinate:
    """Wrap a single ``Point`` as a point-only bundle (no field vectors).

    >>> import coordinax as cx
    >>> p = cx.Point.from_([1.0, 2.0, 3.0], "m")
    >>> pv = cx.Coordinate.from_(p)
    >>> pv.point is p
    True

    """
    return cls(point=p)


@Coordinate.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Coordinate], data: Mapping[str, Any], /, *, point: Point | None = None
) -> Coordinate:
    """Create a ``Coordinate`` from a mapping of named objects.

    The mapping may contain a ``"point"`` key for the base; the explicit
    ``point`` keyword argument takes precedence if both are supplied.

    >>> import unxt as u
    >>> import coordinax as cx
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


# ===================================================================
# gh#936: a chart change is second-order on an order >= 2 fibre


def _ladder_fibres(coord: "Coordinate", /) -> dict[str, int]:
    """Ladder fibre name -> curve-derivative order, for orders >= 1.

    Displacement fibres (order 0) are excluded: a displacement is a
    same-parameter point difference, not a curve derivative, so it is not a
    jet slot and the Jacobian pushforward is its whole transformation law.
    """
    out: dict[str, int] = {}
    for name, vec in coord._data.items():
        order = vec.rep.semantic_kind.order
        if order is not None and order >= 1:
            out[name] = order
    return out


#: Groups of chart types that are affine relabellings of one another -- same
#: parameterisation, coordinates differing by a permutation, a sign and a
#: shift, so the transition Jacobian is constant and $\partial^2\psi \equiv 0$.
#: `LonCosLatSpherical3D` is deliberately absent: its ``lon_coslat`` carries a
#: $\cos(\mathrm{lat})$ factor, which makes the Jacobian base-point dependent
#: like any other curvilinear map. Membership is pinned by a test that probes
#: `jac_pt_map` at two separated points and asserts it is constant.
_AFFINE_RELABELLINGS: tuple[frozenset[type], ...] = (
    frozenset({cxc.Spherical3D, cxc.MathSpherical3D, cxc.LonLatSpherical3D}),
)


def _chart_map_is_affine(
    from_chart: cxc.AbstractChart, to_chart: cxc.AbstractChart, /
) -> bool:
    r"""Whether the chart transition ``from_chart -> to_chart`` is affine.

    Exactly the condition under which $\partial^2\psi \equiv 0$, so the
    Jacobian pushforward is the complete law at every order and the cheap
    per-fibre path stays correct.

    Three ways to qualify: a chart with itself, two Cartesian-type charts
    (which differ by at most a linear relabelling of flat space), and two
    members of the same relabelling family -- `sph3d`, `math_sph3d` and
    `lonlat_sph3d` are the same parameterisation written three ways, with
    $\mathrm{lat} = \pi/2 - \theta$ and friends.

    Conservative where it is unsure: an unrecognised pair reports `False` and
    takes the joint-jet path, which is always correct and merely costlier.
    The price of a false negative is a needless prolongation -- and, for a
    bundle whose ladder has a hole, a needless refusal -- so the family list
    is worth keeping current.
    """
    if from_chart == to_chart:
        return True
    if cxfm_utils.is_flat_chart(from_chart) and cxfm_utils.is_flat_chart(to_chart):
        return True
    pair = {type(from_chart), type(to_chart)}
    return any(pair <= family for family in _AFFINE_RELABELLINGS)


def _cconvert_needs_joint_jet(
    coord: "Coordinate",
    to_chart: cxc.AbstractChart,
    field_charts: Mapping[str, cxc.AbstractChart],
    /,
) -> bool:
    r"""Whether this conversion must carry a jet rather than each fibre alone.

    Converting a fibre of order $m \geq 2$ between charts is not the Jacobian
    pushforward: the law is $a' = \partial\psi \cdot a + \partial^2\psi(v,
    v)$, and the second term is built from the *velocity* fibre. Walking the
    fibres one at a time never has it in hand, so it was dropped -- a
    Cartesian acceleration of exactly zero converted to exactly zero in
    spherical coordinates, where the true coordinate acceleration is not zero
    at all (gh#936).

    Asked per fibre, from **that fibre's own chart** to its own target. The
    point's chart is not the question: a bundle may keep its acceleration
    somewhere else entirely, and a point sitting in `cart3d` says nothing
    about an `sph3d` fibre being carried to `cart3d` beside it.

    Order <= 1 is unaffected -- there the Jacobian *is* the law -- as is any
    leg whose transition is affine, which keeps flat-to-flat bundles off the
    autodiff path.
    """
    return any(
        not _chart_map_is_affine(
            coord._data[name].chart, field_charts.get(name, to_chart)
        )
        for name, order in _ladder_fibres(coord).items()
        if order >= 2
    )


def _cconvert_jointly(
    coord: "Coordinate",
    to_chart: cxc.AbstractChart,
    field_charts: Mapping[str, cxc.AbstractChart],
    usys: OptUSys,
    /,
) -> "Coordinate":
    """Convert a bundle, carrying a jet for each fibre that needs one.

    Decided per fibre rather than for the bundle as a whole, because each
    fibre has its own source chart and may have its own target. A fibre whose
    leg is affine, or whose order is at most 1, keeps the cheap Jacobian
    path; only the rest are prolonged.
    """
    new_point = cast("Point", cxr.cconvert(coord.point, to_chart, usys=usys))
    ladder = _ladder_fibres(coord)

    new_fields: dict[str, Tangent] = {}
    for name, vec in coord._data.items():
        target = field_charts.get(name, to_chart)
        order = ladder.get(name, 0)
        if order < 2 or _chart_map_is_affine(vec.chart, target):
            at = _point_in(coord, vec.chart, usys)
            new_fields[name] = cast(
                "Tangent", cxr.cconvert(vec, target, at=at, usys=usys)
            )
        else:
            new_fields[name] = carry_fibre_across(
                coord, name, order, vec, target, usys, verb="cconvert"
            )

    return Coordinate._create_unchecked(new_point, new_fields)


def _point_in(coord: "Coordinate", chart: cxc.AbstractChart, usys: OptUSys, /) -> Point:
    """Return the base point expressed in ``chart`` (a no-op if it matches)."""
    if coord.point.chart == chart:
        return coord.point
    return cast("Point", cxr.cconvert(coord.point, chart, usys=usys))


def carry_fibre_across(
    coord: "Coordinate",
    name: str,
    order: int,
    fibre: Tangent,
    to_chart: cxc.AbstractChart,
    usys: OptUSys,
    /,
    *,
    verb: str,
) -> Tangent:
    r"""Carry an order >= 2 fibre to ``to_chart``, second-order exact.

    The fibre's jet is assembled in the fibre's **own** chart -- slot 0 is a
    point map and slot 1 is the Jacobian, which is the whole law at order 1,
    so the lower slots convert in exactly and the recursion bottoms out at
    once -- and that jet is prolonged across. Anything less drops
    $\partial^2\psi(v, v)$, and drops it *before* a joint prolongation could
    put it back.

    Shared by `Coordinate.cconvert` and by `act` on a `Coordinate`, which
    needs the same manoeuvre to gather a foreign fibre into the point's chart
    before building the bundle's jet. ``verb`` only names the caller in the
    error messages.
    """
    _require_coordinate_basis(name, order, fibre, verb)
    src = fibre.chart

    if order > 2:  # pragma: no cover - the named ladder stops at `acc`
        # Defensive, and reachable only through a custom semantic kind: the
        # library's own ladder is dpl/vel/acc, so no fibre of order 3 can be
        # built from the public API to exercise it.
        msg = (
            f"{verb} of a Coordinate cannot carry the order-{order} fibre "
            f"{name!r} from {src!r} to {to_chart!r}: assembling its jet there "
            f"would need the order-{order - 1} slot in that chart, which is "
            "the same conversion one level down. Put the fibre in the target "
            "chart first."
        )
        raise TypeError(msg)

    below = [n for n, o in _ladder_fibres(coord).items() if o == 1]
    if not below:
        msg = (
            f"{verb} of a Coordinate cannot carry the order-{order} fibre "
            f"{name!r} to {to_chart!r} without an order-1 fibre: the chart "
            "change contributes d2psi(v, v) at that order, which is built "
            "from it. An absent fibre means 'not tracked', not 'zero', so "
            "this refuses rather than returning the first-order answer. Add "
            "the velocity fibre, or convert between charts whose transition "
            "is affine."
        )
        raise TypeError(msg)
    if len(below) > 1:
        msg = (
            f"Coordinate has more than one order-1 fibre ({sorted(below)}); "
            f"which one anchors {name!r} is ambiguous."
        )
        raise ValueError(msg)

    vel = coord._data[below[0]]
    _require_coordinate_basis(below[0], 1, vel, verb)
    if vel.chart != src:
        # `at` anchors the Jacobian in the tangent's *source* chart.
        vel = cast(
            "Tangent",
            cxr.cconvert(vel, src, at=_point_in(coord, vel.chart, usys), usys=usys),
        )

    jet = {0: _point_in(coord, src, usys).data, 1: vel.data, order: fibre.data}
    out = prolong_point_map(_pt_map_to(src, to_chart, usys), jet)
    return cast("Tangent", dataclassish.replace(fibre, chart=to_chart, data=out[order]))


def _pt_map_to(
    from_chart: cxc.AbstractChart, to_chart: cxc.AbstractChart, usys: OptUSys, /
) -> "Callable[[Any], Any]":
    """Return the chart transition as a plain point map, for the jet engine."""

    def psi(data: Any, /) -> Any:
        return cxc.pt_map(data, from_chart, to_chart, usys=usys)

    return psi


def _require_coordinate_basis(
    name: str, order: int, vec: Tangent, verb: str, /
) -> None:
    """Refuse a fibre whose components are not the curve's coordinate derivatives.

    A physical (orthonormal) basis holds rescaled components, so they are not
    jet slots at all and prolonging them would carry the wrong numbers.
    """
    if vec.basis == cxr.coord_basis:
        return
    msg = (
        f"{verb} of a Coordinate cannot carry the order-{order} fibre "
        f"{name!r} in basis {vec.basis!r}: the jet law is written on the "
        "curve's coordinate derivatives, and a non-coordinate basis holds "
        "rescaled components. Convert it to the coordinate basis first with "
        "change_basis(..., at=point)."
    )
    raise TypeError(msg)
