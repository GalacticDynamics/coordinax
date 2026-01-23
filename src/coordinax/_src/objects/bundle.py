"""Anchored vector bundle: vector bundle over a base point.

A `PointedVector` stores a base position (role `Point`) and a collection
of additional vectors (typically tangent/physical roles such as `PhysDisp`,
`PhysVel`, `PhysAcc`) that are interpreted at the base point.
"""

__all__ = ("PointedVector",)

from collections.abc import ItemsView, KeysView, Mapping, ValuesView
from typing import Any, Generic, cast, final
from typing_extensions import TypeVar

import equinox as eqx
import jax
import plum

import quaxed.numpy as jnp
from xmmutablemap import ImmutableMap

import coordinax.charts as cxc
import coordinax.roles as cxr
from .base import AbstractVectorLike
from .utils import can_broadcast_shapes
from .vector import Vector

ChartT = TypeVar(
    "ChartT", bound=cxc.AbstractChart[Any, Any], default=cxc.AbstractChart[Any, Any]
)


# ==============================================================================
# Internal helpers


def _vconvert_field(
    v: Vector[cxc.AbstractChart[Any, Any], cxr.AbstractPhysicalRole, Any],
    *,
    to_chart: ChartT,
    base: Vector[cxc.AbstractChart[Any, Any], cxr.PhysDisp, Any],
) -> Vector[ChartT, cxr.AbstractPhysicalRole, Any]:
    """Convert a fibre vector to a new representation.

    This is an internal helper that handles the at= parameter computation
    for tangent vector transformations.

    Parameters
    ----------
    v : Vector
        Fibre vector to convert (must have tangent-like role)
    to_chart : AbstractChart
        Target representation
    base : Vector
        Base point (must have role Pos)

    Returns
    -------
    Vector
        Converted fibre vector in target chart

    """
    # Convert base to match the field's current rep (position conversion)
    at = base.vconvert(v.chart)
    # Convert the field vector using the matched base point
    out = v.vconvert(to_chart, at=at)
    return cast("Vector[ChartT, cxr.AbstractPhysicalRole, Any]", out)


# ==============================================================================
# Bundle class


@final
class PointedVector(
    AbstractVectorLike,
    ImmutableMap[str, Vector],  # type: ignore[misc]
    Generic[ChartT],
):
    r"""A vector bundle anchored at a base point.

    An ``PointedVector`` represents a mathematical vector bundle over a single base
    point, consisting of:

    - A base point $q \in M$ (a ``Vector`` with role ``PhysDisp``)
    - A collection of fibre vectors $\{v_i\}$ in the associated fibres at
      $q$, typically elements of $T_q M$ (tangent vectors such as
      displacement, velocity, or acceleration)

    This structure provides ergonomic handling of coordinate transformations for
    tangent vectors, automatically managing the base point dependency required
    for coordinate conversions in curvilinear systems.

    Mathematical Definition
    -----------------------
    For a $k$-dimensional manifold $M$, an anchored vector bundle at
    point $q \in M$ is the disjoint union:

    $$
        E_q = \{q\} \cup \bigsqcup_{i} F_i|_q
    $$
    where $F_i|_q$ are fibres (typically $T_q M$) at $q$.

    Coordinate Transformation Semantics
    ------------------------------------
    When converting the bundle to a new representation:

    1. **Base conversion** (position): $q' = f(q)$ where $f: M \to
       M$ is a coordinate chart map

    2. **Fibre conversion** (tangent vectors): $v' = \mathrm{d}f_q(v)$
       where $\mathrm{d}f_q$ is the pushforward at $q$

    The bundle automatically handles the base point dependency: when converting
    a fibre vector from representation $R$ to $S$, it computes ``at
    = base.vconvert(R)`` (position conversion) to provide the correct base point
    in the source representation before applying the tangent transformation.

    Parameters
    ----------
    base : Vector
        Base point of the bundle. Must have role ``PhysDisp``.
    **fields : Vector
        Named fibre vectors anchored at ``base``. Keys are user-defined names
        (e.g., ``"velocity"``, ``"acceleration"``). Values must be ``Vector``
        instances with tangent-like roles (``PhysDisp``, ``PhysVel``,
        ``PhysAcc``).
        All vectors must have compatible (broadcastable) shapes.

    Raises
    ------
    TypeError
        If ``base`` does not have role ``Point``, or if any field has role
        ``Point``.
    ValueError
        If vector shapes are not broadcastable.

    Notes
    -----
    - **Role separation**: Base must have role ``Point`` (affine location);
      fields must not have role {class}`coordinax.charts.Point` (must have tangent
      roles like PhysDisp, Vel, PhysAcc). This enforces the mathematical distinction
      between points and tangent vectors.

    - **Automatic ``at=`` handling**: The ``vconvert`` method automatically
      provides the ``at=`` parameter required for tangent vector
      transformations, eliminating manual bookkeeping.

    - **JAX compatibility**: The bundle is a PyTree (via Equinox) and works with
      ``jax.jit``, ``jax.vmap``, ``jax.grad``. For best performance, keep the
      bundle structure (field keys) static.

    - **Homogeneous components**: Tangent vectors store *physical* components in
      orthonormal frames with uniform dimensions (e.g., velocity has all
      components in m/s, not mixed [m/s, rad/s, m/s]). See
      ``coordinax.charts.tangent_transform`` for details.

    - **Frame compatibility**: This structure does not re-implement reference
      frames. Use ``coordinax.Coordinate`` to attach frames to vectors.

    See Also
    --------
    Vector : Individual coordinate vectors coordinax.charts.tangent_transform :
    Transformation for tangent vectors coordinax.Coordinate : Vectors with
    reference frames

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import coordinax.charts as cxc

    Create a bundle with position, velocity, and acceleration in Cartesian
    coords:

    >>> base = cx.Vector.from_([1, 2, 3], "km")
    >>> vel = cx.Vector.from_([10, 20, 30], "km/s")
    >>> acc = cx.Vector.from_([0.1, 0.2, 0.3], "km/s^2")
    >>> bundle = cx.PointedVector(base=base, velocity=vel, acceleration=acc)
    >>> print(bundle)
    PointedVector({
       base: <Vector: chart=Cart3D role=Pos (x, y, z) [km]
           [1 2 3]>,
       'velocity': <Vector: chart=Cart3D role=Vel (x, y, z) [km / s]
           [10 20 30]>,
       'acceleration': <Vector: chart=Cart3D role=PhysAcc (x, y, z) [km / s2]
           [0.1 0.2 0.3]>
    })

    Access individual vectors:

    >>> bundle.base
    Vector(...)
    >>> bundle["velocity"]
    Vector(...)

    Convert the entire bundle to spherical coordinates.  The base converts as a
    position, tangent vectors use the base as ``at=``:

    >>> sph_bundle = bundle.vconvert(cxc.sph3d)
    >>> sph_bundle.base.chart
    <Sph3D object at ...>
    >>> sph_bundle["velocity"].chart
    <Sph3D object at ...>

    This is equivalent to manually specifying:

    >>> base_sph = base.vconvert(cxc.sph3d)  # Position conversion
    >>> # Velocity requires base in the SAME rep as velocity (cart3d)
    >>> at_for_vel = base.vconvert(vel.chart)  # = base (already cart3d)
    >>> vel_sph = vel.vconvert(cxc.sph3d, at_for_vel)

    The bundle handles this automatically.

    Batch shapes work naturally (broadcasting):

    >>> batch_base = cx.Vector.from_([[1, 2, 3], [4, 5, 6]], "kpc")  # (2,)
    >>> batch_vel = cx.Vector.from_([10, 20, 30], "km/s")  # ()
    >>> batch_bundle = cx.PointedVector(base=batch_base, velocity=batch_vel)
    >>> batch_bundle.shape
    (2,)
    >>> batch_bundle[0].base.shape
    ()

    Index the bundle to get sub-bundles:

    >>> batch_bundle[0]
    PointedVector({base: ..., 'velocity': ...})

    Create from dictionaries:

    >>> data = {
    ...     "base": u.Q([1, 2, 3], "km"),
    ...     "velocity": u.Q([4, 5, 6], "km/s"),
    ... }
    >>> bundle2 = cx.PointedVector.from_(data)

    """

    base: Vector[ChartT, cxr.PhysDisp, Any]
    """Base point (position) of the bundle. Role must be ``PhysDisp``."""

    _data: dict[str, Vector[Any, Any, Any]] = eqx.field(repr=False)
    """Fibre vectors (fields) anchored at the base."""

    def __init__(
        self,
        /,
        *,
        base: Vector[ChartT, cxr.PhysDisp, Any],
        **fields: Any,
    ) -> None:
        """Initialize an anchored vector bundle.

        Parameters
        ----------
        base : Vector
            Base point. Must have role ``Point``.
        **fields : Any
            Named fibre vectors. Values are converted to ``Vector`` via
            ``Vector.from_`` if not already ``Vector`` instances.

        """
        # Validate base role
        base = eqx.error_if(
            base, not isinstance(base.role, cxr.Point), "base must have role Point"
        )

        # Convert fields to Vectors
        field_vectors = {
            k: Vector.from_(v) if not isinstance(v, Vector) else v
            for k, v in fields.items()
        }

        # Validate field roles: no additional Point vectors
        for name, vec in field_vectors.items():
            vec = eqx.error_if(  # noqa: PLW2901
                vec,
                isinstance(vec.role, cxr.Point),
                f"Field '{name}' has role Point. "
                "PointedVector stores fibre vectors anchored at base; "
                "store additional points elsewhere.",
            )

        # Check broadcastability
        all_vectors = [base, *field_vectors.values()]
        all_vectors = eqx.error_if(
            all_vectors,
            not can_broadcast_shapes([v.shape for v in all_vectors]),
            "vector shapes are not broadcastable.",
        )

        self.base = base
        self._data = field_vectors

    # ===============================================================
    # Mapping API

    @plum.dispatch
    def __getitem__(self, key: Any, /) -> "PointedVector":
        """Index the bundle's batch dimensions.

        Returns a new bundle with indexed base and fields.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> base = cx.Vector.from_(jnp.array([[1, 2, 3], [4, 5, 6]]), "m")
        >>> vel = cx.Vector.from_(jnp.array([[7, 8, 9], [10, 11, 12]]), "m/s")
        >>> bundle = cx.PointedVector(base=base, velocity=vel)
        >>> bundle[0]
        PointedVector(base=..., velocity=...)

        """
        # Use jax.tree.map to apply indexing only to array leaves
        indexed_base = jax.tree.map(
            lambda x: x[key] if eqx.is_array(x) else x, self.base, is_leaf=eqx.is_array
        )
        indexed_fields = {
            k: jax.tree.map(
                lambda x: x[key] if eqx.is_array(x) else x, v, is_leaf=eqx.is_array
            )
            for k, v in self._data.items()
        }

        return PointedVector(base=indexed_base, **indexed_fields)

    @plum.dispatch
    def __getitem__(self, key: str) -> Vector:
        """Get a field vector by name.

        Parameters
        ----------
        key : str
            Field name.

        Returns
        -------
        Vector
            The field vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> base = cx.Vector.from_([1, 2, 3], "m")
        >>> vel = cx.Vector.from_([4, 5, 6], "m/s")
        >>> bundle = cx.PointedVector(base=base, velocity=vel)
        >>> bundle["velocity"]
        Vector(...)

        """
        return ImmutableMap.__getitem__(self, key)

    def keys(self) -> KeysView[str]:
        """Return field names (excluding base)."""
        return self._data.keys()

    def values(self) -> ValuesView[Vector]:
        """Return field vectors (excluding base)."""
        return self._data.values()

    def items(self) -> ItemsView[str, Vector]:
        """Return (name, vector) pairs for fields (excluding base)."""
        return self._data.items()

    # ===============================================================
    # Convenience accessors

    @property
    def q(self) -> Vector[ChartT, cxr.PhysDisp, Any]:
        """Alias for base (position vector).

        Returns
        -------
        Vector
            The base point.

        """
        return self.base

    # ===============================================================
    # Vector API

    def vconvert(
        self,
        to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
        /,
        *,
        field_charts: Mapping[str, cxc.AbstractChart] | None = None,  # type: ignore[type-arg]
    ) -> "PointedVector":
        r"""Convert the bundle to a new representation.

        This method automatically handles the base point dependency required for
        tangent vector transformations. For each fibre vector, it:

        1. Converts the base to match the fibre's current representation
           (position conversion)
        2. Uses that converted base as the ``at=`` parameter for the fibre's
           tangent transformation

        Algorithm
        ---------
        1. Convert base: $q' = f(q)$ (position map, no ``at=`` needed)
        2. For each field $v$:

           a. Compute $\text{at} = q.\text{vconvert}(v.\text{rep})$
              (position conversion to fibre's rep)
           b. Convert fibre:
              $v' = v.\text{vconvert}(\text{to\_rep}, \text{at}=\text{at})$
              (tangent transformation at the base)

        3. Return new bundle with $q'$ and $\{v'_i\}$

        Parameters
        ----------
        to_chart : AbstractChart
            Target representation instance for the base and (by default) all fields.
        field_charts : Mapping[str, AbstractChart], optional
            Override target representation for specific fields. Keys are field
            names, values are target reps for those fields. If not provided,
            all fields use ``to_chart``.

        Returns
        -------
        PointedVector
            New bundle in the target representation(s).

        Notes
        -----
        - **Static structure for JIT**: For best performance with ``jax.jit``,
          keep the bundle's field structure (keys and number of fields) static.
          The representations can vary, but field names should not change
          between JIT compilations.

        - **Role-aware conversion**: The base converts as ``Point`` (position
          map), fields convert according to their roles (``PhysDisp``,
          ``PhysVel``, ``PhysAcc`` use tangent transformation).

        - **Representation matching**: The base is automatically converted to match
          each field's representation before using it as ``at=``, ensuring the
          tangent transformation evaluates at the correct point.

        Examples
        --------
        >>> import coordinax as cx
        >>> base = cx.Vector.from_([1, 1, 1], "m")
        >>> vel = cx.Vector.from_([10, 10, 10], "m/s")
        >>> bundle = cx.PointedVector(base=base, velocity=vel)

        Convert to spherical:

        >>> sph_bundle = bundle.vconvert(cxc.sph3d)
        >>> sph_bundle.base.chart
        <Sph3D object at ...>

        This is equivalent to:

        >>> base_sph = base.vconvert(cxc.sph3d)  # Pos conversion
        >>> at_for_vel = base.vconvert(vel.chart)  # cart3d (already matches)
        >>> vel_sph = vel.vconvert(
        ...     cxc.sph3d, at_for_vel
        ... )  # PhysVel tangent transform

        Specify different target reps for fields:

        >>> mixed_bundle = bundle.vconvert(
        ...     cxc.sph3d,
        ...     field_charts={"velocity": cxc.cyl3d}
        ... )
        >>> mixed_bundle.base.chart  # sph3d
        <Sph3D object at ...>
        >>> mixed_bundle["velocity"].chart  # cyl3d
        <Cyl3D object at ...>

        """
        # Convert base (position map, no 'at' needed)
        new_base = cast(
            "Vector[cxc.AbstractChart[Any, Any], cxr.PhysDisp, Any]",
            self.base.vconvert(to_chart),
        )

        # Prepare field target reps
        if field_charts is None:
            field_charts = {}

        # Convert each field using the internal helper Note: Using dict
        # comprehension here rather than jax.tree_map because the field
        # structure (keys) is static, and we need different target_chart per
        # field when field_charts is provided. The helper function
        # _vconvert_field contains the JIT-friendly conversion logic.
        new_fields = {
            name: _vconvert_field(
                vec,
                to_chart=field_charts.get(name, to_chart),
                base=self.base,  # type: ignore[arg-type]
            )
            for name, vec in self._data.items()
        }

        return PointedVector(base=new_base, **new_fields)

    # ===============================================================
    # Quax API

    def aval(self) -> jax.core.ShapedArray:
        """Return abstract array value for JAX tracing.

        The bundle is represented as a flat concatenation of all component
        arrays (base + fields).

        """
        all_vectors = [self.base, *self._data.values()]
        avals = [v.aval() for v in all_vectors]
        shapes = [a.shape for a in avals]
        shape = (
            *jnp.broadcast_shapes(*[s[:-1] for s in shapes]),
            sum(s[-1] for s in shapes),
        )
        dtype = jnp.result_type(*map(jnp.dtype, avals))
        return jax.core.ShapedArray(shape, dtype)

    # ===============================================================
    # Array API

    def __eq__(self: "PointedVector", other: object) -> Any:
        """Check equality of bundles."""
        if type(other) is not type(self):
            return NotImplemented

        other_bundle = cast("PointedVector", other)

        # Compare base
        base_eq = jnp.equal(self.base, other_bundle.base)

        # Compare fields
        if set(self.keys()) != set(other_bundle.keys()):
            return False

        field_eqs = [jnp.equal(self[k], other_bundle[k]) for k in self.keys()]

        # Combine all comparisons
        return jnp.logical_and(base_eq, jnp.stack(field_eqs).all())

    def __hash__(self) -> int:
        """Hash the bundle."""
        return hash((self.base, tuple(self.items())))

    @property
    def shape(self) -> tuple[int, ...]:
        """Broadcast shape of all vectors in the bundle."""
        all_vectors = [self.base, *self._data.values()]
        shapes = [v.shape for v in all_vectors]
        return jnp.broadcast_shapes(*shapes)


@AbstractVectorLike.from_.dispatch  # type: ignore[untyped-decorator]
def from_(
    cls: type[PointedVector], data: Mapping[str, Any], /, *, base: Vector | None = None
) -> PointedVector:
    """Create an PointedVector from a mapping.

    Parameters
    ----------
    cls
        The PointedVector class.
    data
        Dictionary of vectors. Must contain a ``"base"`` key unless
        ``base`` is provided separately. Values are normalized via
        ``Vector.from_``.
    base
        Explicit base point. If provided, overrides ``data["base"]``.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> data = {
    ...     "base": u.Q([1, 2, 3], "km"),
    ...     "velocity": u.Q([4, 5, 6], "km/s"),
    ... }
    >>> bundle = cx.PointedVector.from_(data)

    """
    data_dict = dict(data)

    # Get base: from parameter or from data_dict
    if base is None:
        base = data_dict.pop("base", None)
    else:
        data_dict.pop("base", None)  # Remove from data_dict to avoid conflict

    if base is None:
        msg = "base must be provided either as a parameter or in data['base']"
        raise ValueError(msg)

    # Normalize base to Vector
    base = Vector.from_(base)

    return cls(
        base=cast("Vector[cxc.AbstractChart[Any, Any], cxr.PhysDisp, Any]", base),
        **data_dict,
    )
