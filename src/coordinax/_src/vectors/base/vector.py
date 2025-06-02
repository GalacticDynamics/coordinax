"""Representation of coordinates in different systems."""

__all__ = ["AbstractVector"]

from abc import abstractmethod
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import jax
import numpy as np
import quax_blocks
import wadler_lindig as wl

import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items, field_values, fields, replace

from .attribute import VectorAttribute
from .base import AbstractVectorLike
from .flags import AttrFilter
from coordinax._src.custom_types import Unit
from coordinax._src.utils import classproperty
from coordinax._src.vectors.api import vector
from coordinax._src.vectors.mixins import (
    AstropyRepresentationAPIMixin,
    IPythonReprMixin,
)

if TYPE_CHECKING:
    from typing import Self

VT = TypeVar("VT", bound="AbstractVector")


class AbstractVector(
    IPythonReprMixin,
    AstropyRepresentationAPIMixin,
    quax_blocks.NumpyInvertMixin[Any],
    quax_blocks.LaxLenMixin,
    AbstractVectorLike,
):
    """Base class for all vector types.

    A vector is a collection of components that can be represented in different
    coordinate systems. This class provides a common interface for all vector
    types. All fields of the vector are expected to be components of the vector.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    Scalar vectors have length 0:

    >>> vec = cx.vecs.CartesianPos1D.from_([1], "m")
    >>> len(vec)
    0

    Vectors with certain lengths:

    >>> vec = cx.vecs.CartesianPos1D(u.Quantity([1], "m"))
    >>> len(vec)
    1

    >>> vec = cx.vecs.CartesianPos1D(u.Quantity([1, 2], "m"))
    >>> len(vec)
    2

    String representation of vectors:

    - a vector with only axis fields

    >>> vec1 = cx.CartesianPos3D.from_([1, 2, 3], "m")
    >>> print(str(vec1))
    <CartesianPos3D: (x, y, z) [m]
        [1 2 3]>

    - a vector with additional attributes

    >>> vec2 = vec1.vconvert(cx.vecs.ProlateSpheroidalPos, Delta=u.Quantity(1, "m"))
    >>> print(str(vec2))
    <ProlateSpheroidalPos: (mu[m2], nu[m2], phi[rad])
        Delta=Quantity(1, unit='m')
        [14.374  0.626  1.107]>

    """

    # TODO: 1) a better variable name. 2) make this public (and frozen)
    #       3) a params_fields property that's better than `components`
    _AUX_FIELDS: ClassVar[tuple[str, ...]]

    def __init_subclass__(cls) -> None:
        """Determine properties of the vector class."""
        # Note that this is called before `dataclass` has had a chance to
        # process the class, so we cannot just call `fields(cls)` to get the
        # fields. Instead, we parse `cls.__annotation__` directly.

        # To find the auxiliary fields, we look for fields that are instances of
        # `VectorAttribute`. This is done by looking at the `__dict__` rather
        # than doing `getattr` since `VectorAttribute` is a descriptor and will
        # raise an error when accessed before the dataclass construction is
        # complete.
        aux = [
            k
            for k in cls.__annotations__
            if isinstance(cls.__dict__.get(k), VectorAttribute)
        ]
        cls._AUX_FIELDS = tuple(aux)

    # ===============================================================
    # Vector API

    # TODO: make public
    @abstractmethod
    def _dimensionality(self) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.CartesianPos2D._dimensionality()
        2

        """
        raise NotImplementedError  # pragma: no cover

    # TODO: a better name
    @property
    def _auxiliary_data(self) -> dict[str, Any]:
        """Get the auxiliary data of the vector.

        This is a dictionary of auxiliary data that is not part of the vector
        itself. This is used for storing additional information about the
        vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> vec = cx.vecs.CartesianPos3D.from_([1, 2, 3], "m")
        >>> vec._auxiliary_data
        {}

        >>> vec = cx.vecs.ProlateSpheroidalPos(
        ...     mu=u.Quantity(3, "m2"), nu=u.Quantity(2, "m2"),
        ...     phi=u.Quantity(4, "rad"), Delta=u.Quantity(1.5, "m"))
        >>> vec._auxiliary_data
        {'Delta': Quantity(Array(1.5, dtype=float32, ...), unit='m')}

        """
        return {k: getattr(self, k) for k in self._AUX_FIELDS}

    # ===============================================================
    # Array API

    # ---------------------------------------------------------------
    # Attributes

    # `.dtype`, `.shape`, `.size` handled by Quax
    # TODO: .device

    @property
    def mT(self) -> "Self":  # noqa: N802
        """Transpose the vector.

        Examples
        --------
        We assume the following imports:

        >>> import unxt as u
        >>> import coordinax as cx

        We can transpose a vector:

        >>> vec = cx.CartesianPos3D(x=u.Quantity([[0, 1], [2, 3]], "m"),
        ...                         y=u.Quantity([[0, 1], [2, 3]], "m"),
        ...                         z=u.Quantity([[0, 1], [2, 3]], "m"))
        >>> vec.mT.x
        Quantity(Array([[0, 2],
                                  [1, 3]], dtype=int32), unit='m')

        """
        return replace(self, **{k: v.mT for k, v in field_items(AttrFilter, self)})

    @property
    def ndim(self) -> int:
        """Number of array dimensions (axes).

        Examples
        --------
        We assume the following imports:

        >>> import unxt as u
        >>> import coordinax as cx

        We can get the number of dimensions of a vector:

        >>> vec = cx.vecs.CartesianPos2D.from_([1, 2], "m")
        >>> vec.ndim
        0

        >>> vec = cx.vecs.CartesianPos2D.from_([[1, 2], [3, 4]], "m")
        >>> vec.ndim
        1

        ``ndim`` is calculated from the broadcasted shape. We can
        see this by creating a 2D vector in which the components have
        different shapes:

        >>> vec = cx.vecs.CartesianPos2D(x=u.Quantity([[1, 2], [3, 4]], "m"),
        ...                              y=u.Quantity(0, "m"))
        >>> vec.ndim
        2

        """
        return len(self.shape)

    @property
    def T(self) -> "Self":  # noqa: N802
        """Transpose the vector.

        Examples
        --------
        We assume the following imports:

        >>> import unxt as u
        >>> import coordinax as cx

        We can transpose a vector:

        >>> vec = cx.CartesianPos3D(x=u.Quantity([[0, 1], [2, 3]], "m"),
        ...                         y=u.Quantity([[0, 1], [2, 3]], "m"),
        ...                         z=u.Quantity([[0, 1], [2, 3]], "m"))
        >>> vec.T.x
        Quantity(Array([[0, 2],
                                  [1, 3]], dtype=int32), unit='m')

        """
        return replace(self, **{k: v.T for k, v in field_items(AttrFilter, self)})

    # ===============================================================
    # Convenience methods

    def asdict(
        self, *, dict_factory: Callable[[Any], Mapping[str, u.AbstractQuantity]] = dict
    ) -> Mapping[str, u.AbstractQuantity]:
        """Return the vector as a Mapping.

        Parameters
        ----------
        dict_factory
            The type of the mapping to return.

        Returns
        -------
        Mapping[str, Quantity]
            The vector as a mapping.

        See Also
        --------
        `dataclasses.asdict`
            This applies recursively to the components of the vector.

        Examples
        --------
        We assume the following imports:

        >>> import unxt as u
        >>> import coordinax as cx

        We can get the vector as a mapping:

        >>> vec = cx.vecs.CartesianPos2D(x=u.Quantity([[1, 2], [3, 4]], "m"),
        ...                              y=u.Quantity(0, "m"))
        >>> vec.asdict()
        {'x': Quantity(Array([[1, 2], [3, 4]], dtype=int32), unit='m'),
         'y': Quantity(Array(0, dtype=int32, ...), unit='m')}

        """
        return dict_factory(field_items(self))

    @classproperty
    @classmethod
    def components(cls) -> tuple[str, ...]:
        """Vector component names.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.CartesianPos2D.components
        ('x', 'y')
        >>> cx.SphericalPos.components
        ('r', 'theta', 'phi')
        >>> cx.vecs.RadialVel.components
        ('r',)

        """
        return tuple(f.name for f in fields(AttrFilter, cls))

    @property
    def units(self) -> MappingProxyType[str, Unit]:
        """Get the units of the vector's components.

        Examples
        --------
        >>> import coordinax as cx
        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")
        >>> vec.units
        mappingproxy({'x': Unit("km"), 'y': Unit("km"), 'z': Unit("km")})

        """
        return MappingProxyType(
            {k: u.unit_of(v) for k, v in field_items(AttrFilter, self)}
        )

    @classproperty
    @classmethod
    def dimensions(cls) -> dict[str, u.dims.AbstractDimension]:
        """Vector physical dimensions.

        Examples
        --------
        >>> import coordinax as cx

        >>> def pprint(d):
        ...     print({k: str(v).split("/")[0] for k, v in d.items()})

        >>> pprint(cx.vecs.CartesianPos1D.dimensions)
        {'x': 'length'}

        >>> pprint(cx.vecs.CartesianVel1D.dimensions)
        {'x': 'speed'}

        >>> pprint(cx.vecs.CartesianAcc1D.dimensions)
        {'x': 'acceleration'}

        >>> pprint(cx.vecs.RadialPos.dimensions)
        {'r': 'length'}

        >>> pprint(cx.vecs.RadialVel.dimensions)
        {'r': 'speed'}

        >>> pprint(cx.vecs.RadialAcc.dimensions)
        {'r': 'acceleration'}

        >>> pprint(cx.vecs.CartesianPos2D.dimensions)
        {'x': 'length', 'y': 'length'}

        >>> pprint(cx.vecs.CartesianVel2D.dimensions)
        {'x': 'speed', 'y': 'speed'}

        >>> pprint(cx.vecs.CartesianAcc2D.dimensions)
        {'x': 'acceleration', 'y': 'acceleration'}

        >>> pprint(cx.vecs.PolarPos.dimensions)
        {'r': 'length', 'phi': 'angle'}

        >>> pprint(cx.vecs.PolarVel.dimensions)
        {'r': 'speed', 'phi': 'angular frequency'}

        >>> pprint(cx.vecs.PolarAcc.dimensions)
        {'r': 'acceleration', 'phi': 'angular acceleration'}

        >>> pprint(cx.vecs.TwoSpherePos.dimensions)
        {'theta': 'angle', 'phi': 'angle'}

        >>> pprint(cx.vecs.TwoSphereVel.dimensions)
        {'theta': 'angular frequency', 'phi': 'angular frequency'}

        >>> pprint(cx.vecs.TwoSphereAcc.dimensions)
        {'theta': 'angular acceleration', 'phi': 'angular acceleration'}

        >>> pprint(cx.vecs.CartesianPos3D.dimensions)
        {'x': 'length', 'y': 'length', 'z': 'length'}

        >>> pprint(cx.vecs.CartesianVel3D.dimensions)
        {'x': 'speed', 'y': 'speed', 'z': 'speed'}

        >>> pprint(cx.vecs.CartesianAcc3D.dimensions)
        {'x': 'acceleration', 'y': 'acceleration', 'z': 'acceleration'}

        >>> pprint(cx.vecs.CylindricalPos.dimensions)
        {'rho': 'length', 'phi': 'angle', 'z': 'length'}

        >>> pprint(cx.vecs.CylindricalVel.dimensions)
        {'rho': 'speed', 'phi': 'angular frequency', 'z': 'speed'}

        >>> pprint(cx.vecs.CylindricalAcc.dimensions)
        {'rho': 'acceleration', 'phi': 'angular acceleration', 'z': 'acceleration'}

        >>> pprint(cx.vecs.SphericalPos.dimensions)
        {'r': 'length', 'theta': 'angle', 'phi': 'angle'}

        >>> pprint(cx.vecs.SphericalVel.dimensions)
        {'r': 'speed', 'theta': 'angular frequency', 'phi': 'angular frequency'}

        >>> pprint(cx.vecs.SphericalAcc.dimensions)
        {'r': 'acceleration', 'theta': 'angular acceleration', 'phi': 'angular acceleration'}

        >>> pprint(cx.vecs.LonLatSphericalPos.dimensions)
        {'lon': 'angle', 'lat': 'angle', 'distance': 'length'}

        >>> pprint(cx.vecs.LonLatSphericalVel.dimensions)
        {'lon': 'angular frequency', 'lat': 'angular frequency', 'distance': 'speed'}

        >>> pprint(cx.vecs.LonLatSphericalAcc.dimensions)
        {'lon': 'angular acceleration', 'lat': 'angular acceleration', 'distance': 'acceleration'}

        >>> pprint(cx.vecs.MathSphericalPos.dimensions)
        {'r': 'length', 'theta': 'angle', 'phi': 'angle'}

        >>> pprint(cx.vecs.MathSphericalVel.dimensions)
        {'r': 'speed', 'theta': 'angular frequency', 'phi': 'angular frequency'}

        >>> pprint(cx.vecs.MathSphericalAcc.dimensions)
        {'r': 'acceleration', 'theta': 'angular acceleration', 'phi': 'angular acceleration'}

        >>> pprint(cx.vecs.ProlateSpheroidalPos.dimensions)
        {'mu': 'area', 'nu': 'area', 'phi': 'angle'}

        >>> pprint(cx.vecs.ProlateSpheroidalVel.dimensions)
        {'mu': 'diffusivity', 'nu': 'diffusivity', 'phi': 'angular frequency'}

        >>> pprint(cx.vecs.ProlateSpheroidalAcc.dimensions)
        {'mu': 'dose of ionizing radiation', 'nu': 'dose of ionizing radiation', 'phi': 'angular acceleration'}

        >>> cx.vecs.Cartesian3D.dimensions
        <property object at ...>

        >>> cx.vecs.FourVector.dimensions
        <property object at ...>

        >>> pprint(cx.vecs.CartesianPosND.dimensions)
        {'q': 'length'}

        >>> pprint(cx.vecs.CartesianVelND.dimensions)
        {'q': 'speed'}

        >>> pprint(cx.vecs.CartesianAccND.dimensions)
        {'q': 'acceleration'}

        >>> pprint(cx.vecs.PoincarePolarVector.dimensions)
        {'rho': 'length', 'pp_phi': 'unknown', 'z': 'length',
         'dt_rho': 'speed', 'dt_pp_phi': 'unknown', 'dt_z': 'speed'}

        """  # noqa: E501
        return {f.name: u.dimension_of(f.type) for f in fields(AttrFilter, cls)}

    @property
    def dtypes(self) -> MappingProxyType[str, jnp.dtype[Any]]:
        """Get the dtypes of the vector's components.

        Examples
        --------
        >>> import coordinax as cx
        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")
        >>> vec.dtypes
        mappingproxy({'x': dtype('int32'), 'y': dtype('int32'),
                      'z': dtype('int32')})

        """
        return MappingProxyType({k: v.dtype for k, v in field_items(AttrFilter, self)})

    @property
    def devices(self) -> MappingProxyType[str, jax.Device]:
        """Get the devices of the vector's components.

        Examples
        --------
        >>> import coordinax as cx
        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")
        >>> vec.devices
        mappingproxy({'x': CpuDevice(id=0), 'y': CpuDevice(id=0),
                      'z': CpuDevice(id=0)})

        """
        return MappingProxyType({k: v.device for k, v in field_items(AttrFilter, self)})

    @property
    def shapes(self) -> MappingProxyType[str, tuple[int, ...]]:
        """Get the shapes of the vector's components.

        Examples
        --------
        >>> import coordinax as cx
        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")
        >>> vec.shapes
        mappingproxy({'x': (), 'y': (), 'z': ()})

        """
        return MappingProxyType({k: v.shape for k, v in field_items(AttrFilter, self)})

    @property
    def sizes(self) -> MappingProxyType[str, int]:
        """Get the sizes of the vector's components.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.CartesianPos2D.from_([1, 2], "m").sizes
        mappingproxy({'x': 1, 'y': 1})

        >>> cx.vecs.CartesianPos2D.from_([[1, 2], [1, 2]], "m").sizes
        mappingproxy({'x': 2, 'y': 2})

        """
        return MappingProxyType({k: v.size for k, v in field_items(AttrFilter, self)})

    # ===============================================================
    # Wadler-Lindig

    def _pdoc_comps(self) -> wl.AbstractDoc:
        # make the components string
        units_ = self.units
        if len(set(units_.values())) == 1:
            cdocs = [wl.TextDoc(f"{c}") for c in self.components]
            end = wl.TextDoc(f") [{units_[self.components[0]]}]")
        else:
            cdocs = [wl.TextDoc(f"{c}[{units_[c]}]") for c in self.components]
            end = wl.TextDoc(")")
        return wl.bracketed(
            begin=wl.TextDoc("("), docs=cdocs, sep=wl.comma, end=end, indent=4
        )

    def __pdoc__(self, *, vector_form: bool = False, **kwargs: Any) -> wl.AbstractDoc:
        """Return the Wadler-Lindig docstring for the vector.

        Parameters
        ----------
        vector_form
            If True, return the vector form of the docstring.
        short_arrays
            If True, use short arrays for the docstring.
        **kwargs
            Additional keyword arguments to pass to the Wadler-Lindig docstring
            formatter.

        """
        if not vector_form:
            # TODO: not use private API.
            return wl._definitions._pformat_dataclass(self, **kwargs)

        # -----------------------------

        cls_name = wl.TextDoc(self.__class__.__name__)

        # make the components string
        comps_doc = self._pdoc_comps()

        # make the aux fields string
        if not self._AUX_FIELDS:
            aux_doc = wl.TextDoc("")
        else:
            aux_doc = (
                wl.TextDoc("\n")  # force a line break
                + wl.bracketed(
                    begin=wl.TextDoc(" "),  # indent to opening "<"
                    docs=wl.named_objs(
                        [(k, getattr(self, k)) for k in self._AUX_FIELDS],
                        **kwargs | {"short_arrays": False, "compact_arrays": True},
                    ),
                    sep=wl.comma,
                    end=wl.TextDoc(""),  # no end bracket
                    indent=1,
                )
            )

        # make the values string
        # TODO: avoid casting to numpy arrays
        fvals = tuple(
            map(u.ustrip, jnp.broadcast_arrays(*field_values(AttrFilter, self)))
        )
        fvs = np.array2string(
            np.stack(fvals, axis=-1),
            precision=kwargs.pop("precision", 3),
            threshold=kwargs.pop("threshold", 1000),
            suffix=">",
        )
        vs_doc = wl.TextDoc(fvs).nest(4)

        # TODO: figure out how to avoid the hacky line breaks
        return (
            (  # header
                (wl.TextDoc("<") + cls_name + wl.TextDoc(":")).group()
                + wl.BreakDoc(" ")
                + comps_doc
            ).group()
            + aux_doc.group()  # aux fields
            + wl.TextDoc("\n")  # force a line break
            + (  # values (including end >)
                wl.TextDoc("    ") + vs_doc + wl.TextDoc(">")
            ).group()
        )


# ---------------------------------------------------------------
# Constructors


@AbstractVector.from_.dispatch  # type: ignore[misc]
def from_(cls: type[AbstractVector], *args: Any, **kwargs: Any) -> AbstractVector:
    """Create a vector from arguments.

    See `coordinax.vector` for more information.

    """
    return vector(cls, *args, **kwargs)
