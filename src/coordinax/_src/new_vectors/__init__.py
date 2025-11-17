"""Vector."""
# ruff:noqa: E501

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Generic, Literal, NoReturn, TypeVar

import equinox as eqx
import jax
import quax_blocks
from jaxtyping import Shaped
from quax import ArrayValue

import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_values
from xmmutablemap import ImmutableMap

from coordinax._src.vectors.base.flags import AttrFilter

Ks = TypeVar("Ks", bound=tuple[str, ...])
Ds = TypeVar("Ds", bound=tuple[str | None, ...])
V = TypeVar("V")


class AbstractRepresentation(Generic[Ks, Ds]):
    """Abstract base class for representations of vectors."""

    components: Ks
    """The names of the components."""

    dimensions: Ds
    """The dimensions of the components."""

    def dimensionality(self) -> int:
        return len(self.components)


# @dataclass(frozen=True, slots=True)
# class Representation(AbstractRepresentation[Ks]):
#     """Abstract base class for representations of vectors."""

#     components: tuple[Ks]
#     """The names of the components."""

#     dimensions: tuple[str | None, ...]
#     """The dimensions of the components."""


# fmt: off
# =========================================================
# 1D

# -----------------------------------------------
# Cartesian

Cart1DKeys = tuple[Literal["x"]]
CartPos1DDims = tuple[Literal["length"]]
CartVel1DDims = tuple[Literal["speed"]]
CartAcc1DDims = tuple[Literal["acceleration"]]

class CartPos1D(AbstractRepresentation[Cart1DKeys, CartPos1DDims]):
    components: Cart1DKeys = ("x",)
    dimensions: CartPos1DDims = ("length",)
class CartVel1D(AbstractRepresentation[Cart1DKeys, CartVel1DDims]):
    components: Cart1DKeys = ("x",)
    dimensions: CartVel1DDims = ("speed",)
class CartAcc1D(AbstractRepresentation[Cart1DKeys, CartAcc1DDims]):
    components: Cart1DKeys = ("x",)
    dimensions: CartAcc1DDims = ("acceleration",)


# -----------------------------------------------
# Radial

RadialKeys = tuple[Literal["r"]]
RadialPos1DDims = tuple[Literal["length"]]
RadialVel1DDims = tuple[Literal["speed"]]
RadialAcc1DDims = tuple[Literal["acceleration"]]


class RadialPos1D(AbstractRepresentation[RadialKeys, RadialPos1DDims]):
    components: RadialKeys = ("r",)
    dimensions: RadialPos1DDims = ("length",)
class RadialVel1D(AbstractRepresentation[RadialKeys, RadialVel1DDims]):
    components: RadialKeys = ("r",)
    dimensions: RadialVel1DDims = ("speed",)
class RadialAcc1D(AbstractRepresentation[RadialKeys, RadialAcc1DDims]):
    components: RadialKeys = ("r",)
    dimensions: RadialAcc1DDims = ("acceleration",)

# =========================================================
# 2D

# -----------------------------------------------
# Cartesian

Cart2DKeys = tuple[Literal["x"], Literal["y"]]
CartPos2DDims = tuple[Literal["length"], Literal["length"]]
CartVel2DDims = tuple[Literal["speed"], Literal["speed"]]
CartAcc2DDims = tuple[Literal["acceleration"], Literal["acceleration"]]

class CartPos2D(AbstractRepresentation[Cart2DKeys, CartPos2DDims]):
    components: Cart2DKeys = ("x", "y")
    dimensions: CartPos2DDims = ("length", "length")
class CartVel2D(AbstractRepresentation[Cart2DKeys, CartVel2DDims]):
    components: Cart2DKeys = ("x", "y")
    dimensions: CartVel2DDims = ("speed", "speed")
class CartAcc2D(AbstractRepresentation[Cart2DKeys, CartAcc2DDims]):
    components: Cart2DKeys = ("x", "y")
    dimensions: CartAcc2DDims = ("acceleration", "acceleration")

# -----------------------------------------------
# Polar

PolarKeys = tuple[Literal["r"], Literal["theta"]]
PolarPosDims = tuple[Literal["length"], Literal["angle"]]
PolarVelDims = tuple[Literal["speed"], Literal["angular speed"]]
PolarAccDims = tuple[Literal["acceleration"], Literal["angular acceleration"]]

class PolarPos(AbstractRepresentation[PolarKeys, PolarPosDims]):
    components: PolarKeys = ("r", "theta")
    dimensions: PolarPosDims = ("length", "angle")
class PolarVel(AbstractRepresentation[PolarKeys, PolarVelDims]):
    components: PolarKeys = ("r", "theta")
    dimensions: PolarVelDims = ("speed", "angular speed")
class PolarAcc(AbstractRepresentation[PolarKeys, PolarAccDims]):
    components: PolarKeys = ("r", "theta")
    dimensions: PolarAccDims = ("acceleration", "angular acceleration")

# -----------------------------------------------
# TwoSphere

TwoSphereKeys = tuple[Literal["theta"], Literal["phi"]]
TwoSpherePosDims = tuple[Literal["angle"], Literal["angle"]]
TwoSphereVelDims = tuple[Literal["angular speed"], Literal["angular speed"]]
TwoSphereAccDims = tuple[Literal["angular acceleration"], Literal["angular acceleration"]]

class TwoSpherePos(AbstractRepresentation[TwoSphereKeys, TwoSpherePosDims]):
    components: TwoSphereKeys = ("theta", "phi")
    dimensions: TwoSpherePosDims = ("angle", "angle")
class TwoSphereVel(AbstractRepresentation[TwoSphereKeys, TwoSphereVelDims]):
    components: TwoSphereKeys = ("theta", "phi")
    dimensions: TwoSphereVelDims = ("angular speed", "angular speed")
class TwoSphereAcc(AbstractRepresentation[TwoSphereKeys, TwoSphereAccDims]):
    components: TwoSphereKeys = ("theta", "phi")
    dimensions: TwoSphereAccDims = ("angular acceleration", "angular acceleration")

# =========================================================
# 3D

# -----------------------------------------------
# Cartesian

Cart3DKeys = tuple[Literal["x"], Literal["y"], Literal["z"]]
CartPos3DDims = tuple[Literal["length"], Literal["length"], Literal["length"]]
CartVel3DDims = tuple[Literal["speed"], Literal["speed"], Literal["speed"]]
CartAcc3DDims = tuple[Literal["acceleration"], Literal["acceleration"], Literal["acceleration"]]

class CartPos3D(AbstractRepresentation[Cart3DKeys, CartPos3DDims]):
    components: Cart3DKeys = ("x", "y", "z")
    dimensions: CartPos3DDims = ("length", "length", "length")
class CartVel3D(AbstractRepresentation[Cart3DKeys, CartVel3DDims]):
    components: Cart3DKeys = ("x", "y", "z")
    dimensions: CartVel3DDims = ("speed", "speed", "speed")
class CartAcc3D(AbstractRepresentation[Cart3DKeys, CartAcc3DDims]):
    components: Cart3DKeys = ("x", "y", "z")
    dimensions: CartAcc3DDims = ("acceleration", "acceleration", "acceleration")

# -----------------------------------------------
# Spherical

SphericalKeys = tuple[Literal["r"], Literal["theta"], Literal["phi"]]
SphericalPosDims = tuple[Literal["length"], Literal["angle"], Literal["angle"]]
SphericalVelDims = tuple[Literal["speed"], Literal["angular speed"], Literal["angular speed"]]
SphericalAccDims = tuple[Literal["acceleration"], Literal["angular acceleration"], Literal["angular acceleration"]]

class SphericalPos(AbstractRepresentation[SphericalKeys, SphericalPosDims]):
    components: SphericalKeys = ("r", "theta", "phi")
    dimensions: SphericalPosDims = ("length", "angle", "angle")
class SphericalVel(AbstractRepresentation[SphericalKeys, SphericalVelDims]):
    components: SphericalKeys = ("r", "theta", "phi")
    dimensions: SphericalVelDims = ("speed", "angular speed", "angular speed")
class SphericalAcc(AbstractRepresentation[SphericalKeys, SphericalAccDims]):
    components: SphericalKeys = ("r", "theta", "phi")
    dimensions: SphericalAccDims = ("acceleration", "angular acceleration", "angular acceleration")


# -----------------------------------------------
# Cylindrical

CylindricalKeys = tuple[Literal["rho"], Literal["phi"], Literal["z"]]
CylindricalPosDims = tuple[Literal["length"], Literal["angle"], Literal["length"]]
CylindricalVelDims = tuple[Literal["speed"], Literal["angular speed"], Literal["speed"]]
CylindricalAccDims = tuple[Literal["acceleration"], Literal["angular acceleration"], Literal["acceleration"]]

class CylindricalPos(AbstractRepresentation[CylindricalKeys, CylindricalPosDims]):
    components: CylindricalKeys = ("rho", "phi", "z")
    dimensions: CylindricalPosDims = ("length", "angle", "length")
class CylindricalVel(AbstractRepresentation[CylindricalKeys, CylindricalVelDims]):
    components: CylindricalKeys = ("rho", "phi", "z")
    dimensions: CylindricalVelDims = ("speed", "angular speed", "speed")
class CylindricalAcc(AbstractRepresentation[CylindricalKeys, CylindricalAccDims]):
    components: CylindricalKeys = ("rho", "phi", "z")
    dimensions: CylindricalAccDims = ("acceleration", "angular acceleration", "acceleration")

# =========================================================
# 4D

SpaceTimeKeys = tuple[Literal["t"], Literal["x"], Literal["y"], Literal["z"]]
SpaceTimeDims = tuple[Literal["time"], Literal["length"], Literal["length"], Literal["length"]]

@dataclass(frozen=True, slots=True)
class SpaceTimeRepresentation(AbstractRepresentation[SpaceTimeKeys, SpaceTimeDims]):
    """Representation for four-vectors."""

    components: SpaceTimeKeys = ("t", "x", "y", "z")
    """The names of the components."""

    dimensions: SpaceTimeDims = ("time", "length", "length", "length")
    """The dimensions of the components."""

    _: KW_ONLY
    c: Shaped[u.Quantity["speed"], ""] = field(default=u.Quantity(299_792.458, "km/s"))
    """Speed of light, by default ``Quantity(299_792.458, "km/s")``."""


# =========================================================
# Poincare

PoincarePolarKeys = tuple[
    Literal["rho"],
    Literal["pp_phi"],
    Literal["z"],
    Literal["dt_rho"],
    Literal["dt_pp_phi"],
    Literal["dt_z"],
]

PoincarePolarDims = tuple[
    Literal["length"],
    Literal["length / time**0.5"],
    Literal["length"],
    Literal["speed"],
    Literal["length / time**1.5"],
    Literal["speed"],
]

class PoincarePolarRep(AbstractRepresentation[PoincarePolarKeys, PoincarePolarDims]):
    components: PoincarePolarKeys = (
        "rho",
        "pp_phi",
        "z",
        "dt_rho",
        "dt_pp_phi",
        "dt_z",
    )
    dimensions: PoincarePolarDims = (
        "length",
        "length / time**0.5",
        "length",
        "speed",
        "length / time**1.5",
        "speed",
    )


# =========================================================
# N-Dimensional

# -----------------------------------------------
# Cartesian

CartesianNDKeys = tuple[Literal["q"]]
CartesianPosNDDims = tuple[Literal["length"]]
CartesianVelNDDims = tuple[Literal["speed"]]
CartesianAccNDDims = tuple[Literal["acceleration"]]

class CartesianNDRep(AbstractRepresentation[CartesianNDKeys, CartesianPosNDDims]):
    components: CartesianNDKeys = ("q",)
    dimensions: CartesianPosNDDims = ("length",)
class CartesianNDVelRep(AbstractRepresentation[CartesianNDKeys, CartesianVelNDDims]):
    components: CartesianNDKeys = ("q",)
    dimensions: CartesianVelNDDims = ("speed",)
class CartesianNDAccRep(AbstractRepresentation[CartesianNDKeys, CartesianAccNDDims]):
    components: CartesianNDKeys = ("q",)
    dimensions: CartesianAccNDDims = ("acceleration",)

# fmt: on
#####################################################################


class AbstractVectorLike(
    ArrayValue,
    quax_blocks.LaxBinaryOpsMixin[Any, Any],  # TODO: type annotation
    quax_blocks.LaxRoundMixin["AbstractVectorLike"],
    quax_blocks.LaxUnaryMixin[Any],
):
    pass


# =========================================================


class Vector(AbstractVectorLike, Generic[Ks, Ds, V]):
    """A vector."""

    data: Mapping[str, V]
    """The data for each """

    kind: AbstractRepresentation[Ks, Ds]

    def _check_init(self) -> None:
        # Check that the keys of data match kind.components
        if set(self.data.keys()) != set(self.kind.components):
            msg = "Data keys do not match kind components"
            raise ValueError(msg)

        # Check that the dimensions match kind.dimensions
        for v, d in zip(self.data.values(), self.kind.dimensions, strict=True):
            if d is not None and u.dimension_of(v) != d:
                raise ValueError

    def __getitem__(self, key: str) -> V:
        return self.data[key]

    # TODO: generalize to work with FourVector, and Space
    def aval(self) -> jax.core.ShapedArray:
        """Return the vector as a JAX array."""
        fvs = field_values(AttrFilter, self)
        shape = (*jnp.broadcast_shapes(*map(jnp.shape, fvs)), len(fvs))
        dtype = jnp.result_type(*map(jnp.dtype, fvs))
        return jax.core.ShapedArray(shape, dtype)

    def materialise(self) -> NoReturn:
        """Materialise the vector for `quax`.

        Examples
        --------
        >>> import coordinax as cx
        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")

        >>> try: vec.materialise()
        ... except RuntimeError as e: print(e)
        Refusing to materialise `CartesianPos3D`.

        """
        msg = f"Refusing to materialise `{type(self).__name__}`."
        raise RuntimeError(msg)


# CartesianPos3D = Vector[Cart3DKeys, CartPos3DDims, u.AbstractQuantity]


# =========================================================


class KinematicSpace(
    AbstractVectorLike,
    ImmutableMap[str, Vector],  # type: ignore[misc]
):
    """A collection of vectors that acts like the primary vector."""

    # TODO: https://peps.python.org/pep-0728/#the-extra-items-class-parameter
    _data: dict[str, Vector[Any, Any, Any]] = eqx.field(repr=False)

    def __eq__(self: "AbstractVectorLike", other: object) -> Any:
        """Check if the vector is equal to another object."""
        if type(other) is not type(self):
            return NotImplemented

        return jnp.equal(self, other)  # type: ignore[arg-type]
