"""Vector."""
# ruff:noqa: E501, E701

__all__ = (
    "AbstractRepresentation",
    "AbstractPos",
    "AbstractVel",
    "AbstractAcc",
    # = 1D ======================================
    # - Cartesian ---------------------
    "CartPos1D",
    "cartpos1d",
    "CartVel1D",
    "cartvel1d",
    "CartAcc1D",
    "cartacc1d",
    # - Radial ---------------------
    "RadialPos",
    "radialpos",
    "RadialVel",
    "radialvel",
    "RadialAcc",
    "radialacc",
    # = 2D ======================================
    # - Cartesian ---------------------
    "CartPos2D",
    "cartpos2d",
    "CartVel2D",
    "cartvel2d",
    "CartAcc2D",
    "cartacc2d",
    # - Polar ---------------------
    "PolarPos",
    "polarpos",
    "PolarVel",
    "polarvel",
    "PolarAcc",
    "polaracc",
    # - TwoSphere ---------------------
    "TwoSpherePos",
    "twospherepos",
    "TwoSphereVel",
    "twospherevel",
    "TwoSphereAcc",
    "twosphereacc",
    # = 3D ======================================
    # - Cartesian ---------------------
    "CartPos3D",
    "cartpos3d",
    "CartVel3D",
    "cartvel3d",
    "CartAcc3D",
    "cartacc3d",
    # - Cylindrical ---------------------
    "CylindricalPos",
    "cylindricalpos",
    "CylindricalVel",
    "cylindricalvel",
    "CylindricalAcc",
    "cylindricalacc",
    # - Spherical ---------------------
    "AbstractSphericalPos",
    "AbstractSphericalVel",
    "AbstractSphericalAcc",
    "SphericalPos",
    "sphericalpos",
    "SphericalVel",
    "sphericalvel",
    "SphericalAcc",
    "sphericalacc",
    # - LonLatSpherical ---------------
    "LonLatSphericalPos",
    "lonlatsphericalpos",
    "LonLatSphericalVel",
    "lonlatsphericalvel",
    "LonLatSphericalAcc",
    "lonlatsphericalacc",
    # - LonCosLatSpherical ---------------
    "LonCosLatSphericalPos",
    "loncoslatsphericalpos",
    "LonCosLatSphericalVel",
    "loncoslatsphericalvel",
    "LonCosLatSphericalAcc",
    "loncoslatsphericalacc",
    # - MathSpherical -----------------
    "MathSphericalPos",
    "mathsphericalpos",
    "MathSphericalVel",
    "mathsphericalvel",
    "MathSphericalAcc",
    "mathsphericalacc",
    # - ProlateSpheroidal -------------
    "ProlateSpheroidalPos",
    "prolatespheroidalpos",
    "ProlateSpheroidalVel",
    "prolatespheroidalvel",
    "ProlateSpheroidalAcc",
    "prolatespheroidalacc",
    # = 4D ======================================
    # - SpaceTime ---------------------
    "CartSpaceTime",
    "cartspacetime",
    "SpaceTime",
    # There's no instance of SpaceTime to export
    # = 6D ======================================
    # - PoincarePolar -----------------
    "PoincarePolarRep",
    "poincarepolar",
    # = N-D =====================================
    # - CartesianND -------------------
    "CartPosND",
    "cartndpos",
    "CartVelND",
    "cartndvel",
    "CartAccND",
    "cartndacc",
)

from abc import ABCMeta, abstractmethod
from dataclasses import KW_ONLY, dataclass, field

from collections.abc import Mapping
from jaxtyping import Shaped
from typing import (
    Any,
    Final,
    Generic,
    Literal as L,  # noqa: N817
    TypeAlias,
    TypeVar,
    final,
    get_args,
    no_type_check,
)

import plum
import wadler_lindig as wl

import quaxed.numpy as jnp
import unxt as u

from . import api, checks

GAT = TypeVar("GAT", bound=type(L[" ", "  "]))  # type: ignore[misc]
Ks = TypeVar("Ks", bound=tuple[str, ...])
Ds = TypeVar("Ds", bound=tuple[str | None, ...])
V = TypeVar("V")


Len: TypeAlias = L["length"]
Spd: TypeAlias = L["speed"]
Acc: TypeAlias = L["acceleration"]
Ang: TypeAlias = L["angle"]
AngSpd: TypeAlias = L["angular speed"]
AngAcc: TypeAlias = L["angular acceleration"]


class AbstractRepresentation(Generic[Ks, Ds], metaclass=ABCMeta):
    """Abstract base class for representations of vectors."""

    def __init_subclass__(cls) -> None:
        # This allows multiple inheritance with other ABCs that might or might
        # not define an `__init_subclass__`
        if hasattr(cls, "__init_subclass__"):
            super().__init_subclass__()

    # ===============================================================
    # Vector API

    @property
    @abstractmethod
    def components(self) -> Ks:
        """The names of the components."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> Ds:
        """The dimensions of the components."""
        ...

    @property
    def dimensionality(self) -> int:
        return len(self.components)

    @property
    def cartesian(self) -> "AbstractRepresentation[Ks, Ds]":
        """Return the corresponding Cartesian vector class."""
        return api.cartesian_rep(self)

    @property
    def time_derivative(self) -> "AbstractRepresentation[Ks, Ds]":
        """Return the corresponding time derivative class."""
        return api.time_derivative_rep(self)

    @property
    def time_antiderivative(self) -> "AbstractRepresentation[Ks, Ds]":
        """Return the corresponding time antiderivative class."""
        return api.time_antiderivative_rep(self)

    def time_nth_derivative(self, n: int) -> "AbstractRepresentation[Ks, Ds]":
        """Return the corresponding time nth derivative class."""
        out = self
        if n == 0:
            pass
        elif n < 0:
            for _ in range(-n):
                out = out.time_antiderivative
        else:
            for _ in range(n):
                out = out.time_derivative

        return out

    def check_data(self, data: Mapping[str, Any], /) -> None:
        # Check that the keys of data match kind.components
        if set(data.keys()) != set(self.components):
            msg = (
                "Data keys do not match kind components: "
                f"{set(data.keys())} != {set(self.components)}"
            )
            raise ValueError(msg)

        # Check that the dimensions match kind.dimensions
        for v, d in zip(data.values(), self.dimensions, strict=True):
            if d is not None and u.dimension_of(v) != d:
                msg = "Data dimensions do not match"
                raise ValueError(msg)

    # ===============================================================
    # Wadler-Lindig API

    def __pdoc__(self, **kw: object) -> wl.AbstractDoc:
        # TODO: vectorform
        return wl.TextDoc(
            f"{self.__class__.__name__}({self.components}, {self.dimensions})"
        )


@no_type_check
def get_tuple(tp: GAT, /) -> GAT:
    return tuple(arg.__args__[0] for arg in get_args(tp))


class AbstractFixedRepresentation(AbstractRepresentation[Ks, Ds]):
    def __init_subclass__(cls) -> None:
        # Extract Ks and Ds from AbstractFixedRepresentation in the inheritance
        for base in getattr(cls, "__orig_bases__", ()):
            origin = getattr(base, "__origin__", None)
            if origin is AbstractFixedRepresentation:
                args = get_args(base)
                if len(args) >= 2:
                    cls._components = get_tuple(args[0])  # type: ignore[attr-defined]
                    cls._dimensions = get_tuple(args[1])  # type: ignore[attr-defined]
                    break

        super().__init_subclass__()  # AbstractRepresentation has.

    @property
    def components(self) -> Ks:
        return self._components  # type: ignore[attr-defined]

    @property
    def dimensions(self) -> Ds:
        return self._dimensions  # type: ignore[attr-defined]


class AbstractPos:
    """ABC Flag for position representations."""


class AbstractVel:
    """ABC Flag for velocity representations."""


class AbstractAcc:
    """ABC Flag for acceleration representations."""


# fmt: off
# =========================================================
# 1D

class Abstract1D:
    """Abstract 1D vector representation."""

    # TODO add a check it's 1D

    def __init_subclass__(cls) -> None:
        # Enforce that this is a subclass of AbstractRepresentation
        if not cls.__name__.startswith("Abstract") and not issubclass(cls, AbstractRepresentation):
            msg = f"{cls.__name__} must be a subclass of AbstractRepresentation"
            raise TypeError(msg)


# -----------------------------------------------
# Cartesian

Cart1DKeys = tuple[L["x"]]
CartPos1DDims = tuple[Len]
CartVel1DDims = tuple[Spd]
CartAcc1DDims = tuple[Acc]

@final
class CartPos1D(AbstractFixedRepresentation[Cart1DKeys, CartPos1DDims],AbstractPos, Abstract1D):
    pass
@final
class CartVel1D(AbstractFixedRepresentation[Cart1DKeys, CartVel1DDims], AbstractVel, Abstract1D):
    pass
@final
class CartAcc1D(AbstractFixedRepresentation[Cart1DKeys, CartAcc1DDims], AbstractAcc, Abstract1D):
    pass
cartpos1d: Final = CartPos1D()
cartvel1d: Final = CartVel1D()
cartacc1d: Final = CartAcc1D()

@plum.dispatch
def time_derivative_rep(obj: CartPos1D, /) -> CartVel1D: return cartvel1d
@plum.dispatch
def time_antiderivative_rep(obj: CartVel1D, /) -> CartPos1D: return cartpos1d
@plum.dispatch
def time_derivative_rep(obj: CartVel1D, /) -> CartAcc1D: return cartacc1d
@plum.dispatch
def time_antiderivative_rep(obj: CartAcc1D, /) -> CartVel1D: return cartvel1d

# -----------------------------------------------
# Radial

RadialKeys = tuple[L["r"]]
RadialPosDims = tuple[Len]
RadialVelDims = tuple[Spd]
RadialAccDims = tuple[Acc]


@final
class RadialPos(AbstractFixedRepresentation[RadialKeys, RadialPosDims], AbstractPos, Abstract1D):
    pass
@final
class RadialVel(AbstractFixedRepresentation[RadialKeys, RadialVelDims], AbstractVel, Abstract1D):
    pass
@final
class RadialAcc(AbstractFixedRepresentation[RadialKeys, RadialAccDims], AbstractAcc, Abstract1D):
    pass

radialpos: Final = RadialPos()
radialvel: Final = RadialVel()
radialacc: Final = RadialAcc()

@plum.dispatch
def time_derivative_rep(obj: RadialPos, /) -> RadialVel: return radialvel
@plum.dispatch
def time_antiderivative_rep(obj: RadialVel, /) -> RadialPos: return radialpos
@plum.dispatch
def time_derivative_rep(obj: RadialVel, /) -> RadialAcc: return radialacc
@plum.dispatch
def time_antiderivative_rep(obj: RadialAcc, /) -> RadialVel: return radialvel

# -----------------------------------------------

class AbstractPos1D(AbstractPos, Abstract1D): pass
class AbstractVel1D(AbstractVel, Abstract1D): pass
class AbstractAcc1D(AbstractAcc, Abstract1D): pass

@plum.dispatch
def cartesian_rep(obj: AbstractPos1D, /) -> CartPos1D: return cartpos1d
@plum.dispatch
def cartesian_rep(obj: AbstractVel1D, /) -> CartVel1D: return cartvel1d
@plum.dispatch
def cartesian_rep(obj: AbstractAcc1D, /) -> CartAcc1D: return cartacc1d

# =========================================================
# 2D

class Abstract2D:
    """Abstract 2D vector representation."""

    # TODO add a check it's 2D

    def __init_subclass__(cls) -> None:
        # Enforce that this is a subclass of AbstractRepresentation
        if not cls.__name__.startswith("Abstract") and not issubclass(cls, AbstractRepresentation):
            msg = f"{cls.__name__} must be a subclass of AbstractRepresentation"
            raise TypeError(msg)


# -----------------------------------------------

class AbstractPos2D(AbstractPos, Abstract2D): pass
class AbstractVel2D(AbstractVel, Abstract2D): pass
class AbstractAcc2D(AbstractAcc, Abstract2D): pass

@plum.dispatch
def cartesian_rep(obj: AbstractPos2D, /) -> "CartPos2D": return cartpos2d
@plum.dispatch
def cartesian_rep(obj: AbstractVel2D, /) -> "CartVel2D": return cartvel2d
@plum.dispatch
def cartesian_rep(obj: AbstractAcc2D, /) -> "CartAcc2D": return cartacc2d

# -----------------------------------------------
# Cartesian

Cart2DKeys = tuple[L["x"], L["y"]]
CartPos2DDims = tuple[Len, Len]
CartVel2DDims = tuple[Spd, Spd]
CartAcc2DDims = tuple[Acc, Acc]

@final
class CartPos2D(AbstractFixedRepresentation[Cart2DKeys, CartPos2DDims], AbstractPos2D):
    pass
@final
class CartVel2D(AbstractFixedRepresentation[Cart2DKeys, CartVel2DDims], AbstractVel2D):
    pass
@final
class CartAcc2D(AbstractFixedRepresentation[Cart2DKeys, CartAcc2DDims], AbstractAcc2D):
    pass

cartpos2d: Final = CartPos2D()
cartvel2d: Final = CartVel2D()
cartacc2d: Final = CartAcc2D()

@plum.dispatch
def time_derivative_rep(obj: CartPos2D, /) -> CartVel2D: return cartvel2d
@plum.dispatch
def time_antiderivative_rep(obj: CartVel2D, /) -> CartPos2D: return cartpos2d
@plum.dispatch
def time_derivative_rep(obj: CartVel2D, /) -> CartAcc2D: return cartacc2d
@plum.dispatch
def time_antiderivative_rep(obj: CartAcc2D, /) -> CartVel2D: return cartvel2d

# -----------------------------------------------
# Polar

PolarKeys = tuple[L["r"], L["theta"]]
PolarPosDims = tuple[Len, Ang]
PolarVelDims = tuple[Spd, AngSpd]
PolarAccDims = tuple[Acc, AngAcc]

@final
class PolarPos(AbstractFixedRepresentation[PolarKeys, PolarPosDims], AbstractPos2D):
    pass
@final
class PolarVel(AbstractFixedRepresentation[PolarKeys, PolarVelDims], AbstractVel2D):
    pass
@final
class PolarAcc(AbstractFixedRepresentation[PolarKeys, PolarAccDims], AbstractAcc2D):
    pass

polarpos: Final = PolarPos()
polarvel: Final = PolarVel()
polaracc: Final = PolarAcc()

@plum.dispatch
def time_derivative_rep(obj: PolarPos, /) -> PolarVel: return polarvel
@plum.dispatch
def time_derivative_rep(obj: PolarVel, /) -> PolarAcc: return polaracc
@plum.dispatch
def time_antiderivative_rep(obj: PolarVel, /) -> PolarPos: return polarpos
@plum.dispatch
def time_antiderivative_rep(obj: PolarAcc, /) -> PolarVel: return polarvel

# -----------------------------------------------
# TwoSphere

TwoSphereKeys = tuple[L["theta"], L["phi"]]
TwoSpherePosDims = tuple[Ang, Ang]
TwoSphereVelDims = tuple[AngSpd, AngSpd]
TwoSphereAccDims = tuple[AngAcc, AngAcc]

@final
class TwoSpherePos(AbstractFixedRepresentation[TwoSphereKeys, TwoSpherePosDims], AbstractPos2D):
    def check_data(self, data: Mapping[str, Any], /) -> None:
        super().check_data(data)  # call base check
        checks.polar_range(data["theta"])
@final
class TwoSphereVel(AbstractFixedRepresentation[TwoSphereKeys, TwoSphereVelDims], AbstractVel2D):
    pass
@final
class TwoSphereAcc(AbstractFixedRepresentation[TwoSphereKeys, TwoSphereAccDims], AbstractAcc2D):
    pass

twospherepos: Final = TwoSpherePos()
twospherevel: Final = TwoSphereVel()
twosphereacc: Final = TwoSphereAcc()

@plum.dispatch
def time_derivative_rep(obj: TwoSpherePos, /) -> TwoSphereVel: return twospherevel
@plum.dispatch
def time_antiderivative_rep(obj: TwoSphereVel, /) -> TwoSpherePos: return twospherepos
@plum.dispatch
def time_derivative_rep(obj: TwoSphereVel, /) -> TwoSphereAcc: return twosphereacc
@plum.dispatch
def time_antiderivative_rep(obj: TwoSphereAcc, /) -> TwoSphereVel: return twospherevel


# =========================================================
# 3D

class Abstract3D(AbstractRepresentation):
    """ABC flag for 3D vector representation."""

    # TODO add a check it's 3D

    def __init_subclass__(cls) -> None:
        # Enforce that this is a subclass of AbstractRepresentation
        if not cls.__name__.startswith("Abstract") and not issubclass(cls, AbstractRepresentation):
            msg = f"{cls.__name__} must be a subclass of AbstractRepresentation"
            raise TypeError(msg)


# -----------------------------------------------

class AbstractPos3D(AbstractPos, Abstract3D): pass
class AbstractVel3D(AbstractVel, Abstract3D): pass
class AbstractAcc3D(AbstractAcc, Abstract3D): pass

@plum.dispatch
def cartesian_rep(obj: AbstractPos3D, /) -> "CartPos3D":
    return cartpos3d
@plum.dispatch
def cartesian_rep(obj: AbstractVel3D, /) -> "CartVel3D":
    return cartvel3d
@plum.dispatch
def cartesian_rep(obj: AbstractAcc3D, /) -> "CartAcc3D":
    return cartacc3d

# -----------------------------------------------
# Cartesian

Cart3DKeys = tuple[L["x"], L["y"], L["z"]]
CartPos3DDims = tuple[Len, Len, Len]
CartVel3DDims = tuple[Spd, Spd, Spd]
CartAcc3DDims = tuple[Acc, Acc, Acc]

@final
class CartPos3D(AbstractFixedRepresentation[Cart3DKeys, CartPos3DDims], AbstractPos3D):
    pass
@final
class CartVel3D(AbstractFixedRepresentation[Cart3DKeys, CartVel3DDims], AbstractVel3D):
    pass
@final
class CartAcc3D(AbstractFixedRepresentation[Cart3DKeys, CartAcc3DDims], AbstractAcc3D):
    pass

cartpos3d: Final = CartPos3D()
cartvel3d: Final = CartVel3D()
cartacc3d: Final = CartAcc3D()

@plum.dispatch
def time_derivative_rep(obj: CartPos3D, /) -> CartVel3D: return cartvel3d
@plum.dispatch
def time_antiderivative_rep(obj: CartVel3D, /) -> CartPos3D: return cartpos3d
@plum.dispatch
def time_derivative_rep(obj: CartVel3D, /) -> CartAcc3D: return cartacc3d
@plum.dispatch
def time_antiderivative_rep(obj: CartAcc3D, /) -> CartVel3D: return cartvel3d


# -----------------------------------------------
# Cylindrical

CylindricalKeys = tuple[L["rho"], L["phi"], L["z"]]
CylindricalPosDims = tuple[Len, Ang, Len]
CylindricalVelDims = tuple[Spd, AngSpd, Spd]
CylindricalAccDims = tuple[Acc, AngAcc, Acc]

@final
class CylindricalPos(AbstractFixedRepresentation[CylindricalKeys, CylindricalPosDims], AbstractPos3D):
    pass
@final
class CylindricalVel(AbstractFixedRepresentation[CylindricalKeys, CylindricalVelDims], AbstractVel3D):
    pass
@final
class CylindricalAcc(AbstractFixedRepresentation[CylindricalKeys, CylindricalAccDims], AbstractAcc3D):
    pass

cylindricalpos: Final = CylindricalPos()
cylindricalvel: Final = CylindricalVel()
cylindricalacc: Final = CylindricalAcc()

@plum.dispatch
def time_derivative_rep(obj: CylindricalPos, /) -> CylindricalVel: return cylindricalvel
@plum.dispatch
def time_antiderivative_rep(obj: CylindricalVel, /) -> CylindricalPos: return cylindricalpos
@plum.dispatch
def time_derivative_rep(obj: CylindricalVel, /) -> CylindricalAcc: return cylindricalacc
@plum.dispatch
def time_antiderivative_rep(obj: CylindricalAcc, /) -> CylindricalVel: return cylindricalvel

# -----------------------------------------------
# Spherical

class AbstractSphericalPos(AbstractPos):
    """Abstract spherical vector representation."""
class AbstractSphericalVel(AbstractVel):
    """Spherical differential representation."""
class AbstractSphericalAcc(AbstractAcc):
    """Spherical acceleration representation."""

SphericalKeys = tuple[L["r"], L["theta"], L["phi"]]
SphericalPosDims = tuple[Len, Ang, Ang]
SphericalVelDims = tuple[Spd, AngSpd, AngSpd]
SphericalAccDims = tuple[Acc, AngAcc, AngAcc]

@final
class SphericalPos(AbstractFixedRepresentation[SphericalKeys, SphericalPosDims], AbstractPos3D):
    def check_data(self, data: Mapping[str, Any], /) -> None:
        super().check_data(data)  # call base check
        checks.polar_range(data["theta"])
@final
class SphericalVel(AbstractFixedRepresentation[SphericalKeys, SphericalVelDims], AbstractVel3D):
    pass
@final
class SphericalAcc(AbstractFixedRepresentation[SphericalKeys, SphericalAccDims], AbstractAcc3D):
    pass

sphericalpos: Final = SphericalPos()
sphericalvel: Final = SphericalVel()
sphericalacc: Final = SphericalAcc()

@plum.dispatch
def time_derivative_rep(obj: SphericalPos, /) -> SphericalVel: return sphericalvel
@plum.dispatch
def time_antiderivative_rep(obj: SphericalVel, /) -> SphericalPos: return sphericalpos
@plum.dispatch
def time_derivative_rep(obj: SphericalVel, /) -> SphericalAcc: return sphericalacc
@plum.dispatch
def time_antiderivative_rep(obj: SphericalAcc, /) -> SphericalVel: return sphericalvel

# -----------------------------------------------
# LonLatSpherical

LonLatSphericalKeys = tuple[L["lon"], L["lat"], L["distance"]]
LonLatSphericalPosDims = tuple[Ang, Ang, Len]
LonLatSphericalVelDims = tuple[AngSpd, AngSpd, Spd]
LonLatSphericalAccDims = tuple[AngAcc, AngAcc, Acc]

@final
class LonLatSphericalPos(AbstractFixedRepresentation[LonLatSphericalKeys, LonLatSphericalPosDims], AbstractPos3D):
    def check_data(self, data: Mapping[str, Any], /) -> None:
        super().check_data(data)  # call base check
        checks.polar_range(data["lat"], -u.Angle(90, "deg"), u.Angle(90, "deg"))
@final
class LonLatSphericalVel(AbstractFixedRepresentation[LonLatSphericalKeys, LonLatSphericalVelDims], AbstractVel3D):
    pass
@final
class LonLatSphericalAcc(AbstractFixedRepresentation[LonLatSphericalKeys, LonLatSphericalAccDims], AbstractAcc3D):
    pass


lonlatsphericalpos: Final = LonLatSphericalPos()
lonlatsphericalvel: Final = LonLatSphericalVel()
lonlatsphericalacc: Final = LonLatSphericalAcc()

@plum.dispatch
def time_derivative_rep(obj: LonLatSphericalPos, /) -> LonLatSphericalVel: return lonlatsphericalvel
@plum.dispatch
def time_antiderivative_rep(obj: LonLatSphericalVel, /) -> LonLatSphericalPos: return lonlatsphericalpos
@plum.dispatch
def time_derivative_rep(obj: LonLatSphericalVel, /) -> LonLatSphericalAcc: return lonlatsphericalacc
@plum.dispatch
def time_antiderivative_rep(obj: LonLatSphericalAcc, /) -> LonLatSphericalVel: return lonlatsphericalvel


# -----------------------------------------------
# LonCosLatSpherical

LonCosLatSphericalKeys = tuple[L["lon_coslat"], L["lat"], L["distance"]]
LonCosLatSphericalPosDims = tuple[Ang, Ang, Len]
LonCosLatSphericalVelDims = tuple[AngSpd, AngSpd, Spd]
LonCosLatSphericalAccDims = tuple[AngAcc, AngAcc, Acc]

@final
class LonCosLatSphericalPos(AbstractFixedRepresentation[LonCosLatSphericalKeys, LonCosLatSphericalPosDims], AbstractPos3D):
    def check_data(self, data: Mapping[str, Any], /) -> None:
        super().check_data(data)  # call base check
        checks.polar_range(data["lat"], -u.Angle(90, "deg"), u.Angle(90, "deg"))
@final
class LonCosLatSphericalVel(AbstractFixedRepresentation[LonCosLatSphericalKeys, LonCosLatSphericalVelDims], AbstractVel3D):
    pass
@final
class LonCosLatSphericalAcc(AbstractFixedRepresentation[LonCosLatSphericalKeys, LonCosLatSphericalAccDims], AbstractAcc3D):
    pass

loncoslatsphericalpos: Final = LonCosLatSphericalPos()
loncoslatsphericalvel: Final = LonCosLatSphericalVel()
loncoslatsphericalacc: Final = LonCosLatSphericalAcc()


# -----------------------------------------------
# MathSpherical

MathSphericalKeys = tuple[L["r"], L["theta"], L["phi"]]

@final
class MathSphericalPos(AbstractFixedRepresentation[MathSphericalKeys, SphericalPosDims], AbstractPos3D):
    def check_data(self, data: Mapping[str, Any], /) -> None:
        super().check_data(data)  # call base check
        checks.polar_range(data["phi"])
@final
class MathSphericalVel(AbstractFixedRepresentation[MathSphericalKeys, SphericalVelDims], AbstractVel3D):
    pass
@final
class MathSphericalAcc(AbstractFixedRepresentation[MathSphericalKeys, SphericalAccDims], AbstractAcc3D):
    pass

mathsphericalpos: Final = MathSphericalPos()
mathsphericalvel: Final = MathSphericalVel()
mathsphericalacc: Final = MathSphericalAcc()

@plum.dispatch
def time_derivative_rep(obj: MathSphericalPos, /) -> MathSphericalVel: return mathsphericalvel
@plum.dispatch
def time_antiderivative_rep(obj: MathSphericalVel, /) -> MathSphericalPos: return mathsphericalpos
@plum.dispatch
def time_derivative_rep(obj: MathSphericalVel, /) -> MathSphericalAcc: return mathsphericalacc
@plum.dispatch
def time_antiderivative_rep(obj: MathSphericalAcc, /) -> MathSphericalVel: return mathsphericalvel

# -----------------------------------------------
# Prolate Spheroidal

ProlateSpheroidalKeys = tuple[L["mu"], L["nu"], L["phi"]]
ProlateSpheroidalPosDims = tuple[L["area"], L["area"], Ang]
ProlateSpheroidalVelDims = tuple[L["diffusivity"], L["diffusivity"], AngSpd]
ProlateSpheroidalAccDims = tuple[L["specific energy"], L["specific energy"], AngAcc]

@final
class ProlateSpheroidalPos(AbstractFixedRepresentation[ProlateSpheroidalKeys, ProlateSpheroidalPosDims], AbstractPos3D):
    _: KW_ONLY
    Delta: Shaped[u.Quantity["length"], ""]
    """Focal length of the coordinate system."""

    def check_data(self, data: Mapping[str, Any], /) -> None:
        super().check_data(data)  # call base check
        checks.strictly_positive(self.Delta, name="Delta")
        checks.geq(
            data["mu"], self.Delta**2, name="mu", comp_name="Delta^2"
        )
        checks.leq(
            jnp.abs(data["nu"]), self.Delta**2, name="nu", comp_name="Delta^2"
        )
@final
class ProlateSpheroidalVel(AbstractFixedRepresentation[ProlateSpheroidalKeys, ProlateSpheroidalVelDims], AbstractVel3D):
    pass
@final
class ProlateSpheroidalAcc(AbstractFixedRepresentation[ProlateSpheroidalKeys, ProlateSpheroidalAccDims], AbstractAcc3D):
    pass

prolatespheroidalpos: Final = ProlateSpheroidalPos()
prolatespheroidalvel: Final = ProlateSpheroidalVel()
prolatespheroidalacc: Final = ProlateSpheroidalAcc()

@plum.dispatch
def time_derivative_rep(obj: ProlateSpheroidalPos, /) -> ProlateSpheroidalVel: return prolatespheroidalvel
@plum.dispatch
def time_antiderivative_rep(obj: ProlateSpheroidalVel, /) -> ProlateSpheroidalPos: return prolatespheroidalpos
@plum.dispatch
def time_derivative_rep(obj: ProlateSpheroidalVel, /) -> ProlateSpheroidalAcc: return prolatespheroidalacc
@plum.dispatch
def time_antiderivative_rep(obj: ProlateSpheroidalAcc, /) -> ProlateSpheroidalVel: return prolatespheroidalvel



# =========================================================
# 4D

SpaceTimeKeys = tuple[L["t"], L["x"], L["y"], L["z"]]
SpaceTimeDims = tuple[L["time"], Len, Len, Len]

@final
@dataclass(frozen=True, slots=True)
class CartSpaceTime(AbstractFixedRepresentation[SpaceTimeKeys, SpaceTimeDims], AbstractPos):
    """Representation for four-vectors."""

    _: KW_ONLY
    c: Shaped[u.Quantity["speed"], ""] = field(default=u.Quantity(299_792.458, "km/s"))
    """Speed of light, by default ``Quantity(299_792.458, "km/s")``."""

cartspacetime: Final = CartSpaceTime()


@final
@dataclass(frozen=True, slots=True)
class SpaceTime(AbstractRepresentation[Ks, Ds], AbstractPos):

    spatial_kind: AbstractPos
    """Spatial part of the representation."""

    @property
    def components(self) -> tuple[str, ...]:  # type: ignore[override]
        return ("t", *self.spatial_kind.components)

    @property
    def dimensions(self) -> tuple[str | None, ...]:  # type: ignore[override]
        return ("time", *self.spatial_kind.dimensions)

# =========================================================
# Poincare

PoincarePolarKeys = tuple[
    L["rho"], L["pp_phi"], L["z"], L["dt_rho"], L["dt_pp_phi"], L["dt_z"]
]

PoincarePolarDims = tuple[
    Len, L["length / time**0.5"], Len, Spd, L["length / time**1.5"], Spd
]

@final
class PoincarePolarRep(AbstractFixedRepresentation[PoincarePolarKeys, PoincarePolarDims], AbstractPos):
    pass

poincarepolar: Final = PoincarePolarRep()

# =========================================================
# N-Dimensional

# -----------------------------------------------
# Cartesian

CartNDKeys = tuple[L["q"]]
CartPosNDDims = tuple[Len]
CartVelNDDims = tuple[Spd]
CartAccNDDims = tuple[Acc]

@final
class CartPosND(AbstractFixedRepresentation[CartNDKeys, CartPosNDDims], AbstractPos):
    pass
@final
class CartVelND(AbstractFixedRepresentation[CartNDKeys, CartVelNDDims], AbstractVel):
    pass
@final
class CartAccND(AbstractFixedRepresentation[CartNDKeys, CartAccNDDims], AbstractAcc):
    pass

cartndpos: Final = CartPosND()
cartndvel: Final = CartVelND()
cartndacc: Final = CartAccND()

@plum.dispatch
def cartesian_rep(obj: CartPosND, /) -> CartPosND:
    return CartPosND
@plum.dispatch
def cartesian_rep(obj: CartVelND, /) -> CartVelND:
    return cartndvel
@plum.dispatch
def cartesian_rep(obj: CartAccND, /) -> CartAccND:
    return cartndacc
