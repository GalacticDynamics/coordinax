"""Representations for embedded manifolds."""

__all__ = ("AbstractEmbeddingMap", "CustomEmbeddingMap", "IntrinsicT", "AmbientT")

import abc
import dataclasses

from typing import Any, Generic, Protocol, TypeVar, final, runtime_checkable

import equinox as eqx

from coordinax._src.base import AbstractChart
from coordinax._src.base.charts import is_abstract_class
from coordinax._src.custom_types import CDict, OptUSys

IntrinsicT = TypeVar("IntrinsicT", bound=AbstractChart[Any, Any, Any])
AmbientT = TypeVar("AmbientT", bound=AbstractChart[Any, Any, Any])


class AbstractEmbeddingMap(Generic[IntrinsicT, AmbientT], metaclass=abc.ABCMeta):
    r"""Abstract base class representing a smooth embedding.

    An embedding represents a smooth injective map
    $\iota : M \hookrightarrow N$ of an intrinsic manifold (with charts in
    `coordinax.charts`) into an ambient manifold.

    Conceptually, an embedding provides:

    - A smooth map from intrinsic coordinates ``q^i`` to ambient coordinates
      ``x^a = x^a(q)`` via `embed`.
    - A (possibly local) inverse or projection map from ambient coordinates back
      to intrinsic coordinates via `project`.

    Examples
    --------
    A concrete example is the embedding of ``SphericalTwoSphere`` into
    ``Spherical3D``: the intrinsic coordinates may be ``(θ, φ)`` on the unit
    2-sphere, while the ambient coordinates are ``(r, θ, φ)`` with fixed radius
    ``r = R``. A concrete subclass can therefore:

    - Map ``(θ, φ) ↦ (R, θ, φ)`` in ``Spherical3D`` via `embed`.
    - Drop the radial component via `project`.
    - Realize to Cartesian coordinates by first embedding into ``Spherical3D``
      and then delegating to its Cartesian realization.

    Subclasses are responsible for implementing the coordinate-level maps;
    higher-level metric machinery (e.g. induced metrics) can be built on top of
    this interface.

    This class is deliberately *not* an `equinox.Module` itself: the two
    attributes below are an interface for concrete maps to satisfy with either a
    field (`CustomEmbeddingMap.ambient`) or a property
    (`TwoSphereIn3D.intrinsic`), and a `Module` base would turn them into
    required constructor arguments.

    """

    intrinsic: IntrinsicT
    ambient: AmbientT  # e.g. Cart3D

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        # `eqx.Module` is not inherited from here, so it can be forgotten -- and
        # a non-pytree map is one opaque leaf that hides its parameters from
        # JAX, which is the bug this whole arrangement exists to prevent.
        if not is_abstract_class(cls) and not issubclass(cls, eqx.Module):
            msg = (
                f"{cls.__name__} must subclass `equinox.Module`: an embedding map "
                "holds coordinate values, so it has to be a pytree."
            )
            raise TypeError(msg)

    @abc.abstractmethod
    def embed(self, point: CDict, /, *, usys: OptUSys = None) -> CDict:
        """Embed intrinsic coordinates into ambient coordinates.

        Parameters
        ----------
        point
            A point in intrinsic coordinates.
        usys
            Optional unit system for the input and output coordinates.

        """
        raise NotImplementedError  # pragma: no cover

    @abc.abstractmethod
    def project(self, point: CDict, /, *, usys: OptUSys = None) -> CDict:
        """Project ambient coordinates to intrinsic coordinates.

        Parameters
        ----------
        point
            A point in ambient coordinates.
        usys
            Optional unit system for the input and output coordinates.

        """
        raise NotImplementedError  # pragma: no cover

    def __repr__(self) -> str:
        # The plain dataclass repr, which `eqx.Module` replaces with a
        # Wadler-Lindig one that elides array values and default fields.
        fs = dataclasses.fields(self)
        args = ", ".join(f"{f.name}={getattr(self, f.name)!r}" for f in fs)
        return f"{type(self).__name__}({args})"


@runtime_checkable
class EPCallable(Protocol):
    """Protocol for the embed and project callables in CustomEmbeddingMap."""

    def __call__(self, point: CDict, /, *, usys: OptUSys = None) -> CDict: ...


@final
class CustomEmbeddingMap(AbstractEmbeddingMap[IntrinsicT, AmbientT], eqx.Module):
    """A concrete embedding map defined by user-provided functions.

    This class allows users to define an embedding by providing custom `embed`
    and `project` functions, without needing to create a new subclass.

    Parameters
    ----------
    intrinsic
        The intrinsic chart.
    ambient
        The ambient chart.
    embed_fn
        A function that takes a point in intrinsic coordinates and returns the
        corresponding point in ambient coordinates.
    project_fn
        A function that takes a point in ambient coordinates and returns the
        corresponding point in intrinsic coordinates.

    """

    intrinsic: IntrinsicT
    ambient: AmbientT
    # Plain functions are not JAX types, so they must not be pytree leaves.
    embed_fn: EPCallable = eqx.field(static=True)
    project_fn: EPCallable = eqx.field(static=True)

    def embed(self, point: CDict, /, *, usys: OptUSys = None) -> CDict:
        return self.embed_fn(point, usys=usys)

    def project(self, point: CDict, /, *, usys: OptUSys = None) -> CDict:
        return self.project_fn(point, usys=usys)
