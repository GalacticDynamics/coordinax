"""Representations for embedded manifolds."""

__all__ = ("AbstractEmbeddingMap", "CustomEmbeddingMap", "IntrinsicT", "AmbientT")

import abc
import dataclasses

from typing import Any, Generic, Protocol, TypeVar, final, runtime_checkable

import jax

import coordinax.charts as cxc
from coordinax._src.manifolds.custom_types import CDict, OptUSys

IntrinsicT = TypeVar("IntrinsicT", bound=cxc.AbstractChart[Any, Any])
AmbientT = TypeVar("AmbientT", bound=cxc.AbstractChart[Any, Any])


@jax.tree_util.register_static
class AbstractEmbeddingMap(Generic[IntrinsicT, AmbientT], metaclass=abc.ABCMeta):
    r"""Abstract base class representing a smooth embedding.

    An embedding represents a smooth injective map $$ \iota : M \hookrightarrow
    N $$ of an intrinsic manifold (with charts in `coordinax.charts`) into an
    ambient manifold.

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

    """

    intrinsic: IntrinsicT
    ambient: AmbientT  # e.g. Cart3D

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


@runtime_checkable
class EPCallable(Protocol):
    """Protocol for the embed and project callables in CustomEmbeddingMap."""

    def __call__(self, point: CDict, /, *, usys: OptUSys = None) -> CDict: ...


@final
@dataclasses.dataclass(frozen=True, slots=True)
class CustomEmbeddingMap(AbstractEmbeddingMap[IntrinsicT, AmbientT]):
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
    embed_fn: EPCallable
    project_fn: EPCallable

    def embed(self, point: CDict, /, *, usys: OptUSys = None) -> CDict:
        return self.embed_fn(point, usys=usys)

    def project(self, point: CDict, /, *, usys: OptUSys = None) -> CDict:
        return self.project_fn(point, usys=usys)
