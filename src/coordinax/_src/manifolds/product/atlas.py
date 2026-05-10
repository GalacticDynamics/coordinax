"""Product Manifolds."""

__all__ = ("CartesianProductAtlas",)


import dataclasses

from collections.abc import Iterator
from typing import cast, final
from typing_extensions import override

import jax

import coordinax.charts as cxc
from coordinax._src.base_atlas import AbstractAtlas


@jax.tree_util.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True)
class CartesianProductAtlas(AbstractAtlas):
    r"""Atlas for a product manifold.

    The atlas consists of Cartesian product charts formed from the atlases of
    the factor manifolds.

    Examples
    --------
    Consider the product manifold $S^2 \times \\mathbb{R}$, where

    - $S^2$ is the 2-sphere with spherical coordinates $(\theta, \\phi)$ and
      atlas of charts including `SphericalTwoSphere`.
    - $\mathbb{R}$ is the real line with Cartesian coordinate $x$ and atlas of
      charts including `Cartesian1D`.

    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc

    >>> atlas = cxm.CartesianProductAtlas(
    ...     factors=(cxm.HyperSphericalAtlas(), cxm.EuclideanAtlas(1)),
    ...     factor_names=("S2", "R1"))
    >>> atlas
    CartesianProductAtlas(factors=(HyperSphericalAtlas(ndim=2), EuclideanAtlas(ndim=1)),
        factor_names=('S2', 'R1'))

    >>> atlas.ndim
    3

    >>> chart = cxc.CartesianProductChart(
    ...     factors=(cxc.sph2, cxc.cart1d), factor_names=("S2", "R1"))
    >>> chart in atlas
    True

    >>> cxc.sph2 in atlas
    False

    >>> atlas["R1"]  # Access factor atlas by name
    EuclideanAtlas(ndim=1)

    """

    factors: tuple[AbstractAtlas, ...]
    """Factor atlases that define the product atlas."""

    factor_names: tuple[str, ...]
    """Names of the factor atlases, used for indexing and validation."""

    def __post_init__(self) -> None:
        # Validate lengths match
        if len(self.factors) != len(self.factor_names):
            msg = (
                f"factors and factor_names must have the same length, "
                f"got {len(self.factors)} factors and {len(self.factor_names)} names"
            )
            raise ValueError(msg)

        # Validate unique names
        if len(set(self.factor_names)) != len(self.factor_names):
            msg = f"factor_names must be unique, got {self.factor_names}"
            raise ValueError(msg)

    def __iter__(self) -> Iterator[cxc.AbstractChart]:
        """Iterate over charts in the product atlas.

        >>> import coordinax.manifolds as cxm
        >>> import coordinax.charts as cxc
        >>> atlas = cxm.CartesianProductAtlas(
        ...     factors=(cxm.HyperSphericalAtlas(), cxm.EuclideanAtlas(1)),
        ...     factor_names=("S2", "R1"))
        >>> list(iter(atlas))
        [HyperSphericalAtlas(ndim=2), EuclideanAtlas(ndim=1)]

        """
        return cast("Iterator[cxc.AbstractChart]", iter(self.factors))

    def __getitem__(self, idx: str) -> AbstractAtlas:
        """Allow indexing to access factor charts.

        >>> import coordinax.manifolds as cxm
        >>> import coordinax.charts as cxc
        >>> atlas = cxm.CartesianProductAtlas(
        ...     factors=(cxm.HyperSphericalAtlas(), cxm.EuclideanAtlas(1)),
        ...     factor_names=("S2", "R1"))
        >>> atlas["S2"]
        HyperSphericalAtlas(ndim=2)

        """
        return self.factors[self.factor_names.index(idx)]

    def default_chart(self) -> cxc.CartesianProductChart:
        """Return a default chart for the product atlas.

        Examples
        --------
        >>> import coordinax.manifolds as cxm
        >>> atlas = cxm.CartesianProductAtlas(
        ...     factors=(cxm.HyperSphericalAtlas(), cxm.EuclideanAtlas(1)),
        ...     factor_names=("S2", "R1"))
        >>> chart = atlas.default_chart()
        >>> chart
        CartesianProductChart(
            factors=(SphericalTwoSphere(), Cart1D()), factor_names=('S2', 'R1')
        )

        >>> chart["S2"] == cxm.HyperSphericalAtlas().default_chart()
        True

        >>> chart["R1"] == cxm.EuclideanAtlas(ndim=1).default_chart()
        True

        """
        # Get default charts for each factor
        factor_charts = [factor.default_chart() for factor in self.factors]
        # Construct and return the product chart
        return cxc.CartesianProductChart(
            factors=tuple(factor_charts), factor_names=self.factor_names
        )

    def has_chart(self, chart: cxc.AbstractChart) -> bool:
        """Check if the atlas supports the given chart.

        >>> import coordinax.manifolds as cxm
        >>> import coordinax.charts as cxc
        >>> atlas = cxm.CartesianProductAtlas(
        ...     factors=(cxm.HyperSphericalAtlas(), cxm.EuclideanAtlas(1)),
        ...     factor_names=("S2", "R1"))
        >>> chart = cxc.CartesianProductChart(
        ...     factors=(cxc.sph2, cxc.cart1d), factor_names=("S2", "R1"))
        >>> atlas.has_chart(chart)
        True

        """
        return (
            isinstance(chart, cxc.AbstractCartesianProductChart)
            and chart.factor_names == self.factor_names
            and all(
                a.has_chart(c) for a, c in zip(self.factors, chart.factors, strict=True)
            )
        )

    @override
    @property
    def ndim(self) -> int:
        """The sum of the factor atlas' dimensions.

        >>> import coordinax.manifolds as cxm
        >>> atlas = cxm.CartesianProductAtlas(
        ...     factors=(cxm.HyperSphericalAtlas(), cxm.EuclideanAtlas(1)),
        ...     factor_names=("S2", "R1"))
        >>> atlas.ndim
        3

        """
        return sum(f.ndim for f in self.factors)
