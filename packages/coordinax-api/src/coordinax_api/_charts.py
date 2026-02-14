"""Vector API for coordinax."""

__all__ = ("cartesian_chart", "guess_chart", "cdict")

from typing import TYPE_CHECKING, Any

import plum

from ._custom_types import CsDict

if TYPE_CHECKING:
    import coordinax.charts  # noqa: ICN001


@plum.dispatch.abstract
def cartesian_chart(obj: Any, /) -> "coordinax.charts.AbstractChart":  # type: ignore[type-arg]
    r"""Return the corresponding Cartesian chart for a given chart.

    This function provides the canonical flat-space Cartesian chart
    associated with any coordinate system. It maps each chart to its
    natural Cartesian equivalent in the same dimensional space.

    Mathematical Definition:

    For a coordinate chart $\mathcal{R}$ in $n$-dimensional
    space, this returns the Cartesian chart $\mathcal{C}_n$ such that:

    $$ \mathrm{cartesian\_chart}(\mathcal{R}) = \mathcal{C}_n $$

    where $\mathcal{C}_n \in \{\text{Cart1D}, \text{Cart2D}, \text{Cart3D},
    \text{CartND}\}$ depending on $n = \text{ndim}$.

    The Cartesian chart uses orthonormal basis vectors with components
    typically denoted $(x)$, $(x, y)$, $(x, y, z)$, or
    $(q_1, \ldots, q_n)$ for arbitrary dimension.

    Parameters
    ----------
    obj : Any
        A coordinate chart instance (e.g., `coordinax.charts.sph3d`,
        `coordinax.charts.polar2d`) or any object for which a Cartesian
        equivalent is defined.

    Returns
    -------
    coordinax.charts.AbstractChart
        The Cartesian chart in the same dimensional space:

        - 1D charts → ``Cart1D`` (component: ``x``)
        - 2D charts → ``Cart2D`` (components: ``x``, ``y``)
        - 3D charts → ``Cart3D`` (components: ``x``, ``y``, ``z``)
        - N-D charts → ``CartND`` (components: ``q``)

    Raises
    ------
    NotImplementedError
        If no Cartesian chart is defined for the input object.

    Notes
    -----
    - Cartesian charts use the Euclidean metric with orthonormal bases.
    - This function does **not** perform coordinate transformation; it only
      returns the chart type. Use ``point_transform`` for actual
      coordinate conversion.
    - All standard Euclidean coordinate systems (spherical, cylindrical, polar)
      map to their dimensional Cartesian equivalent.
    - For embedded manifolds, this returns the Cartesian form of the ambient
      space, not the intrinsic coordinates.

    See Also
    --------
    point_transform : Transform coordinates between charts
    coordinax.charts.AbstractChart : Base class for coordinate charts

    Examples
    --------
    >>> import coordinax.charts as cxc

    1D coordinate systems map to Cart1D:

    >>> cxc.cartesian_chart(cxc.cart1d)
    Cart1D(...)

    >>> cxc.cartesian_chart(cxc.radial1d)
    Cart1D(...)

    2D coordinate systems map to Cart2D:

    >>> cxc.cartesian_chart(cxc.cart2d)
    Cart2D(...)

    >>> cxc.cartesian_chart(cxc.polar2d)
    Cart2D(...)

    3D coordinate systems map to Cart3D:

    >>> cxc.cartesian_chart(cxc.cart3d)
    Cart3D(...)

    >>> cxc.cartesian_chart(cxc.sph3d)
    Cart3D(...)

    >>> cxc.cartesian_chart(cxc.cyl3d)
    Cart3D(...)

    N-dimensional systems map to CartND:

    >>> cxc.cartesian_chart(cxc.cartnd)
    CartND(...)

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def guess_chart(*args: Any) -> "coordinax.charts.AbstractChart":  # type: ignore[type-arg]
    """Infer a Cartesian chart from the shape of a value / quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> q = u.Q([1.0, 2.0, 3.0], "m")
    >>> cxc.guess_chart(q)
    Cart3D()

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch.abstract
def cdict(obj: Any, /) -> CsDict:
    """Extract component dictionary from an object.

    This function converts various coordinate charts into a component
    dictionary where keys are component names and values are the corresponding
    values.

    Parameters
    ----------
    obj
        An object to extract a component dictionary from. Supported types include:
        - Vector: extracted from ``obj.data``
        - unxt.Quantity: treated as Cartesian coordinates with components in the
          last dimension, matched to the appropriate Cartesian chart
        - Mappings: returned as-is

    Returns
    -------
    dict[str, Any]
        A mapping from component names to values.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    Extract from a Vector:

    >>> vec = cx.Vector.from_(u.Q([1, 2, 3], "m"))
    >>> d = cx.cdict(vec)
    >>> list(d.keys())
    ['x', 'y', 'z']

    Extract from a Quantity treated as Cartesian:

    >>> q = u.Q([1, 2, 3], "m")
    >>> d = cx.cdict(q)
    >>> list(d.keys())
    ['x', 'y', 'z']

    """
    raise NotImplementedError  # pragma: no cover
