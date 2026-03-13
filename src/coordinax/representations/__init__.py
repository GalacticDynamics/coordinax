r"""Representation descriptors for geometric coordinate data.

This subpackage defines the **representation layer** of `coordinax`: static,
immutable objects that describe what coordinate or component data *means*
geometrically.

In `coordinax`, a representation is separate from a chart.

- A **chart** describes how a manifold is coordinatized locally.
- A **representation** describes what sort of geometric object the data
  represents, whether it is basis-dependent, and what semantic interpretation
  it carries.

Concretely, a representation is an ordered triple
$$
R = (K, B, S),
$$
where

- $K$ is the **geometric kind**,
- $B$ is the **basis kind**, and
- $S$ is the **semantic kind**.

For a point, the canonical representation is
$$
(\mathrm{PointGeometry},\, \mathrm{NoBasis},\, \mathrm{Location}),
$$

which indicates that the data represents a point on a manifold, that it is not
expressed in a basis-dependent linear space, and that its semantic meaning is a
location.

This separation is important because the same chart may be used to express
different geometric objects, while the same representation may be expressed in
many different charts. The role of this subpackage is therefore to provide the
metadata needed to interpret geometric data correctly and to dispatch
representation-aware conversions such as `vconvert`.

The main public objects provided here are:

- `Representation`, the full representation descriptor,
- `AbstractGeometry` and concrete geometric kinds such as `PointGeometry`,
- `AbstractBasis` and concrete basis kinds such as `NoBasis`,
- `AbstractSemanticKind` and concrete semantic kinds such as `Location`, and
- `vconvert`, the central conversion function that converts data between charts
  and, more generally, between compatible representations.

Now let's work through some examples.

>>> import coordinax.representations as cxr

Construct the canonical point representation explicitly:

>>> rep = cxr.Representation(cxr.PointGeometry(), cxr.NoBasis(), cxr.Location())
>>> rep
Representation(geom_kind=PointGeometry(), basis=NoBasis(), semantic_kind=Location())

This is also provided as a convenient pre-defined instance.

>>> rep == cxr.point
True

Representations are used together with charts to convert geometric data while
preserving its interpretation:

>>> import coordinax.charts as cxc
>>> p = {"x": 1.0, "y": 2.0, "z": 3.0}
>>> q = cxr.vconvert(cxc.sph3d, cxr.point, cxc.cart3d, cxr.point, p)
>>> q
{'r': Array(3.74165739, dtype=float64, ...),
 'theta': Array(0.64052231, dtype=float64),
 'phi': Array(1.10714872, dtype=float64, ...)}

Here `p` is interpreted as point data in Cartesian 3D coordinates, and `q` is
the same point expressed in spherical coordinates.

The important distinction is that the chart changes, while the representation
remains the same:

>>> from_rep = cxr.point
>>> to_rep = cxr.point
>>> q = cxr.vconvert(cxc.sph3d, to_rep, cxc.cart3d, from_rep, p)

This reflects the current design focus of `coordinax.representations`: point
data first, with the representation structure already in place for later
extension to tangent, cotangent, and other geometric objects.

"""

__all__ = (
    "vconvert",
    "guess_rep",
    # Representations
    "Representation",
    "point",
    # Geometric Kinds
    "AbstractGeometry",
    "PointGeometry",
    "point_geom",
    # Coordinate Bases
    "AbstractBasis",
    "NoBasis",
    "nobasis",
    # Semantic Kinds
    "AbstractSemanticKind",
    "Location",
    "location",
)

from ._setup_package import install_import_hook

with install_import_hook("coordinax.representations"):
    from ._src import (
        AbstractBasis,
        AbstractGeometry,
        AbstractSemanticKind,
        Location,
        NoBasis,
        PointGeometry,
        Representation,
        location,
        nobasis,
        point,
        point_geom,
    )
    from coordinax.api.representations import guess_rep, vconvert


del install_import_hook
