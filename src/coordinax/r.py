"""`coordinax.r` Module.

This module contains the representations and roles used in `coordinax`.

Let's start by constructing some representation objects and see how they work.

>>> from coordinax import r

We can create a 3D Cartesian representation like this:

>>> cart_rep = r.Cart3D()
>>> cart_rep == r.cart3d  # predefined instance
True

Representations have components and coordinate dimensions:

>>> cart_rep.components
('x', 'y', 'z')

>>> cart_rep.coord_dimensions
('length', 'length', 'length')

There are many different representations available, such as spherical
coordinates:

>>> sph_rep = r.Spherical3D()
>>> sph_rep == r.sph3d  # predefined instance
True

>>> sph_rep.components
('r', 'theta', 'phi')
>>> sph_rep.coord_dimensions
('length', 'angle', 'angle')

With a representation we can transform mappings between different coordinate
representations.

>>> import unxt as u
>>> q = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}
>>> q_sph = r.coord_map(r.sph3d, r.cart3d, q)
>>> q_sph
{'r': Quantity(Array(3.7416575, dtype=float32, ...), unit='km'),
 'theta': Quantity(Array(1.1071487, dtype=float32, ...), unit='rad'),
 'phi': Quantity(Array(1.1071487, dtype=float32, ...), unit='rad')}


For a transformation of a physical velocity vector, we can use `diff_map`:

>>> v = {"x": u.Q(10, "km/s"), "y": u.Q(20, "km/s"), "z": u.Q(30, "km/s")}
>>> v_sph = r.diff_map(r.sph3d, r.cart3d, v, q)
>>> v_sph
{'r': Quantity(Array(37.416573, dtype=float32, ...), unit='km/s'),
 'theta': Quantity(Array(5.773502, dtype=float32, ...), unit='rad/s'),
 'phi': Quantity(Array(3.3333333, dtype=float32, ...), unit='rad/s')}


Roles define the physical meaning of vectors, such as position, velocity, and
acceleration:

>>> pos_role = r.Pos()
>>> pos_role == r.pos  # predefined instance
True

With roles, we can use the `coordinax.vconvert` function to convert vectors
between different representations while respecting their physical meaning.

>>> import coordinax as cx
>>> cx.vconvert(r.pos, r.sph3d, r.cart3d, q) == q_sph
True
>>> cx.vconvert(r.vel, r.sph3d, r.cart3d, v, q) == v_sph
True

"""

__all__ = (
    "cartesian_rep",
    # ===============================================
    "AbstractRep",
    "AbstractDimensionalFlag",
    "DIMENSIONAL_FLAGS",
    "AbstractEmbedded",
    "EmbeddedManifold",
    # = 0D ======================================
    "Abstract0D",
    "Cart0D",
    "cart0d",
    # = 1D ======================================
    "Abstract1D",
    "Cart1D",
    "cart1d",
    "Radial1D",
    "radial1d",
    # = 2D ======================================
    "Abstract2D",
    "Cart2D",
    "cart2d",
    "Polar2D",
    "polar2d",
    "TwoSphere",
    "twosphere",
    # = 3D ======================================
    "Abstract3D",
    "Cart3D",
    "cart3d",
    "Cylindrical3D",
    "cyl3d",
    "AbstractSpherical3D",
    "Spherical3D",
    "sph3d",
    "LonLatSpherical3D",
    "lonlatsph3d",
    "LonCosLatSpherical3D",
    "loncoslatsph3d",
    "MathSpherical3D",
    "mathsph3d",
    "ProlateSpheroidal3D",  # Not exported as instance
    # = 6D ======================================
    "Abstract6D",
    "PoincarePolar6D",
    "poincarepolar6d",
    # = N-D =====================================
    "AbstractND",
    "CartND",
    "cartnd",
    "SpaceTimeCT",  # Not exported as instance
    "SpaceTimeEuclidean",  # Not exported as instance
    # ===============================================
    # Roles
    "AbstractRoleFlag",
    "Pos",
    "pos",
    "Vel",
    "vel",
    "Acc",
    "acc",
    "coord_map",
    "diff_map",
    "frame_to_cart",
    "embed_pos",
    "project_pos",
    "embed_dif",
    "project_dif",
    "metric_of",
    # ===============================================
    # Metrics
    "AbstractMetric",
    "EuclideanMetric",
    "MinkowskiMetric",
    "SphereMetric",
)

from .setup_package import RUNTIME_TYPECHECKER, install_import_hook

with install_import_hook("coordinax.r"):
    from ._src.representations import (
        DIMENSIONAL_FLAGS,
        Abstract0D,
        Abstract1D,
        Abstract2D,
        Abstract3D,
        Abstract6D,
        AbstractDimensionalFlag,
        AbstractEmbedded,
        AbstractMetric,
        AbstractND,
        AbstractRep,
        AbstractRoleFlag,
        AbstractSpherical3D,
        Acc,
        Cart0D,
        Cart1D,
        Cart2D,
        Cart3D,
        CartND,
        Cylindrical3D,
        EmbeddedManifold,
        EuclideanMetric,
        LonCosLatSpherical3D,
        LonLatSpherical3D,
        MathSpherical3D,
        MinkowskiMetric,
        PoincarePolar6D,
        Polar2D,
        Pos,
        ProlateSpheroidal3D,
        Radial1D,
        SpaceTimeCT,
        SpaceTimeEuclidean,
        SphereMetric,
        Spherical3D,
        TwoSphere,
        Vel,
        acc,
        cart0d,
        cart1d,
        cart2d,
        cart3d,
        cartesian_rep,
        cartnd,
        coord_map,
        cyl3d,
        diff_map,
        embed_dif,
        embed_pos,
        frame_to_cart,
        loncoslatsph3d,
        lonlatsph3d,
        mathsph3d,
        metric_of,
        poincarepolar6d,
        polar2d,
        pos,
        project_dif,
        project_pos,
        radial1d,
        sph3d,
        twosphere,
        vel,
    )


del install_import_hook, RUNTIME_TYPECHECKER
