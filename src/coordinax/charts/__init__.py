"""`coordinax.charts` Module.

Let's start by constructing some charts and see how they work.

>>> import coordinax.charts as cxc

We can create a 3D Cartesian chart like this:

>>> cart_chart = cxc.Cart3D()
>>> cart_chart == cxc.cart3d  # predefined instance
True

Charts have components and coordinate dimensions:

>>> cart_chart.components
('x', 'y', 'z')

>>> cart_chart.coord_dimensions
('length', 'length', 'length')

There are many different charts available, such as spherical coordinates:

>>> sph_chart = cxc.Spherical3D()
>>> sph_chart == cxc.sph3d  # predefined instance
True

>>> sph_chart.components
('r', 'theta', 'phi')
>>> sph_chart.coord_dimensions
('length', 'angle', 'angle')

With charts we can transform mappings between different coordinate systems.

>>> import unxt as u
>>> import coordinax.transforms as cxt
>>> q = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}
>>> q_sph = cxt.point_transform(cxc.sph3d, cxc.cart3d, q)
>>> q_sph
{'r': Quantity(Array(3.74165739, dtype=float64, ...), unit='km'),
 'theta': Quantity(Array(0.64052231, dtype=float64), unit='rad'),
 'phi': Quantity(Array(1.10714872, dtype=float64, ...), unit='rad')}


For a transformation of a physical velocity vector, we can use
`physical_tangent_transform`:

>>> import jax
>>> v = {"x": u.Q(10, "km/s"), "y": u.Q(20, "km/s"), "z": u.Q(30, "km/s")}
>>> v_sph = cxt.physical_tangent_transform(cxc.sph3d, cxc.cart3d, v, at=q)
>>> jax.tree.map(lambda x: x.round(2), v_sph)
{'phi': Quantity(Array(0., dtype=float64), unit='km / s'),
 'r': Quantity(Array(37.42, dtype=float64), unit='km / s'),
 'theta': Quantity(Array(0., dtype=float64), unit='km / s')}


Roles define the physical meaning of vectors, such as point,
position-difference, velocity, and acceleration:

>>> import coordinax.roles as cxr
>>> point_role = cxr.Point()
>>> point_role == cxr.point  # predefined instance
True

With roles, we can use the `coordinax.vconvert` function to convert vectors
between different representations while respecting their physical meaning.

>>> import coordinax as cx
>>> cx.vconvert(cxr.point, cxc.sph3d, cxc.cart3d, q) == q_sph
True
>>> cx.vconvert(cxr.phys_vel, cxc.sph3d, cxc.cart3d, v, q) == v_sph
True

"""

__all__ = (
    # ===============================================
    "AbstractChart",
    "AbstractFixedComponentsChart",
    "AbstractDimensionalFlag",
    "DIMENSIONAL_FLAGS",
    "AbstractCartesianProductChart",
    "AbstractFlatCartesianProductChart",
    "CartesianProductChart",
    "cartesian_chart",
    "guess_chart",
    "cdict",
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
    "Time1D",
    "time1d",
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
)

from coordinax import setup_package

with setup_package.install_import_hook("coordinax.charts"):
    from ._src import (
        DIMENSIONAL_FLAGS,
        Abstract0D,
        Abstract1D,
        Abstract2D,
        Abstract3D,
        Abstract6D,
        AbstractCartesianProductChart,
        AbstractChart,
        AbstractDimensionalFlag,
        AbstractFixedComponentsChart,
        AbstractFlatCartesianProductChart,
        AbstractND,
        AbstractSpherical3D,
        Cart0D,
        Cart1D,
        Cart2D,
        Cart3D,
        CartesianProductChart,
        CartND,
        Cylindrical3D,
        LonCosLatSpherical3D,
        LonLatSpherical3D,
        MathSpherical3D,
        PoincarePolar6D,
        Polar2D,
        ProlateSpheroidal3D,
        Radial1D,
        SpaceTimeCT,
        SpaceTimeEuclidean,
        Spherical3D,
        Time1D,
        TwoSphere,
        cart0d,
        cart1d,
        cart2d,
        cart3d,
        cartnd,
        cyl3d,
        loncoslatsph3d,
        lonlatsph3d,
        mathsph3d,
        poincarepolar6d,
        polar2d,
        radial1d,
        sph3d,
        time1d,
        twosphere,
    )
    from coordinax.api import cartesian_chart, cdict, guess_chart


del setup_package
