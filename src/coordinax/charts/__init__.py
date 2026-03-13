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

Another example is a 4D spacetime chart:

>>> st_chart = cxc.SpaceTimeCT()
>>> st_chart.components
('ct', 'x', 'y', 'z')
>>> st_chart.coord_dimensions
('length', 'length', 'length', 'length')
>>> st_chart.time_chart
Time1D()
>>> st_chart.spatial_chart
Cart3D()

>>> cxc.SpaceTimeCT(cxc.sph3d)
SpaceTimeCT(spatial_chart=Spherical3D())

`SpaceTimeCT` is a special case of a Cartesian product chart. It has a fixed
time factor `time1d` and a user-selectable spatial factor and flattens its chart
factors into a "single" chart.

We can also build arbitrary Cartesian products of charts (without flattening)
using `CartesianProductChart`:

>>> prod_chart = cxc.CartesianProductChart((cxc.time1d, cxc.sph3d), ("t", "q"))
>>> prod_chart
CartesianProductChart(
    factors=(Time1D(), Spherical3D()), factor_names=('t', 'q')
)
>>> prod_chart.components
('t.t', 'q.r', 'q.theta', 'q.phi')

As a preview of how {mod}`coordinax.charts` work with the other modules in
`coordinax`, let's see how we can use charts to compute a metric matrix and a
norm.

With charts we can transform mappings between different coordinate systems.

>>> import unxt as u
>>> q = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}
>>> q_sph = cxc.point_realization_map(cxc.sph3d, cxc.cart3d, q)
>>> q_sph
{'r': Q(3.74165739, 'km'), 'theta': Q(0.64052231, 'rad'),
 'phi': Q(1.10714872, 'rad')}

"""

__all__ = (
    # ===========================================
    "AbstractChart",
    "AbstractFixedComponentsChart",
    "AbstractDimensionalFlag",
    "DIMENSIONAL_FLAGS",
    "AbstractCartesianProductChart",
    "AbstractFlatCartesianProductChart",
    "CartesianProductChart",
    # -------------------------------------------
    "NoGlobalCartesianChartError",
    # -------------------------------------------
    "cartesian_chart",
    "guess_chart",
    "cdict",
    "point_realization_map",
    "realize_cartesian",
    "point_transition_map",
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
    "SphericalTwoSphere",
    "sph2",
    "LonLatSphericalTwoSphere",
    "lonlat_sph2",
    "LonCosLatSphericalTwoSphere",
    "loncoslat_sph2",
    "MathSphericalTwoSphere",
    "math_sph2",
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
    "lonlat_sph3d",
    "LonCosLatSpherical3D",
    "loncoslat_sph3d",
    "MathSpherical3D",
    "math_sph3d",
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
)

from ._setup_package import install_import_hook

with install_import_hook("coordinax.charts"):
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
        LonCosLatSphericalTwoSphere,
        LonLatSpherical3D,
        LonLatSphericalTwoSphere,
        MathSpherical3D,
        MathSphericalTwoSphere,
        NoGlobalCartesianChartError,
        PoincarePolar6D,
        Polar2D,
        ProlateSpheroidal3D,
        Radial1D,
        SpaceTimeCT,
        Spherical3D,
        SphericalTwoSphere,
        Time1D,
        cart0d,
        cart1d,
        cart2d,
        cart3d,
        cartnd,
        cyl3d,
        loncoslat_sph2,
        loncoslat_sph3d,
        lonlat_sph2,
        lonlat_sph3d,
        math_sph2,
        math_sph3d,
        poincarepolar6d,
        polar2d,
        radial1d,
        sph2,
        sph3d,
        time1d,
    )
    from coordinax.api.charts import (
        cartesian_chart,
        cdict,
        guess_chart,
        point_realization_map,
        point_transition_map,
        realize_cartesian,
    )


del install_import_hook
