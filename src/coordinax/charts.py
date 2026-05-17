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

Another example is a 4D Minkowski spacetime chart:

>>> st_chart = cxc.MinkowskiCT()
>>> st_chart.components
('ct', 'x', 'y', 'z')
>>> st_chart.coord_dimensions
('length', 'length', 'length', 'length')
>>> st_chart.cartesian is st_chart
True

We can also build arbitrary Cartesian products of charts (without flattening)
using `CartesianProductChart`:

>>> prod_chart = cxc.CartesianProductChart((cxc.time1d, cxc.sph3d), ("t", "q"))
>>> prod_chart
CartesianProductChart(
    factors=(Time1D(M=Rn(1)), Spherical3D(M=Rn(3))),
    factor_names=('t', 'q')
)
>>> prod_chart.components
('t.t', 'q.r', 'q.theta', 'q.phi')

With charts we can transform point coordinates between different coordinate
systems.

>>> import unxt as u
>>> q = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}
>>> q_sph = cxc.pt_map(q, cxc.cart3d, cxc.sph3d)
>>> q_sph
{'r': Q(3.74165739, 'km'), 'theta': Q(0.64052231, 'rad'),
 'phi': Q(1.10714872, 'rad')}

We can also compute the Jacobian of the point map:

>>> jac = cxc.jac_pt_map(q, cxc.cart3d, cxc.sph3d)

"""

__all__ = (
    # ===========================================
    "AbstractChart",
    "AbstractFixedComponentsChart",
    "AbstractDimensionalFlag",
    "DIMENSIONAL_FLAGS",
    "CHART_CLASSES",
    "NoGlobalCartesianChartError",
    # -------------------------------------------
    "cartesian_chart",
    "guess_chart",
    "cdict",
    "jac_pt_map",
    "pt_map",
    # ===========================================
    # R^n
    # - 0D --------------------------------------
    "Abstract0D",
    "Cart0D",
    "cart0d",
    # - 1D --------------------------------------
    "Abstract1D",
    "Cart1D",
    "cart1d",
    "Radial1D",
    "radial1d",
    "Time1D",
    "time1d",
    # - 2D --------------------------------------
    "Abstract2D",
    "Cart2D",
    "cart2d",
    "Polar2D",
    "polar2d",
    # - 3D --------------------------------------
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
    # - 4D --------------------------------------
    "Abstract4D",
    # - 6D --------------------------------------
    "Abstract6D",
    "PoincarePolar6D",
    "poincarepolar6d",
    # - N-D -------------------------------------
    "AbstractND",
    "CartND",
    "cartnd",
    # ===========================================
    # S^n
    "AbstractSphericalHyperSphere",
    "AbstractSphericalOneSphere",
    # - 1D --------------------------------------
    "CircularOneSphere",
    "sph1",
    # - 2D --------------------------------------
    "AbstractSphericalTwoSphere",
    "SphericalTwoSphere",
    "sph2",
    "LonLatSphericalTwoSphere",
    "lonlat_sph2",
    "LonCosLatSphericalTwoSphere",
    "loncoslat_sph2",
    "MathSphericalTwoSphere",
    "math_sph2",
    # ===========================================
    "AbstractCartesianProductChart",
    "AbstractFlatCartesianProductChart",
    "CartesianProductChart",
    "GalileanCT",
    "galileanct",
    "MinkowskiCT",
    "minkowskict",
)

from coordinax._src.setup_package import install_import_hook

with install_import_hook("coordinax.charts"):
    from coordinax._src.base import (
        CHART_CLASSES,
        DIMENSIONAL_FLAGS,
        AbstractChart,
        AbstractDimensionalFlag,
        AbstractFixedComponentsChart,
    )
    from coordinax._src.charts import (
        Abstract0D,
        Abstract1D,
        Abstract2D,
        Abstract3D,
        Abstract4D,
        Abstract6D,
        AbstractND,
        AbstractSpherical3D,
        Cart0D,
        Cart1D,
        Cart2D,
        Cart3D,
        CartND,
        Cylindrical3D,
        LonCosLatSpherical3D,
        LonLatSpherical3D,
        MathSpherical3D,
        PoincarePolar6D,
        Polar2D,
        ProlateSpheroidal3D,
        Radial1D,
        Spherical3D,
        Time1D,
        cart0d,
        cart1d,
        cart2d,
        cart3d,
        cartnd,
        cyl3d,
        jac_pt_map,
        loncoslat_sph3d,
        lonlat_sph3d,
        math_sph3d,
        poincarepolar6d,
        polar2d,
        radial1d,
        sph3d,
        time1d,
    )
    from coordinax._src.exceptions import NoGlobalCartesianChartError
    from coordinax._src.minkowski import MinkowskiCT, minkowskict
    from coordinax._src.product import (
        AbstractCartesianProductChart,
        AbstractFlatCartesianProductChart,
        CartesianProductChart,
        GalileanCT,
        galileanct,
    )
    from coordinax._src.spherical import (
        AbstractSphericalHyperSphere,
        AbstractSphericalOneSphere,
        AbstractSphericalTwoSphere,
        CircularOneSphere,
        LonCosLatSphericalTwoSphere,
        LonLatSphericalTwoSphere,
        MathSphericalTwoSphere,
        SphericalTwoSphere,
        loncoslat_sph2,
        lonlat_sph2,
        math_sph2,
        sph1,
        sph2,
    )
    from coordinax.api.charts import cartesian_chart, cdict, guess_chart, pt_map


del install_import_hook
