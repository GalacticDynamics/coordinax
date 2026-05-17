"""Tests for predefined chart instances in coordinax.charts."""

import equinox as eqx
import pytest

import unxt as u

import coordinax.charts as cxc

# =============================================================================
# Data tables
# =============================================================================

# (id, chart, class, components, coord_dimensions, ndim)
_CHART_PARAMS: list[tuple] = [
    # 0D
    ("cart0d", cxc.cart0d, cxc.Cart0D, (), (), 0),
    # 1D
    ("cart1d", cxc.cart1d, cxc.Cart1D, ("x",), ("length",), 1),
    ("radial1d", cxc.radial1d, cxc.Radial1D, ("r",), ("length",), 1),
    ("time1d", cxc.time1d, cxc.Time1D, ("t",), ("time",), 1),
    # 2D Euclidean
    ("cart2d", cxc.cart2d, cxc.Cart2D, ("x", "y"), ("length", "length"), 2),
    ("polar2d", cxc.polar2d, cxc.Polar2D, ("r", "theta"), ("length", "angle"), 2),
    # 2D Two-sphere (curved manifold — no global Cartesian)
    ("sph2", cxc.sph2, cxc.SphericalTwoSphere, ("theta", "phi"), ("angle", "angle"), 2),
    (
        "lonlat_sph2",
        cxc.lonlat_sph2,
        cxc.LonLatSphericalTwoSphere,
        ("lon", "lat"),
        ("angle", "angle"),
        2,
    ),
    (
        "loncoslat_sph2",
        cxc.loncoslat_sph2,
        cxc.LonCosLatSphericalTwoSphere,
        ("lon_coslat", "lat"),
        ("angle", "angle"),
        2,
    ),
    (
        "math_sph2",
        cxc.math_sph2,
        cxc.MathSphericalTwoSphere,
        ("theta", "phi"),
        ("angle", "angle"),
        2,
    ),
    # 3D
    (
        "cart3d",
        cxc.cart3d,
        cxc.Cart3D,
        ("x", "y", "z"),
        ("length", "length", "length"),
        3,
    ),
    (
        "cyl3d",
        cxc.cyl3d,
        cxc.Cylindrical3D,
        ("rho", "phi", "z"),
        ("length", "angle", "length"),
        3,
    ),
    (
        "sph3d",
        cxc.sph3d,
        cxc.Spherical3D,
        ("r", "theta", "phi"),
        ("length", "angle", "angle"),
        3,
    ),
    (
        "lonlat_sph3d",
        cxc.lonlat_sph3d,
        cxc.LonLatSpherical3D,
        ("lon", "lat", "distance"),
        ("angle", "angle", "length"),
        3,
    ),
    (
        "loncoslat_sph3d",
        cxc.loncoslat_sph3d,
        cxc.LonCosLatSpherical3D,
        ("lon_coslat", "lat", "distance"),
        ("angle", "angle", "length"),
        3,
    ),
    (
        "math_sph3d",
        cxc.math_sph3d,
        cxc.MathSpherical3D,
        ("r", "theta", "phi"),
        ("length", "angle", "angle"),
        3,
    ),
    # 4D spacetime
    (
        "minkowskict",
        cxc.minkowskict,
        cxc.MinkowskiCT,
        ("ct", "x", "y", "z"),
        ("length", "length", "length", "length"),
        4,
    ),
    # 6D Poincaré polar
    (
        "poincarepolar6d",
        cxc.poincarepolar6d,
        cxc.PoincarePolar6D,
        ("rho", "pp_phi", "z", "dt_rho", "dt_pp_phi", "dt_z"),
        (
            "length",
            "length / time**0.5",
            "length",
            "speed",
            "length / time**1.5",
            "speed",
        ),
        6,
    ),
    # N-D
    ("cartnd", cxc.cartnd, cxc.CartND, ("q",), ("length",), 1),
]

_CHART_IDS = [p[0] for p in _CHART_PARAMS]

# (id, chart, expected_cartesian) — charts that have a global Cartesian
_CARTESIAN_PARAMS: list[tuple] = [
    ("cart0d", cxc.cart0d, cxc.cart0d),
    ("cart1d", cxc.cart1d, cxc.cart1d),
    ("radial1d", cxc.radial1d, cxc.cart1d),
    ("time1d", cxc.time1d, cxc.time1d),
    ("cart2d", cxc.cart2d, cxc.cart2d),
    ("polar2d", cxc.polar2d, cxc.cart2d),
    ("cart3d", cxc.cart3d, cxc.cart3d),
    ("cyl3d", cxc.cyl3d, cxc.cart3d),
    ("sph3d", cxc.sph3d, cxc.cart3d),
    ("lonlat_sph3d", cxc.lonlat_sph3d, cxc.cart3d),
    ("loncoslat_sph3d", cxc.loncoslat_sph3d, cxc.cart3d),
    ("math_sph3d", cxc.math_sph3d, cxc.cart3d),
    ("minkowskict", cxc.minkowskict, cxc.minkowskict),
    ("cartnd", cxc.cartnd, cxc.cartnd),
]

# (id, chart) — charts with no global Cartesian chart
_NO_CARTESIAN_PARAMS: list[tuple] = [
    ("sph2", cxc.sph2),
    ("lonlat_sph2", cxc.lonlat_sph2),
    ("loncoslat_sph2", cxc.loncoslat_sph2),
    ("math_sph2", cxc.math_sph2),
    ("poincarepolar6d", cxc.poincarepolar6d),
]


_DIM_TO_UNIT: dict[str, str] = {
    "length": "m",
    "time": "s",
    "angle": "deg",
    "speed": "m / s",
}


def _quantity_for_dimension(dim: str | None) -> u.AbstractQuantity:
    if dim is None:
        return u.Q(1, "")
    return u.Q(1, _DIM_TO_UNIT[dim])


def _mismatched_quantity_for_dimension(dim: str | None) -> u.AbstractQuantity:
    if dim == "time":
        return u.Q(1, "m")
    return u.Q(1, "s")


def _strict_dimension_component(chart: cxc.AbstractChart) -> str:
    """Component validated by chart-specific values=True checks."""
    if isinstance(chart, cxc.SphericalTwoSphere):
        return "theta"
    if isinstance(chart, cxc.LonLatSphericalTwoSphere):
        return "lat"
    if isinstance(chart, cxc.LonCosLatSphericalTwoSphere):
        return "lat"
    if isinstance(chart, cxc.MathSphericalTwoSphere):
        return "phi"
    if isinstance(chart, cxc.Spherical3D):
        return "theta"
    if isinstance(chart, cxc.LonLatSpherical3D):
        return "lat"
    if isinstance(chart, cxc.LonCosLatSpherical3D):
        return "lat"
    if isinstance(chart, cxc.MathSpherical3D):
        return "phi"
    return chart.components[0]


# =============================================================================
# Type
# =============================================================================


@pytest.mark.parametrize(
    ("chart", "cls"),
    [(p[1], p[2]) for p in _CHART_PARAMS],
    ids=_CHART_IDS,
)
def test_predef_chart_type(chart, cls) -> None:
    """Each predefined chart is an instance of its declared class."""
    assert isinstance(chart, cls)


@pytest.mark.parametrize("chart", [p[1] for p in _CHART_PARAMS], ids=_CHART_IDS)
def test_predef_chart_is_abstract_chart(chart) -> None:
    """Every predefined chart is an AbstractChart."""
    assert isinstance(chart, cxc.AbstractChart)


# =============================================================================
# Coordinate schema
# =============================================================================


@pytest.mark.parametrize(
    ("chart", "expected"),
    [(p[1], p[3]) for p in _CHART_PARAMS],
    ids=_CHART_IDS,
)
def test_predef_chart_components(chart, expected) -> None:
    """Components matches the expected tuple."""
    assert chart.components == expected


@pytest.mark.parametrize(
    ("chart", "expected"),
    [(p[1], p[4]) for p in _CHART_PARAMS],
    ids=_CHART_IDS,
)
def test_predef_chart_coord_dimensions(chart, expected) -> None:
    """coord_dimensions matches the expected tuple."""
    assert chart.coord_dimensions == expected


@pytest.mark.parametrize(
    ("chart", "expected_ndim"),
    [(p[1], p[5]) for p in _CHART_PARAMS],
    ids=_CHART_IDS,
)
def test_predef_chart_ndim(chart, expected_ndim) -> None:
    """Ndim matches the expected value and is consistent with components."""
    assert chart.ndim == expected_ndim
    assert chart.ndim == len(chart.components)
    assert chart.ndim == len(chart.coord_dimensions)


# =============================================================================
# Singleton equality
# =============================================================================


@pytest.mark.parametrize(
    ("chart", "cls"),
    [(p[1], p[2]) for p in _CHART_PARAMS],
    ids=_CHART_IDS,
)
def test_predef_chart_equals_fresh_instance(chart, cls) -> None:
    """The predefined instance equals a freshly constructed instance."""
    assert chart == cls()


# =============================================================================
# Repr
# =============================================================================


@pytest.mark.parametrize(
    ("chart", "cls"),
    [(p[1], p[2]) for p in _CHART_PARAMS],
    ids=_CHART_IDS,
)
def test_predef_chart_repr_contains_class_name(chart, cls) -> None:
    """Repr includes the class name."""
    assert cls.__name__ in repr(chart)


# =============================================================================
# Cartesian chart
# =============================================================================


@pytest.mark.parametrize(
    ("chart", "expected_cartesian"),
    [(p[1], p[2]) for p in _CARTESIAN_PARAMS],
    ids=[p[0] for p in _CARTESIAN_PARAMS],
)
def test_predef_chart_cartesian(chart, expected_cartesian) -> None:
    """chart.cartesian returns the expected global Cartesian chart."""
    assert chart.cartesian == expected_cartesian


@pytest.mark.parametrize(
    "chart",
    [p[1] for p in _NO_CARTESIAN_PARAMS],
    ids=[p[0] for p in _NO_CARTESIAN_PARAMS],
)
def test_predef_chart_no_cartesian_raises(chart) -> None:
    """Charts on curved manifolds raise NoGlobalCartesianChartError."""
    with pytest.raises(cxc.NoGlobalCartesianChartError):
        _ = chart.cartesian


# =============================================================================
# check_data
# =============================================================================


@pytest.mark.parametrize("chart", [p[1] for p in _CHART_PARAMS], ids=_CHART_IDS)
def test_predef_chart_check_data_valid_keys_default_dimensions_false(chart) -> None:
    """check_data defaults to key checks only when values=False."""
    data = {k: u.Q(1, "m") for k in chart.components}
    chart.check_data(data)  # should not raise


@pytest.mark.parametrize("chart", [p[1] for p in _CHART_PARAMS], ids=_CHART_IDS)
def test_predef_chart_check_data_wrong_keys_raises(chart) -> None:
    """check_data raises ValueError when a key is missing."""
    if chart.ndim == 0:
        pytest.skip("0D chart has no components to remove")
    # Drop the last component to trigger the mismatch
    data = {k: u.Q(1, "m") for k in chart.components[:-1]}
    with pytest.raises(ValueError, match="Data keys do not match"):
        chart.check_data(data)


@pytest.mark.parametrize("chart", [p[1] for p in _CHART_PARAMS], ids=_CHART_IDS)
def test_predef_chart_check_data_dimensions_true_with_valid_units(chart) -> None:
    """check_data(values=True) passes for values with compatible dimensions."""
    if any(
        dim not in _DIM_TO_UNIT and dim is not None for dim in chart.coord_dimensions
    ):
        pytest.skip("No parseable unit mapping for one or more chart dimensions")

    data = {
        component: _quantity_for_dimension(dim)
        for component, dim in zip(chart.components, chart.coord_dimensions, strict=True)
    }
    chart.check_data(data, values=True)


@pytest.mark.parametrize("chart", [p[1] for p in _CHART_PARAMS], ids=_CHART_IDS)
def test_predef_chart_check_data_dimensions_true_wrong_unit_raises(chart) -> None:
    """check_data(values=True) raises on dimensional mismatch."""
    if chart.ndim == 0:
        pytest.skip("0D chart has no components")
    if any(
        dim not in _DIM_TO_UNIT and dim is not None for dim in chart.coord_dimensions
    ):
        pytest.skip("No parseable unit mapping for one or more chart dimensions")

    data = {
        component: _quantity_for_dimension(dim)
        for component, dim in zip(chart.components, chart.coord_dimensions, strict=True)
    }
    target_component = _strict_dimension_component(chart)
    target_dim = chart.coord_dimensions[chart.components.index(target_component)]
    data[target_component] = _mismatched_quantity_for_dimension(target_dim)

    with pytest.raises((ValueError, eqx.EquinoxTracetimeError), match=r".+"):
        chart.check_data(data, values=True)


# =============================================================================
# Fixed-component schema consistency (generic)
# =============================================================================


def test_predefined_charts_have_expected_components() -> None:
    """For AbstractFixedComponentsChart, components == class-level _components."""
    for name in cxc.__dir__():
        chart = getattr(cxc, name)

        # TODO: also work with non-fixed charts
        if not isinstance(chart, cxc.AbstractFixedComponentsChart):
            continue

        assert chart.components == type(chart)._components
