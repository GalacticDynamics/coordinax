"""Property tests for physical_tangent_transform round-trip."""

import equinox as eqx
from hypothesis import HealthCheck, given, reject, settings, strategies as st

import quaxed.numpy as jnp
import unxt as u

import coordinax.charts as cxc
import coordinax.roles as cxr
import coordinax.transforms as cxt
import coordinax_hypothesis as cxst
from coordinax._src.custom_types import CsDict

# TODO: dynamically determine supported charts
FRAME_CART_SUPPORTED = (
    cxc.Cart1D,
    cxc.Cart2D,
    cxc.Cart3D,
    cxc.Cylindrical3D,
    cxc.Spherical3D,
    cxc.LonLatSpherical3D,
    cxc.MathSpherical3D,
    cxc.LonCosLatSpherical3D,
)


def base_point_valid(chart: cxc.AbstractChart, point: CsDict, /) -> bool:
    """Reject base points at coordinate singularities or outside valid ranges."""
    try:
        chart.check_data(point)
    except (TypeError, ValueError, eqx.EquinoxRuntimeError):
        return False
    return True


@st.composite
def chart_pairs_same_dim(
    draw: st.DrawFn,
) -> tuple[cxc.AbstractChart, cxc.AbstractChart]:
    """Draw two charts with matching dimensionality to avoid unsupported conversions."""
    dim = draw(st.sampled_from([1, 2, 3]))
    chart_strategy = cxst.charts(
        ndim=dim,
        exclude=(
            cxc.EmbeddedManifold,
            cxc.TwoSphere,
            cxc.SpaceTimeCT,
            cxc.SpaceTimeEuclidean,
        ),
    )
    charts = draw(chart_strategy), draw(chart_strategy)
    # Hypothesis sometimes still draws embedded manifolds despite the exclude
    # filter; reject those cases to avoid recursive cartesian fallback.
    if any(isinstance(c, (cxc.EmbeddedManifold, cxc.TwoSphere)) for c in charts):
        reject()
    # Restrict to charts with a registered frame_cart rule to avoid
    # NotImplemented errors when mapping velocities.
    if not all(isinstance(c, FRAME_CART_SUPPORTED) for c in charts):
        reject()
    return charts


@settings(
    max_examples=20, deadline=None, suppress_health_check=[HealthCheck.filter_too_much]
)
@given(data=st.data(), charts_pair=chart_pairs_same_dim())
def test_physical_tangent_roundtrip_velocity(charts_pair, data):
    """Round-trip property for velocity via physical_tangent_transform."""
    chart_A, chart_B = charts_pair
    # Base point and velocity in chart_B
    p_B = data.draw(cxst.cdicts(chart_B, cxr.point))
    if not base_point_valid(chart_B, p_B):
        reject()
    vel_elems = st.floats(
        min_value=-1e2,
        max_value=1e2,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    )
    v_B = data.draw(cxst.cdicts(chart_B, cxr.phys_vel, elements=vel_elems))

    # Forward transform to chart_A
    v_A = cxt.physical_tangent_transform(chart_A, chart_B, v_B, at=p_B)
    # Transform base point
    p_A = cxt.point_transform(chart_A, chart_B, p_B)
    # Back transform to chart_B
    v_B2 = cxt.physical_tangent_transform(chart_B, chart_A, v_A, at=p_A)

    # Compare component values (with tolerance for floating point precision)
    # Note: Numerical errors on the order of 1e-6 observed in practice,
    # likely due to accumulated floating point operations in the Jacobian calculations
    for k in chart_B.components:
        assert jnp.allclose(
            u.ustrip(v_B[k].unit, v_B2[k]),
            u.ustrip(v_B[k].unit, v_B[k]),
            rtol=1e-5,
            atol=1e-4,
        )
