"""Property tests for physical_tangent_transform round-trip."""

from hypothesis import HealthCheck, given, reject, settings, strategies as st

import quaxed.numpy as jnp
import unxt as u

import coordinax as cx
import coordinax.charts as cxc
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
    cxc.SpaceTimeCT,
    cxc.SpaceTimeEuclidean,
)


def base_point_valid(chart: cxc.AbstractChart, point: "CsDict", /) -> bool:
    """Reject base points at coordinate singularities or outside valid ranges."""
    for key in ("r", "distance", "rho"):
        if key in chart.components:
            val = u.ustrip(point[key].unit, point[key])
            # Reject zero or negative radii (coordinate singularities)
            if jnp.less_equal(val, 0):
                return False

    # For spherical coordinates, reject poles (points on z-axis where azimuthal
    # angle is undefined) Spherical3D (physics convention): theta is polar
    # angle, singularity at theta ≈ 0 or π MathSpherical3D (math convention):
    # phi is polar angle, singularity at phi ≈ 0 or π

    # Detect which convention by checking component names: If has (r, theta,
    # phi) with r first: could be either, need to check chart type name
    chart_type = type(chart).__name__

    if "theta" in chart.components and "phi" in chart.components:
        # Both coordinates present - check which is the polar angle
        if "Math" in chart_type:
            # MathSpherical3D: phi is polar (range [0, π]), theta is azimuthal
            phi_val = u.ustrip("rad", point["phi"])
            # Reject if outside valid range
            if not (0 <= phi_val <= jnp.pi):
                return False
            # Reject poles
            if jnp.isclose(phi_val, 0, atol=1e-2) or jnp.isclose(
                phi_val, jnp.pi, atol=1e-2
            ):
                return False
        else:
            # Spherical3D (physics): theta is polar (range [0, π]), phi is azimuthal
            theta_val = u.ustrip("rad", point["theta"])
            # Reject if outside valid range
            if not (0 <= theta_val <= jnp.pi):
                return False
            # Reject poles
            if jnp.isclose(theta_val, 0, atol=1e-2) or jnp.isclose(
                theta_val, jnp.pi, atol=1e-2
            ):
                return False

    return True


@st.composite  # type: ignore[untyped-decorator]
def chart_pairs_same_dim(
    draw: st.DrawFn,
) -> tuple[cxc.AbstractChart, cxc.AbstractChart]:
    """Draw two charts with matching dimensionality to avoid unsupported conversions."""
    dim = draw(st.sampled_from([1, 2, 3]))
    chart_strategy = cxst.charts(
        dimensionality=dim,
        exclude=(cxc.EmbeddedManifold, cxc.TwoSphere),
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
@given(
    charts_pair=chart_pairs_same_dim(),
    data=st.data(),
)
def test_physical_tangent_roundtrip_velocity(charts_pair, data):
    """Round-trip property for velocity via physical_tangent_transform."""
    chart_A, chart_B = charts_pair
    # Base point and velocity in chart_B
    p_B = data.draw(cxst.pdicts(chart_B, cx.roles.point))
    if not base_point_valid(chart_B, p_B):
        reject()
    v_B = data.draw(cxst.pdicts(chart_B, cx.roles.phys_vel))

    # Forward transform to chart_A
    v_A = cxc.physical_tangent_transform(chart_A, chart_B, v_B, at=p_B)
    # Transform base point
    p_A = cx.transforms.point_transform(chart_A, chart_B, p_B)
    # Back transform to chart_B
    v_B2 = cxc.physical_tangent_transform(chart_B, chart_A, v_A, at=p_A)

    # Compare component values (with tolerance for floating point precision)
    # Note: Numerical errors on the order of 1e-6 observed in practice,
    # likely due to accumulated floating point operations in the Jacobian calculations
    for k in chart_B.components:
        assert jnp.allclose(
            u.ustrip(v_B[k].unit, v_B[k]),
            u.ustrip(v_B2[k].unit, v_B2[k]),
            rtol=1e-5,
            atol=2e-6,  # Tolerance allows for ~1ppm error
        )
