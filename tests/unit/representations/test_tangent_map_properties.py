"""Tests for ``tangent_map``."""

__all__: tuple[str, ...] = ()

import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings, strategies as st
from strategies import (
    any_angle_rad as _any_angle_rad,
    any_m as _any_m,
    polar_rad as _polar_rad,
    pos_m as _pos_m,
    v_elem as _v_elem,
)

import unxt as u

import coordinax.charts as cxc
import coordinax.representations as cxr

usys = u.unitsystems.si


def _assert_cdict_close(got, ref, *, atol=1e-5, rtol=1e-5):
    """Assert two CDicts agree component-wise (strips units if present)."""
    for key in ref:
        g = got[key]
        r = ref[key]
        # Strip Quantity wrappers to plain arrays for comparison
        g_val = g.value if hasattr(g, "value") else jnp.asarray(g)
        r_val = r.value if hasattr(r, "value") else jnp.asarray(r)
        np.testing.assert_allclose(
            np.asarray(g_val),
            np.asarray(r_val),
            atol=atol,
            rtol=rtol,
            err_msg=f"component '{key}' differs",
        )


# ===========================================================================
# 1. Round-trip: Cart2D ↔ Polar2D
# ===========================================================================


class TestTangentMapRoundTripCart2dPolar2d:
    r"""Round-trip invariant: Cart2D → Polar2D → Cart2D ≈ identity.

    For any non-zero base point p and tangent vector v:

        J_{polar→cart}(p_polar) @ J_{cart→polar}(p_cart) @ v ≈ v

    Here we check this end-to-end via two successive ``tangent_map`` calls.
    """

    @given(
        r_=_pos_m,
        theta=_any_angle_rad,
        vx=_v_elem,
        vy=_v_elem,
    )
    @settings(deadline=None)
    def test_round_trip(self, r_, theta, vx, vy) -> None:
        """Tangent map round-trip Cart2D → Polar2D → Cart2D recovers original v."""
        # Base point starting in polar (guaranteed r > 0)
        p_polar = {"r": r_, "theta": theta}
        p_cart = cxc.pt_map(p_polar, cxc.polar2d, cxc.cart2d)

        v_cart = {"x": jnp.array(vx), "y": jnp.array(vy)}

        v_polar = cxr.tangent_map(
            v_cart, cxc.cart2d, cxr.coord_disp, cxc.polar2d, at=p_cart
        )
        v_back = cxr.tangent_map(
            v_polar, cxc.polar2d, cxr.coord_disp, cxc.cart2d, at=p_polar
        )

        np.testing.assert_allclose(
            np.asarray(v_back["x"]),
            vx,
            atol=1e-4,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            np.asarray(v_back["y"]),
            vy,
            atol=1e-4,
            rtol=1e-4,
        )


# ===========================================================================
# 2. Round-trip: Cart3D ↔ Sph3D
# ===========================================================================


class TestTangentMapRoundTripCart3dSph3d:
    r"""Round-trip invariant: Cart3D → Sph3D → Cart3D ≈ identity."""

    @given(
        r_=_pos_m,
        theta=_polar_rad,  # away from poles
        phi=_any_angle_rad,
        vx=_v_elem,
        vy=_v_elem,
        vz=_v_elem,
    )
    @settings(deadline=None)
    def test_round_trip(self, r_, theta, phi, vx, vy, vz) -> None:
        """Tangent map round-trip Cart3D → Sph3D → Cart3D recovers v."""
        p_sph = {"r": r_, "theta": theta, "phi": phi}
        p_cart = cxc.pt_map(p_sph, cxc.sph3d, cxc.cart3d)

        v_cart = {"x": jnp.array(vx), "y": jnp.array(vy), "z": jnp.array(vz)}

        v_sph = cxr.tangent_map(
            v_cart, cxc.cart3d, cxr.coord_disp, cxc.sph3d, at=p_cart
        )
        v_back = cxr.tangent_map(v_sph, cxc.sph3d, cxr.coord_disp, cxc.cart3d, at=p_sph)

        np.testing.assert_allclose(np.asarray(v_back["x"]), vx, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(np.asarray(v_back["y"]), vy, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(np.asarray(v_back["z"]), vz, atol=1e-4, rtol=1e-4)


# ===========================================================================
# 3. Round-trip: Cart3D ↔ Cyl3D
# ===========================================================================


class TestTangentMapRoundTripCart3dCyl3d:
    r"""Round-trip invariant: Cart3D → Cyl3D → Cart3D ≈ identity."""

    @given(
        rho=_pos_m,  # cylindrical radius — must be positive
        phi=_any_angle_rad,
        z=_any_m,
        vx=_v_elem,
        vy=_v_elem,
        vz=_v_elem,
    )
    @settings(deadline=None)
    def test_round_trip(self, rho, phi, z, vx, vy, vz) -> None:
        """Tangent map round-trip Cart3D → Cyl3D → Cart3D recovers v."""
        p_cyl = {"rho": rho, "phi": phi, "z": z}
        p_cart = cxc.pt_map(p_cyl, cxc.cyl3d, cxc.cart3d)

        v_cart = {"x": jnp.array(vx), "y": jnp.array(vy), "z": jnp.array(vz)}

        v_cyl = cxr.tangent_map(
            v_cart, cxc.cart3d, cxr.coord_disp, cxc.cyl3d, at=p_cart
        )
        v_back = cxr.tangent_map(v_cyl, cxc.cyl3d, cxr.coord_disp, cxc.cart3d, at=p_cyl)

        np.testing.assert_allclose(np.asarray(v_back["x"]), vx, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(np.asarray(v_back["y"]), vy, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(np.asarray(v_back["z"]), vz, atol=1e-4, rtol=1e-4)


# ===========================================================================
# 4. Known Cyl3D examples (hand-verified)
# ===========================================================================


class TestTangentMapCyl3dKnownExamples:
    r"""Specific known-value tests for Cyl3D that aren't in the existing suite.

    Analytical pushforward review:

    At (x=1, y=0, z=0) → (ρ=1, φ=0, z=0), J_{cart→cyl} = I, so:
        v_cart → v_cyl component-wise identical.

    At (x=0, y=1, z=0) → (ρ=1, φ=π/2, z=0):
        J = [[0,  1,  0],      rows: (ρ, φ, z)
             [-1, 0,  0],      cols: (x, y, z)
             [0,  0,  1]]
        v_cart = (1, 0, 0) → v_cyl = (0, -1, 0)   (azimuthal component flips sign)
        v_cart = (0, 1, 0) → v_cyl = (1,  0, 0)   (becomes radial)
        v_cart = (0, 0, 1) → v_cyl = (0,  0, 1)   (z is unchanged)
    """

    def test_at_x1_y0_z0_identity(self) -> None:
        """At (1,0,0) the Jacobian is identity: every component is preserved."""
        at = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
        for vx, vy, vz in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]:
            v = {"x": jnp.array(vx), "y": jnp.array(vy), "z": jnp.array(vz)}
            result = cxr.tangent_map(
                v, cxc.cart3d, cxr.coord_disp, cxc.cyl3d, at=at, usys=usys
            )
            np.testing.assert_allclose(result["rho"], vx, atol=1e-6)
            np.testing.assert_allclose(result["phi"], vy, atol=1e-6)
            np.testing.assert_allclose(result["z"], vz, atol=1e-6)

    def test_x_hat_at_0_1_0_maps_to_minus_phi_hat(self) -> None:
        """At (0,1,0): x̂ → (ρ=0, φ=-1, z=0) — negative azimuthal direction."""
        at = {"x": jnp.array(0.0), "y": jnp.array(1.0), "z": jnp.array(0.0)}
        v = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
        result = cxr.tangent_map(
            v, cxc.cart3d, cxr.coord_disp, cxc.cyl3d, at=at, usys=usys
        )
        np.testing.assert_allclose(result["rho"], 0.0, atol=1e-6)
        np.testing.assert_allclose(result["phi"], -1.0, atol=1e-6)
        np.testing.assert_allclose(result["z"], 0.0, atol=1e-6)

    def test_y_hat_at_0_1_0_maps_to_rho_hat(self) -> None:
        """At (0,1,0): ŷ → (ρ=1, φ=0, z=0) — becomes radial."""
        at = {"x": jnp.array(0.0), "y": jnp.array(1.0), "z": jnp.array(0.0)}
        v = {"x": jnp.array(0.0), "y": jnp.array(1.0), "z": jnp.array(0.0)}
        result = cxr.tangent_map(
            v, cxc.cart3d, cxr.coord_disp, cxc.cyl3d, at=at, usys=usys
        )
        np.testing.assert_allclose(result["rho"], 1.0, atol=1e-6)
        np.testing.assert_allclose(result["phi"], 0.0, atol=1e-6)
        np.testing.assert_allclose(result["z"], 0.0, atol=1e-6)

    def test_cyl_to_cart_rho_hat_at_phi0(self) -> None:
        """At (ρ=1, φ=0, z=0): ρ̂ → (x=1, y=0, z=0) — becomes x̂."""
        at = {"rho": jnp.array(1.0), "phi": jnp.array(0.0), "z": jnp.array(0.0)}
        v = {"rho": jnp.array(1.0), "phi": jnp.array(0.0), "z": jnp.array(0.0)}
        result = cxr.tangent_map(
            v, cxc.cyl3d, cxr.coord_disp, cxc.cart3d, at=at, usys=usys
        )
        np.testing.assert_allclose(result["x"], 1.0, atol=1e-6)
        np.testing.assert_allclose(result["y"], 0.0, atol=1e-6)
        np.testing.assert_allclose(result["z"], 0.0, atol=1e-6)

    def test_cyl_phi_hat_at_phi0_maps_to_y_hat(self) -> None:
        """At (R=1, φ=0, z=0): φ̂ in Cartesian is ŷ.

        ∂x/∂φ = -R sinφ = 0,  ∂y/∂φ = R cosφ = 1,  ∂z/∂φ = 0.
        So φ̂ (i.e. v_phi=1, others=0) → (x=0, y=1, z=0).
        """
        at = {"rho": jnp.array(1.0), "phi": jnp.array(0.0), "z": jnp.array(0.0)}
        v = {"rho": jnp.array(0.0), "phi": jnp.array(1.0), "z": jnp.array(0.0)}
        result = cxr.tangent_map(
            v, cxc.cyl3d, cxr.coord_disp, cxc.cart3d, at=at, usys=usys
        )
        np.testing.assert_allclose(result["x"], 0.0, atol=1e-6)
        np.testing.assert_allclose(result["y"], 1.0, atol=1e-6)
        np.testing.assert_allclose(result["z"], 0.0, atol=1e-6)


# ===========================================================================
# 5. Linearity property
# ===========================================================================


class TestTangentMapLinearity:
    r"""Linearity.

    tangent_map(a·u + b·w, ...) ≈ a·tangent_map(u,...) + b·tangent_map(w,...).

    The Jacobian is a linear map, so this is a sanity-check that the
    implementation doesn't introduce nonlinear artefacts.
    """

    @given(
        r_=_pos_m,
        theta=_polar_rad,
        phi=_any_angle_rad,
        ux=_v_elem,
        uy=_v_elem,
        uz=_v_elem,
        wx=_v_elem,
        wy=_v_elem,
        wz=_v_elem,
        a=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, width=32),
        b=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, width=32),
    )
    @settings(deadline=None)
    def test_linearity_cart3d_to_sph3d(
        self, r_, theta, phi, ux, uy, uz, wx, wy, wz, a, b
    ) -> None:
        """J(a·u + b·w) ≈ a·J(u) + b·J(w) for Cart3D → Sph3D."""
        p_sph = {"r": r_, "theta": theta, "phi": phi}
        p_cart = cxc.pt_map(p_sph, cxc.sph3d, cxc.cart3d)

        u_ = {"x": jnp.array(ux), "y": jnp.array(uy), "z": jnp.array(uz)}
        w_ = {"x": jnp.array(wx), "y": jnp.array(wy), "z": jnp.array(wz)}
        comb = {
            "x": jnp.array(a * ux + b * wx),
            "y": jnp.array(a * uy + b * wy),
            "z": jnp.array(a * uz + b * wz),
        }

        J_u = cxr.tangent_map(u_, cxc.cart3d, cxr.coord_disp, cxc.sph3d, at=p_cart)
        J_w = cxr.tangent_map(w_, cxc.cart3d, cxr.coord_disp, cxc.sph3d, at=p_cart)
        J_comb = cxr.tangent_map(comb, cxc.cart3d, cxr.coord_disp, cxc.sph3d, at=p_cart)

        for key in ("r", "theta", "phi"):
            expected = float(a * J_u[key] + b * J_w[key])
            np.testing.assert_allclose(
                float(J_comb[key]),
                expected,
                atol=1e-4,
                rtol=1e-4,
                err_msg=f"linearity failed for component '{key}'",
            )


# ===========================================================================
# 6. Unit-tracking: at with Quantity values
# ===========================================================================


class TestTangentMapWithQuantityAt:
    r"""When ``at`` contains Quantity values, coordinate units are tracked correctly.

    The Jacobian entries J[j, i] carry units  to_dim[j] / from_dim[i].
    When applied to a tangent vector v whose components carry units (e.g. m/s),
    the result components carry the correct physical units.

    For Cart3D → Sph3D:
        v["x"] in m/s → result["r"]    in m/s   (dimensionless x m/s)
        v["x"] in m/s → result["theta"] in rad/s (rad/m x m/s)
        v["x"] in m/s → result["phi"]  in rad/s  (rad/m x m/s)
    """

    def test_result_r_unit_matches_input_unit(self) -> None:
        """J[r, *] is dimensionless, so result['r'] has same unit as v['x']."""
        at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        v = {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")}
        result = cxr.tangent_map(v, cxc.cart3d, cxr.coord_vel, cxc.sph3d, at=at)
        # At (1, 0, 0): only r has non-zero result: r̂ component = 1 m/s
        r_result = result["r"]
        assert hasattr(r_result, "unit"), "result['r'] should be a Quantity"
        assert u.dimension_of(r_result) == u.dimension("speed"), (
            f"result['r'] should have speed dimensions, got {u.dimension_of(r_result)}"
        )

    def test_result_theta_unit_is_angular_velocity(self) -> None:
        """J[θ, *] is rad/m, so result['theta'] has rad * (m/s) / m = rad/s."""
        at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        # Use ŷ input which has non-zero dφ component at (1,0,0)
        # At (1,0,0): dθ/dy = 0, dφ/dy = 1 rad/m → result phi = 1 rad/s for vy=1 m/s
        v = {"x": u.Q(0.0, "m/s"), "y": u.Q(1.0, "m/s"), "z": u.Q(0.0, "m/s")}
        result = cxr.tangent_map(v, cxc.cart3d, cxr.coord_vel, cxc.sph3d, at=at)
        phi_result = result["phi"]
        assert hasattr(phi_result, "unit"), "result['phi'] should be a Quantity"
        assert u.dimension_of(phi_result) == u.dimension("angular frequency"), (
            "result['phi'] should have angular-velocity dimensions"
        )


# ===========================================================================
# 7. Integration via cconvert
# ===========================================================================


class TestTangentMapViaCconvert:
    """``cconvert`` should route tangent conversions through ``tangent_map``.

    ``cconvert`` internally calls
        api.tangent_map(v, from_chart, from_geom, from_rep,
                        to_chart, to_geom, to_rep, at=at)

    These tests cover that integration path end to end for
    ``TangentGeometry`` conversions.
    """

    def test_cart2d_polar2d_via_cconvert(self) -> None:
        """Cart2D tangent components convert to Polar2D via ``cconvert``."""
        v = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        at = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        result = cxr.cconvert(
            v,
            cxc.cart2d,
            cxr.tangent_geom,
            cxr.coord_disp,
            cxc.polar2d,
            cxr.tangent_geom,
            cxr.coord_disp,
            at=at,
        )
        np.testing.assert_allclose(result["r"], 1.0, atol=1e-6)
        np.testing.assert_allclose(result["theta"], 0.0, atol=1e-6)

    def test_cart3d_sph3d_via_cconvert(self) -> None:
        """``cconvert`` matches a direct ``tangent_map`` call for Cart3D -> Sph3D."""
        v = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
        at = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}

        direct = cxr.tangent_map(
            v, cxc.cart3d, cxr.coord_disp, cxc.sph3d, at=at, usys=usys
        )
        via_cc = cxr.cconvert(
            v,
            cxc.cart3d,
            cxr.tangent_geom,
            cxr.coord_disp,
            cxc.sph3d,
            cxr.tangent_geom,
            cxr.coord_disp,
            at=at,
            usys=usys,
        )

        for key in ("r", "theta", "phi"):
            np.testing.assert_allclose(
                float(via_cc[key]), float(direct[key]), atol=1e-6
            )

    def test_cart3d_cyl3d_via_cconvert(self) -> None:
        """Cart3D -> Cyl3D matches the expected tangent components at ``(0, 1, 0)``."""
        v = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
        at = {"x": jnp.array(0.0), "y": jnp.array(1.0), "z": jnp.array(0.0)}
        result = cxr.cconvert(
            v,
            cxc.cart3d,
            cxr.tangent_geom,
            cxr.coord_disp,
            cxc.cyl3d,
            cxr.tangent_geom,
            cxr.coord_disp,
            at=at,
            usys=usys,
        )
        np.testing.assert_allclose(result["rho"], 0.0, atol=1e-6)
        np.testing.assert_allclose(result["phi"], -1.0, atol=1e-6)
        np.testing.assert_allclose(result["z"], 0.0, atol=1e-6)


# ===========================================================================
# 8. Semantic preservation (vel / acc representations)
# ===========================================================================


class TestTangentMapSemanticPreservationCyl3d:
    """Semantic kind (vel, acc) is preserved through Cyl3D transformations.

    The tangent_map result keys should match to_chart.components regardless
    of semantic kind.  This extends the existing vel/acc tests to the Cyl3D pair.
    """

    def test_coord_vel_cart3d_to_cyl3d(self) -> None:
        """coord_vel converts Cart3D → Cyl3D correctly."""
        v = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
        at = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
        result = cxr.tangent_map(
            v, cxc.cart3d, cxr.coord_vel, cxc.cyl3d, at=at, usys=usys
        )
        assert set(result.keys()) == {"rho", "phi", "z"}
        np.testing.assert_allclose(result["rho"], 1.0, atol=1e-6)

    def test_coord_acc_cart3d_to_cyl3d(self) -> None:
        """coord_acc converts Cart3D → Cyl3D correctly."""
        v = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
        at = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
        result = cxr.tangent_map(
            v, cxc.cart3d, cxr.coord_acc, cxc.cyl3d, at=at, usys=usys
        )
        assert set(result.keys()) == {"rho", "phi", "z"}
