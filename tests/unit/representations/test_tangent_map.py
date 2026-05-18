"""Tests for tangent_map function.

Tests that tangent_map correctly transforms tangent vectors between charts using
the Jacobian pushforward (CoordinateBasis) or frame matrix rotation (PhysicalBasis).
"""

__all__: tuple[str, ...] = ()

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

import unxt as u

import coordinax.charts as cxc
import coordinax.main as cx
import coordinax.representations as cxr

usys = u.unitsystems.si


class TestTangentMapExistence:
    """tangent_map is importable and callable."""

    def test_importable_from_representations(self) -> None:
        """tangent_map is in coordinax.representations."""
        assert hasattr(cxr, "tangent_map")
        assert callable(cxr.tangent_map)

    def test_importable_from_main(self) -> None:
        """tangent_map is in coordinax.main."""
        assert hasattr(cx, "tangent_map")


class TestTangentMapSameChart:
    """Same-chart optimisation: tangent_map returns v unchanged."""

    def test_cart3d_to_cart3d(self) -> None:
        """Cart3D → Cart3D with CoordinateBasis returns input unchanged."""
        v = {"x": jnp.array(1.0), "y": jnp.array(2.0), "z": jnp.array(3.0)}
        at = {"x": jnp.array(0.5), "y": jnp.array(0.5), "z": jnp.array(0.0)}
        result = cxr.tangent_map(
            v, cxc.cart3d, cxr.coord_disp, cxc.cart3d, at=at, usys=usys
        )
        np.testing.assert_allclose(result["x"], v["x"])
        np.testing.assert_allclose(result["y"], v["y"])
        np.testing.assert_allclose(result["z"], v["z"])

    def test_same_chart_identity(self) -> None:
        """Any same chart passes through unchanged."""
        v = {"r": jnp.array(1.0), "theta": jnp.array(0.0), "phi": jnp.array(0.0)}
        at = {"r": jnp.array(2.0), "theta": jnp.array(0.5), "phi": jnp.array(0.5)}
        result = cxr.tangent_map(v, cxc.sph3d, cxr.coord_disp, cxc.sph3d, at=at)
        np.testing.assert_allclose(result["r"], v["r"])
        np.testing.assert_allclose(result["theta"], v["theta"])
        np.testing.assert_allclose(result["phi"], v["phi"])


class TestTangentMapCart3dToSph3d:
    """Cart3D → Sph3D CoordinateBasis: Jacobian pushforward at (x=1, y=0, z=0).

    Uses physics spherical conventions: theta=polar, phi=azimuthal.
    At (x=1, y=0, z=0), the base point is (r=1, theta=pi/2, phi=0).

    The Jacobian J = d(r,theta,phi)/d(x,y,z) at this point is::

        J = [[1,  0,  0],
             [0,  0, -1],
             [0,  1,  0]]

    Resulting pushforwards:
      - (dx,dy,dz)=(1,0,0) → (dr,dtheta,dphi)=(1,0,0)  [radial]
      - (dx,dy,dz)=(0,1,0) → (dr,dtheta,dphi)=(0,0,1)  [phi direction]
      - (dx,dy,dz)=(0,0,1) → (dr,dtheta,dphi)=(0,-1,0) [minus theta direction]
    """

    def test_radial_vector(self) -> None:
        """Purely x-direction at (1,0,0) maps to purely radial (dr=1)."""
        v = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
        at = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
        result = cxr.tangent_map(
            v, cxc.cart3d, cxr.coord_disp, cxc.sph3d, at=at, usys=usys
        )
        np.testing.assert_allclose(result["r"], 1.0, atol=1e-6)
        np.testing.assert_allclose(result["theta"], 0.0, atol=1e-6)
        np.testing.assert_allclose(result["phi"], 0.0, atol=1e-6)

    def test_phi_direction_vector(self) -> None:
        """y-direction at (1,0,0) maps to phi direction (dphi=1)."""
        v = {"x": jnp.array(0.0), "y": jnp.array(1.0), "z": jnp.array(0.0)}
        at = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
        result = cxr.tangent_map(
            v, cxc.cart3d, cxr.coord_disp, cxc.sph3d, at=at, usys=usys
        )
        np.testing.assert_allclose(result["r"], 0.0, atol=1e-6)
        np.testing.assert_allclose(result["theta"], 0.0, atol=1e-6)
        np.testing.assert_allclose(result["phi"], 1.0, atol=1e-6)

    def test_z_direction_vector(self) -> None:
        """z-direction at (1,0,0) maps to -theta direction (dtheta=-1)."""
        v = {"x": jnp.array(0.0), "y": jnp.array(0.0), "z": jnp.array(1.0)}
        at = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
        result = cxr.tangent_map(
            v, cxc.cart3d, cxr.coord_disp, cxc.sph3d, at=at, usys=usys
        )
        np.testing.assert_allclose(result["r"], 0.0, atol=1e-6)
        np.testing.assert_allclose(result["theta"], -1.0, atol=1e-6)
        np.testing.assert_allclose(result["phi"], 0.0, atol=1e-6)


class TestTangentMapPhysicalBasis:
    """PhysicalBasis transformations are supported for tangent vectors."""

    def test_cart3d_to_sph3d_radial_direction(self) -> None:
        """Cartesian x-direction maps to spherical radial in physical basis."""
        v = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
        at = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}

        result = cxr.tangent_map(
            v, cxc.cart3d, cxr.phys_basis, cxc.sph3d, at=at, usys=usys
        )

        np.testing.assert_allclose(result["r"], 1.0, atol=1e-6)
        np.testing.assert_allclose(result["theta"], 0.0, atol=1e-6)
        np.testing.assert_allclose(result["phi"], 0.0, atol=1e-6)

    def test_representation_dispatch_with_phys_disp(self) -> None:
        """Representation overload supports physical-basis representations."""
        v = {"x": jnp.array(0.0), "y": jnp.array(0.0), "z": jnp.array(1.0)}
        at = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}

        result = cxr.tangent_map(
            v, cxc.cart3d, cxr.phys_disp, cxc.sph3d, at=at, usys=usys
        )

        np.testing.assert_allclose(result["r"], 0.0, atol=1e-6)
        np.testing.assert_allclose(result["theta"], -1.0, atol=1e-6)
        np.testing.assert_allclose(result["phi"], 0.0, atol=1e-6)

    def test_same_chart_identity_phys_basis(self) -> None:
        """Same-chart optimisation also holds for PhysicalBasis inputs."""
        v = {"r": jnp.array(2.0), "theta": jnp.array(-1.5), "phi": jnp.array(0.25)}
        at = {"r": jnp.array(3.0), "theta": jnp.array(0.5), "phi": jnp.array(0.1)}

        result = cxr.tangent_map(v, cxc.sph3d, cxr.phys_basis, cxc.sph3d, at=at)

        np.testing.assert_allclose(result["r"], v["r"])
        np.testing.assert_allclose(result["theta"], v["theta"])
        np.testing.assert_allclose(result["phi"], v["phi"])

    def test_roundtrip_cart3d_sph3d_phys_disp(self) -> None:
        """Physical-basis round-trip Cart3D→Sph3D→Cart3D preserves components."""
        v_cart = {
            "x": u.Q(jnp.array(2.0), "m/s"),
            "y": u.Q(jnp.array(-1.0), "m/s"),
            "z": u.Q(jnp.array(0.5), "m/s"),
        }
        at_cart = {
            "x": u.Q(jnp.array(2.0), "m"),
            "y": u.Q(jnp.array(1.0), "m"),
            "z": u.Q(jnp.array(0.5), "m"),
        }

        v_sph = cxr.tangent_map(
            v_cart, cxc.cart3d, cxr.phys_disp, cxc.sph3d, at=at_cart, usys=usys
        )
        at_sph = cxc.pt_map(at_cart, cxc.cart3d, cxc.sph3d, usys=usys)
        v_back = cxr.tangent_map(
            v_sph, cxc.sph3d, cxr.phys_disp, cxc.cart3d, at=at_sph, usys=usys
        )

        np.testing.assert_allclose(v_back["x"].value, v_cart["x"].value, atol=1e-6)
        np.testing.assert_allclose(v_back["y"].value, v_cart["y"].value, atol=1e-6)
        np.testing.assert_allclose(v_back["z"].value, v_cart["z"].value, atol=1e-6)

    def test_cconvert_7arg_with_phys_disp(self) -> None:
        """7-arg cconvert path works for physical-basis tangent representations."""
        v = {
            "x": u.Q(jnp.array(1.0), "m/s"),
            "y": u.Q(jnp.array(0.0), "m/s"),
            "z": u.Q(jnp.array(0.0), "m/s"),
        }
        at = {
            "x": u.Q(jnp.array(1.0), "m"),
            "y": u.Q(jnp.array(0.0), "m"),
            "z": u.Q(jnp.array(0.0), "m"),
        }

        direct = cxr.tangent_map(v, cxc.cart3d, cxr.phys_disp, cxc.sph3d, at=at)
        via_cc = cxr.cconvert(
            v,
            cxc.cart3d,
            cxr.tangent_geom,
            cxr.phys_disp,
            cxc.sph3d,
            cxr.tangent_geom,
            cxr.phys_disp,
            at=at,
        )

        via_cc_cdict = cast("dict[str, Any]", via_cc)
        np.testing.assert_allclose(
            via_cc_cdict["r"].value, direct["r"].value, atol=1e-6
        )
        np.testing.assert_allclose(
            via_cc_cdict["theta"].value, direct["theta"].value, atol=1e-6
        )
        np.testing.assert_allclose(
            via_cc_cdict["phi"].value, direct["phi"].value, atol=1e-6
        )


class TestTangentMapCart2dToPolar2d:
    """Cart2D → Polar2D CoordinateBasis: Jacobian pushforward at (x=1, y=0).

    At (x=1, y=0), the base point is (r=1, theta=0).

    The Jacobian J = d(r,theta)/d(x,y) at this point is::

        J = [[1, 0],
             [0, 1]]

    Resulting pushforwards:
      - (dx,dy)=(1,0) → (dr,dtheta)=(1,0)  [radial]
      - (dx,dy)=(0,1) → (dr,dtheta)=(0,1)  [angular]
    """

    def test_x_direction(self) -> None:
        """x-direction at (1,0) maps to radial (dr=1)."""
        v = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        at = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        result = cxr.tangent_map(v, cxc.cart2d, cxr.coord_disp, cxc.polar2d, at=at)
        np.testing.assert_allclose(result["r"], 1.0, atol=1e-6)
        np.testing.assert_allclose(result["theta"], 0.0, atol=1e-6)

    def test_y_direction(self) -> None:
        """y-direction at (1,0) maps to angular (dtheta=1)."""
        v = {"x": jnp.array(0.0), "y": jnp.array(1.0)}
        at = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        result = cxr.tangent_map(v, cxc.cart2d, cxr.coord_disp, cxc.polar2d, at=at)
        np.testing.assert_allclose(result["r"], 0.0, atol=1e-6)
        np.testing.assert_allclose(result["theta"], 1.0, atol=1e-6)

    def test_at_45_deg(self) -> None:
        """At (x=1,y=1): x-hat component check (dr/dx = x/r = 1/sqrt2)."""
        v = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        at = {"x": jnp.array(1.0), "y": jnp.array(1.0)}
        result = cxr.tangent_map(v, cxc.cart2d, cxr.coord_disp, cxc.polar2d, at=at)
        np.testing.assert_allclose(result["r"], 1.0 / jnp.sqrt(2.0), atol=1e-6)
        np.testing.assert_allclose(result["theta"], -1.0 / 2.0, atol=1e-6)


class TestTangentMapJAXCompatibility:
    """tangent_map is compatible with jax.jit and jax.vmap."""

    def test_jit(self) -> None:
        """tangent_map can be JIT-compiled."""
        v = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        at = {"x": jnp.array(1.0), "y": jnp.array(0.0)}

        @jax.jit
        def jitted_tangent_map(v, at):
            return cxr.tangent_map(v, cxc.cart2d, cxr.coord_disp, cxc.polar2d, at=at)

        result = jitted_tangent_map(v, at)
        np.testing.assert_allclose(result["r"], 1.0, atol=1e-6)

    def test_vmap(self) -> None:
        """tangent_map can be vmap-ped over a batch of base points."""
        vs = {"x": jnp.ones(3), "y": jnp.zeros(3)}
        ats = {"x": jnp.array([1, 2, 3]), "y": jnp.zeros(3)}

        def single_map(v: dict[str, Any], at: dict[str, Any]) -> dict[str, Any]:
            return cxr.tangent_map(v, cxc.cart2d, cxr.coord_disp, cxc.polar2d, at=at)

        batched = jax.vmap(single_map)(vs, ats)
        # At y=0, any x>0: dr/dx = x/r = 1, so dr = 1 always
        np.testing.assert_allclose(batched["r"], 1, atol=1e-6)


class TestTangentMapSemanticPreservation:
    """tangent_map works with vel and acc representations too."""

    def test_vel_rep(self) -> None:
        """tangent_map works with coord_vel representation."""
        v = {"x": jnp.array(1), "y": jnp.array(0)}
        at = {"x": jnp.array(1), "y": jnp.array(0)}
        result = cxr.tangent_map(v, cxc.cart2d, cxr.coord_vel, cxc.polar2d, at=at)
        np.testing.assert_allclose(result["r"], 1, atol=1e-6)

    def test_acc_rep(self) -> None:
        """tangent_map works with coord_acc representation."""
        v = {"x": jnp.array(1), "y": jnp.array(0)}
        at = {"x": jnp.array(1), "y": jnp.array(0)}
        result = cxr.tangent_map(v, cxc.cart2d, cxr.coord_acc, cxc.polar2d, at=at)
        np.testing.assert_allclose(result["r"], 1, atol=1e-6)
