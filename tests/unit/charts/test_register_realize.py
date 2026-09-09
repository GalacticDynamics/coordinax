"""Tests for coordinate realization functions (register_realize.py).

cartesian_chart, pt_map.
"""

import math

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import plum
import pytest
from hypothesis import assume, given

import unxt as u

import coordinax as cx
import coordinax.charts as cxc
import coordinax.manifolds as cxm
import coordinaxs.hypothesis.main as cxst
from coordinax._src.charts.register_ptmap import _ratio_zero_on_axis

# =============================================================================
# cartesian_chart
# =============================================================================


class TestCartesianChartFunction:
    """Tests for cartesian_chart function."""

    @pytest.mark.parametrize(
        ("chart", "expected_cartesian"),
        [
            (cxc.cart0d, cxc.cart0d),
            (cxc.cart1d, cxc.cart1d),
            (cxc.radial1d, cxc.cart1d),
            (cxc.cart2d, cxc.cart2d),
            (cxc.polar2d, cxc.cart2d),
            (cxc.cart3d, cxc.cart3d),
            (cxc.sph3d, cxc.cart3d),
            (cxc.lonlat_sph3d, cxc.cart3d),
            (cxc.loncoslat_sph3d, cxc.cart3d),
            (cxc.cyl3d, cxc.cart3d),
            (cxc.cartnd, cxc.cartnd),
        ],
    )
    def test_cartesian_chart_examples(self, chart, expected_cartesian):
        """Test that cartesian_chart returns the expected cartesian chart."""
        assert cxc.cartesian_chart(chart) == expected_cartesian

    @given(chart=cxst.charts())
    def test_cartesian_chart_idempotent(self, chart):
        """Property test: cartesian_chart is idempotent."""
        try:
            cart1 = cxc.cartesian_chart(chart)
            cart2 = cxc.cartesian_chart(cart1)
        except (cxc.NoGlobalCartesianChartError, plum.NotFoundLookupError):
            pass
        else:
            assert cart1 == cart2


# =============================================================================
# cartesian_chart for product charts
# =============================================================================


class TestCartesianChartProductCharts:
    """Test cartesian_chart dispatch for product charts."""

    def test_namespaced_product_cartesian_chart(self) -> None:
        """cartesian_chart should convert factors while preserving factor_names."""
        phase_sph = cxc.CartesianProductChart((cxc.sph3d, cxc.sph3d), ("q", "p"))
        phase_cart = cxc.cartesian_chart(phase_sph)
        assert isinstance(phase_cart.factors[0], cxc.Cart3D)
        assert isinstance(phase_cart.factors[1], cxc.Cart3D)
        assert phase_cart.factor_names == ("q", "p")

    def test_cartesian_chart_idempotent(self) -> None:
        """cartesian_chart applied twice should return same object."""
        phase_sph = cxc.CartesianProductChart((cxc.sph3d, cxc.sph3d), ("q", "p"))
        cart1 = cxc.cartesian_chart(phase_sph)
        cart2 = cxc.cartesian_chart(cart1)
        assert cart1 is cart2


# =============================================================================
# pt_map with product charts
# =============================================================================


class TestPointTransformProductCharts:
    """Test ``pt_map`` works correctly with product charts."""

    def test_namespaced_phase_space_transform(self) -> None:
        """pt_map should work with namespaced CartesianProductChart."""
        phase_cart = cxc.CartesianProductChart((cxc.cart3d, cxc.cart3d), ("q", "p"))
        phase_sph = cxc.CartesianProductChart((cxc.sph3d, cxc.sph3d), ("q", "p"))
        p = {
            "q.x": u.Q(1, "m"),
            "q.y": u.Q(0, "m"),
            "q.z": u.Q(0, "m"),
            "p.x": u.Q(0, "m"),
            "p.y": u.Q(1, "m"),
            "p.z": u.Q(0, "m"),
        }
        result = cxc.pt_map(p, phase_cart, phase_sph)
        assert u.ustrip("m", result["q.r"]) == pytest.approx(1)
        assert u.ustrip("rad", result["q.phi"]) == pytest.approx(0)
        assert u.ustrip("m", result["p.r"]) == pytest.approx(1)
        assert u.ustrip("rad", result["p.phi"]) == pytest.approx(jnp.pi / 2)


class TestPointTransformProlate:
    """``pt_map`` into ProlateSpheroidal3D routes via Cylindrical3D (no recursion)."""

    def test_cart3d_to_prolate_roundtrips(self) -> None:
        prolate = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))
        p = {"x": u.Q(0.5, "m"), "y": u.Q(1.5, "m"), "z": u.Q(3.0, "m")}
        out = cxc.pt_map(p, cxc.cart3d, prolate)
        assert set(out) == {"mu", "nu", "phi"}
        back = cxc.pt_map(out, prolate, cxc.cart3d)
        for k in ("x", "y", "z"):
            assert u.ustrip("m", back[k]) == pytest.approx(u.ustrip("m", p[k]))

    def test_spherical_to_prolate(self) -> None:
        # Routes via the generic fallback: Spherical3D -> Cart3D -> Cyl -> Prolate.
        prolate = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))
        p = {"r": u.Q(3.0, "m"), "theta": u.Q(0.6, "rad"), "phi": u.Q(0.4, "rad")}
        out = cxc.pt_map(p, cxc.sph3d, prolate)
        assert set(out) == {"mu", "nu", "phi"}

    @pytest.mark.parametrize(
        "to_chart", [cxc.cart3d, cxc.cyl3d], ids=["cart3d", "cyl3d"]
    )
    def test_bare_mu_nu_without_a_unit_system_is_refused(self, to_chart) -> None:
        """`Delta` is a length; a bare `mu`/`nu` gives nothing to measure it against.

        Rejecting beats guessing: silently reading them as `Delta`'s own unit
        would make the answer depend on how the chart happened to be built.
        """
        prolate = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))
        p = {"mu": 12.0, "nu": 0.5, "phi": 0.3}
        with pytest.raises(ValueError, match="usys must be a UnitSystem"):
            cxc.pt_map(p, prolate, to_chart)

        # With one, it goes through.
        out = cxc.pt_map(p, prolate, to_chart, usys=u.unitsystems.si)
        assert all(v is not None for v in out.values())


class TestPointTransformCartND:
    """``pt_map`` from CartND reads components on the last axis (batch-safe)."""

    def test_cartnd_to_cart3d_batched(self) -> None:
        # A batch of 2 points in 3D: the dimensionality guard must read the
        # component axis (last), not the batch axis (leading), and not raise.
        p = {"q": u.Q(jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), "m")}
        out = cxc.pt_map(p, cxc.cartnd, cxc.cart3d)
        assert set(out) == {"x", "y", "z"}
        assert u.ustrip("m", out["x"]) == pytest.approx([1.0, 4.0])
        assert u.ustrip("m", out["y"]) == pytest.approx([2.0, 5.0])
        assert u.ustrip("m", out["z"]) == pytest.approx([3.0, 6.0])


# Concrete non-canonical two-sphere charts, drawn by type + realization so the
# test covers any future ``AbstractSphericalTwoSphere`` subclass automatically.
_two_sphere_charts = cxst.charts(
    filter=cxc.AbstractSphericalTwoSphere, exclude=(cxc.SphericalTwoSphere,)
)


class TestPointTransformTwoSphereCrossChart:
    """Non-canonical two-sphere charts convert to each other via SphericalTwoSphere."""

    @given(data=st.data())
    def test_cross_chart_matches_route_via_canonical(self, data):
        a = data.draw(_two_sphere_charts)
        b = data.draw(_two_sphere_charts)
        assume(type(a) is not type(b))  # matching types are the identity route
        p = data.draw(cxst.cdicts(a))

        out = cxc.pt_map(p, a, b)
        ref = cxc.pt_map(cxc.pt_map(p, a, cxc.sph2), cxc.sph2, b)

        assert set(out) == set(ref)
        # Compare in a shared canonical unit (rad); equal_nan lets pole
        # singularities (cos(lat) -> 0) match on both routes.
        for k in out:
            assert bool(
                jnp.allclose(
                    u.ustrip("rad", out[k]), u.ustrip("rad", ref[k]), equal_nan=True
                )
            )


# =============================================================================
# Coordinate-singularity division is grad-safe (double-where idiom)
# =============================================================================


class TestAxisSingularityGradSafe:
    """``pt_map`` divisions at a coordinate singularity stay finite in value and grad.

    A plain ``jnp.where(denom == 0, 0, num / denom)`` returns the right value but
    leaks ``NaN`` into reverse-mode gradients; these guard against regressing to it.
    """

    def test_ratio_helper_value_and_grad(self):
        def f(d):
            return _ratio_zero_on_axis(2.0 * d, d)

        d0 = jnp.asarray(0.0)
        assert float(f(d0)) == 0.0
        assert bool(jnp.isfinite(jax.grad(f)(d0)))

    def test_loncoslat_pole_is_finite(self):
        """LonCosLat -> Cart3D at the pole (cos(lat) == 0) is finite."""
        p = {
            "lon_coslat": u.Q(0.4, "rad"),
            "lat": u.Q(90.0, "deg"),
            "distance": u.Q(2.0, "m"),
        }
        out = cxc.pt_map(p, cxm.R3, cxc.loncoslat_sph3d, cxm.R3, cxc.cart3d)
        assert all(bool(jnp.isfinite(v.value)) for v in out.values())

    def test_poincarepolar6d_axis_roundtrip_is_finite(self):
        """Forward + inverse through the rho == 0 axis stays finite (dt_rho, lz/rho)."""
        ps = cxc.CartesianProductChart((cxc.cart3d, cxc.cart3d), ("q", "p"))
        q = {
            "q.x": u.Q(0.0, "kpc"),
            "q.y": u.Q(0.0, "kpc"),
            "q.z": u.Q(5.0, "kpc"),
            "p.x": u.Q(0.0, "kpc/Myr"),
            "p.y": u.Q(0.0, "kpc/Myr"),
            "p.z": u.Q(0.5, "kpc/Myr"),
        }
        pp = cxc.pt_map(q, ps.M, ps, cxc.poincarepolar6d.M, cxc.poincarepolar6d)
        back = cxc.pt_map(pp, cxc.poincarepolar6d.M, cxc.poincarepolar6d, ps.M, ps)
        assert all(bool(jnp.isfinite(v.value)) for v in pp.values())
        assert all(bool(jnp.isfinite(v.value)) for v in back.values())


# =============================================================================


class TestSphericalConditioning:
    """The transition into ``Spherical3D`` is conditioned at the range edges and poles.

    It used to compute ``r = sqrt(x**2 + y**2 + z**2)`` and ``theta = acos(z / r)``.
    Squaring costs half the exponent range, and ``acos`` saturates as ``z / r -> 1``;
    ``hypot`` chains and ``atan2`` have neither problem. The conventions the old
    ``where(r == 0, ...)`` guard produced are unchanged, and ``atan2``'s sensitivity
    to the sign of its first argument -- which ``acos`` did not have -- is guarded.
    """

    @pytest.mark.parametrize("mag", [3e20, 1e-25])
    def test_r_survives_the_float32_range_edges(self, mag):
        """``mag ** 2`` is out of float32 range either way; ``mag`` itself is not.

        3e20 m is about 10 kpc, and squaring it overflowed to ``inf``; 1e-25 m
        underflowed to 0. ``abs=0`` because `approx`'s default absolute
        tolerance is far larger than the small magnitude being checked.
        """
        f32 = jnp.float32
        p = {k: u.Q(f32(v), "m") for k, v in (("x", mag), ("y", 0.0), ("z", 0.0))}
        r = cxc.pt_map(p, cxc.cart3d, cxc.sph3d)["r"]
        assert float(r.ustrip("m")) == pytest.approx(mag, rel=1e-5, abs=0)

    @pytest.mark.parametrize(
        ("from_chart", "build"),
        [
            (
                cxc.cart3d,
                lambda rho: {
                    "x": u.Q(rho, "m"),
                    "y": u.Q(0.0, "m"),
                    "z": u.Q(1.0, "m"),
                },
            ),
            (
                cxc.cyl3d,
                lambda rho: {
                    "rho": u.Q(rho, "m"),
                    "phi": u.Angle(0.0, "rad"),
                    "z": u.Q(1.0, "m"),
                },
            ),
        ],
        ids=["cart3d", "cyl3d"],
    )
    def test_theta_resolves_near_the_pole(self, from_chart, build):
        """``acos(z / r)`` returned exactly 0 here; both sources share the fix."""
        rho = 1e-10
        theta = cxc.pt_map(build(rho), from_chart, cxc.sph3d)["theta"]
        assert float(theta.ustrip("rad")) == pytest.approx(math.atan2(rho, 1.0))

    def test_theta_is_sign_invariant_in_rho(self):
        """A negative ``rho`` gives its positive twin's ``theta``, and stays in domain.

        ``acos(z / hypot(rho, z))`` squared the sign away; ``atan2`` would not.
        ``Cylindrical3D`` does not value-validate ``rho``, so a hand-built negative
        one is reachable, and a negative ``theta`` is outside the ``[0, pi]`` that
        ``Spherical3D.check_data`` enforces.
        """

        def to_sph(rho):
            p = {"rho": u.Q(rho, "m"), "phi": u.Angle(0.0, "rad"), "z": u.Q(4.0, "m")}
            return cxc.pt_map(p, cxc.cyl3d, cxc.sph3d)

        out = to_sph(-3.0)
        assert float(out["theta"].ustrip("rad")) == pytest.approx(
            float(to_sph(3.0)["theta"].ustrip("rad"))
        )
        cxc.sph3d.check_data(out, values=True)  # in domain

    def test_origin_keeps_the_zero_theta_convention(self):
        """What the removed ``where(r == 0, ...)`` guard used to supply."""
        p = {k: u.Q(0.0, "m") for k in ("x", "y", "z")}
        out = cxc.pt_map(p, cxc.cart3d, cxc.sph3d)
        assert float(out["r"].ustrip("m")) == 0.0
        assert float(out["theta"].ustrip("rad")) == 0.0


# =============================================================================


class TestPtMapUnitContract:
    """Which unit a transition's output carries, and which components keep their own.

    The bodies used to do their arithmetic on `Quantity` operands, so these rules
    were a *consequence* of `unxt` promotion. `strip`/`wrap` reproduce them
    deliberately, so they are pinned here. Every case passes against the
    pre-`strip` bodies too -- that is what makes them a contract.

    Not repeated here: unitless-in/unitless-out and angular canonicalisation,
    which the `pt_map` doctests and `test_container_canonicalisation` already
    cover for every chart pair.
    """

    @pytest.mark.parametrize(("x_unit", "y_unit"), [("km", "m"), ("m", "km")])
    def test_result_takes_the_first_components_unit(self, x_unit, y_unit):
        """Mixed units resolve to the unit of the chart's *first* component, `x`.

        Not the largest, not the smallest, and not the input dict's ordering --
        which is what `Quantity` arithmetic happened to produce.
        """
        p = {"x": u.Q(1.0, x_unit), "y": u.Q(2.0, y_unit), "z": u.Q(3.0, "cm")}
        assert cxc.pt_map(p, cxc.cart3d, cxc.sph3d)["r"].unit == u.unit(x_unit)

    @pytest.mark.parametrize(
        ("frm", "to", "rest"),
        [
            (cxc.cart3d, cxc.cyl3d, {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}),
            (cxc.cyl3d, cxc.cart3d, {"rho": u.Q(1.0, "m"), "phi": u.Angle(0.5, "rad")}),
        ],
        ids=["cart3d->cyl3d", "cyl3d->cart3d"],
    )
    def test_untouched_length_keeps_its_unit_and_container(self, frm, to, rest):
        """`z` is not consumed by either body's arithmetic, so it is not converted.

        Round-tripping it through the group unit would turn `Q(3, "km")` into
        `Q(3000, "m")` and degrade a `Distance` to a `Quantity`.
        """
        z = cxc.pt_map({**rest, "z": u.Q(3.0, "km")}, frm, to)["z"]
        assert z.unit == u.unit("km")
        assert z.value == pytest.approx(3.0)

        z = cxc.pt_map({**rest, "z": cx.Distance(3.0, "m")}, frm, to)["z"]
        assert isinstance(z, cx.Distance)

    @pytest.mark.parametrize(
        ("frm", "to", "rest"),
        [
            (cxc.cyl3d, cxc.sph3d, {"rho": u.Q(3.0, "m"), "z": u.Q(4.0, "m")}),
            (cxc.sph3d, cxc.cyl3d, {"r": u.Q(5.0, "m"), "theta": u.Angle(0.9, "rad")}),
        ],
        ids=["cyl3d->sph3d", "sph3d->cyl3d"],
    )
    def test_untouched_angle_passes_through(self, frm, to, rest):
        """`phi` sits outside each body's length group, and keeps degrees."""
        phi = cxc.pt_map({**rest, "phi": u.Angle(30.0, "deg")}, frm, to)["phi"]
        assert phi.unit == u.unit("deg")
        assert phi.value == pytest.approx(30.0)

    def test_the_output_container_follows_the_length_not_the_angle(self):
        """A unitless radius stays unitless however the angle is wrapped.

        `Quantity` arithmetic used to promote whenever *any* operand was one,
        so `sph3d -> cart3d` returned a *dimensionless* `Quantity` for a length
        when the angle happened to be an `Angle` and a bare array when it did
        not -- and, in the same point, left an untouched `z` bare either way.
        The output pytree structure therefore depended on how the angles were
        wrapped, which is the route-dependence `canonical_containers` exists to
        remove. Only the operands carrying length information decide.
        """
        ang = {"theta": u.Angle(0.7, "rad"), "phi": u.Angle(1.2, "rad")}
        wrapped = cxc.pt_map({"r": 2.0, **ang}, cxc.sph3d, cxc.cart3d)
        bare = cxc.pt_map({"r": 2.0, "theta": 0.7, "phi": 1.2}, cxc.sph3d, cxc.cart3d)

        assert all(u.unit_of(v) is None for v in wrapped.values())
        assert jax.tree.structure(wrapped) == jax.tree.structure(bare)

    def test_a_length_component_still_decides_its_own_output(self):
        """The rule is about *which* operands decide, not about dropping units."""
        p = {"r": u.Q(2.0, "km"), "theta": 0.7, "phi": 1.2}
        out = cxc.pt_map(p, cxc.sph3d, cxc.cart3d)
        assert all(u.unit_of(v) == u.unit("km") for v in out.values())


# =============================================================================


class TestBareAnglesHonourTheUnitSystem:
    """A bare angle means whatever `usys` says, in every chart that reads one.

    `sph2 -> lonlat_sph2` and its inverse discarded `usys` and read a bare
    colatitude as radians, so a `usys` of degrees put `lat` ~39 off. The same
    point routed through `loncoslat_sph2`, or through the 3-D `lonlat_sph3d`,
    gave the right answer -- three charts, one point, two answers for the same
    component. A round trip hid it, because both directions were wrong the same
    way and `pi / 2 - (pi / 2 - x)` is `x` whatever the units.
    """

    @pytest.mark.parametrize(
        ("frm", "to", "point"),
        [
            (cxc.sph2, cxc.lonlat_sph2, {"theta": 40.0, "phi": 70.0}),
            (cxc.sph2, cxc.loncoslat_sph2, {"theta": 40.0, "phi": 70.0}),
            (cxc.sph3d, cxc.lonlat_sph3d, {"r": 1.0, "theta": 40.0, "phi": 70.0}),
        ],
        ids=["sph2->lonlat", "sph2->loncoslat", "sph3d->lonlat"],
    )
    def test_every_chart_agrees_on_lat(self, frm, to, point):
        """A 40 degree colatitude is a 50 degree latitude, however it is reached.

        Expressed in the unit system's own angle unit, since that is how the
        bare input was read -- see `TestBareAngleRoundTrips`.
        """
        usys = u.unitsystem("m", "deg")
        lat = cxc.pt_map(point, frm, to, usys=usys)["lat"]
        assert float(lat) == pytest.approx(50.0)

    @pytest.mark.parametrize(
        ("frm", "to", "point"),
        [
            (cxc.lonlat_sph2, cxc.sph2, {"lon": 70.0, "lat": 50.0}),
            (cxc.loncoslat_sph2, cxc.sph2, {"lon_coslat": 70.0, "lat": 50.0}),
            (cxc.lonlat_sph3d, cxc.sph3d, {"lon": 70.0, "lat": 50.0, "distance": 1.0}),
        ],
        ids=["lonlat->sph2", "loncoslat->sph2", "lonlat->sph3d"],
    )
    def test_every_chart_agrees_on_theta(self, frm, to, point):
        """And back: the inverse map discarded `usys` in the same way.

        `lonlat_sph2 -> sph2` gave `theta = -48.43` for a 50 degree latitude
        where its two siblings gave `+0.69813`. Both directions were changed,
        so both are pinned.
        """
        usys = u.unitsystem("m", "deg")
        theta = cxc.pt_map(point, frm, to, usys=usys)["theta"]
        assert float(theta) == pytest.approx(40.0)


# =============================================================================


class TestPoincarePolarUnits:
    """`PoincarePolar6D` mixes two dimensional groups, so its units are derived.

    `lz = x*vy - y*vx` is a length times a velocity, so `pp_phi = sqrt(2|lz|)`
    carries the square root of that product. Its unit therefore has to be
    *computed* from the group units rather than passed through, which is what
    distinguishes this chart from the rest of the rollout.
    """

    _PS = cxc.CartesianProductChart((cxc.cart3d, cxc.cart3d), ("q", "p"))

    def _phase(self, **over):
        p = {
            "q.x": u.Q(3.0, "kpc"),
            "q.y": u.Q(4.0, "kpc"),
            "q.z": u.Q(5.0, "kpc"),
            "p.x": u.Q(1.0, "kpc/Myr"),
            "p.y": u.Q(2.0, "kpc/Myr"),
            "p.z": u.Q(0.5, "kpc/Myr"),
        }
        return {**p, **over}

    def test_pp_components_carry_sqrt_of_length_times_velocity(self):
        out = cxc.pt_map(self._phase(), self._PS, cxc.poincarepolar6d)
        assert out["rho"].unit == u.unit("kpc")
        assert out["dt_rho"].unit == u.unit("kpc/Myr")
        for k in ("pp_phi", "pp_phidot"):
            assert (
                out[k].unit
                == u.unit("kpc") * u.unit("kpc/Myr") ** 0.5 / u.unit("kpc") ** 0.5
            )

    def test_untouched_components_keep_their_own_units(self):
        """`z` and `dt_z` are never read by the arithmetic."""
        out = cxc.pt_map(
            self._phase(**{"q.z": u.Q(5000.0, "pc"), "p.z": u.Q(0.489, "km/s")}),
            self._PS,
            cxc.poincarepolar6d,
        )
        assert out["z"].unit == u.unit("pc")
        assert out["dt_z"].unit == u.unit("km/s")
        assert out["rho"].unit == u.unit("kpc")

    def test_the_derived_unit_does_not_depend_on_operand_order(self):
        """A velocity given in other units names the same quantity, not another.

        `lz` used to take its unit from `x * vy` alone, so one component in
        `km/s` relabelled `pp_phi` as `km(1/2) kpc(1/2) / s(1/2)`.
        """
        # The *same* velocity, written in km/s rather than kpc/Myr.
        same_vy = u.uconvert(u.unit("km/s"), u.Q(2.0, "kpc/Myr"))
        mixed = cxc.pt_map(
            self._phase(**{"p.y": same_vy}), self._PS, cxc.poincarepolar6d
        )
        plain = cxc.pt_map(self._phase(), self._PS, cxc.poincarepolar6d)
        assert mixed["pp_phi"].unit == plain["pp_phi"].unit
        assert float(u.ustrip(plain["pp_phi"].unit, mixed["pp_phi"])) == pytest.approx(
            float(plain["pp_phi"].value), rel=1e-6
        )

    def test_round_trip_is_exact(self):
        p = self._phase()
        back = cxc.pt_map(
            cxc.pt_map(p, self._PS, cxc.poincarepolar6d), cxc.poincarepolar6d, self._PS
        )
        for k, v in p.items():
            assert u.ustrip(v.unit, back[k]) == pytest.approx(float(v.value), rel=1e-9)


# ===========================================================================


class TestBareAngleRoundTrips:
    """A bare angle is read *and written* in ``usys["angle"]``.

    `rad_value` reads a bare angle in the unit system's angle unit, but the
    inverse-trigonometric functions that produce one return radians. Writing
    those out raw made the two disagree, and for charts whose own components
    are angles the disagreement compounded: `sph3d -> lonlat_sph3d` emitted a
    latitude in radians and the inverse read that number as degrees, so a 40
    degree colatitude came back as 89.13 degrees.

    Round trips are the sharpest statement of the property, since they fail
    only when reading and writing disagree.
    """

    _USYS = u.unitsystem("m", "deg")

    @pytest.mark.parametrize(
        ("mid", "point"),
        [
            (cxc.lonlat_sph3d, {"r": 1.0, "theta": 40.0, "phi": 70.0}),
            (cxc.loncoslat_sph3d, {"r": 1.0, "theta": 40.0, "phi": 70.0}),
            (cxc.cart3d, {"r": 1.0, "theta": 40.0, "phi": 70.0}),
            (cxc.cyl3d, {"r": 1.0, "theta": 40.0, "phi": 70.0}),
        ],
        ids=["lonlat", "loncoslat", "cart3d", "cyl3d"],
    )
    def test_sph3d_round_trips_through(self, mid, point):
        """Out and back returns the angle it started with, in the same unit."""
        kw = {"usys": self._USYS}
        back = cxc.pt_map(cxc.pt_map(point, cxc.sph3d, mid, **kw), mid, cxc.sph3d, **kw)
        assert float(back["theta"]) == pytest.approx(point["theta"], rel=1e-6)

    @pytest.mark.parametrize(
        "mid", [cxc.lonlat_sph2, cxc.loncoslat_sph2], ids=["lonlat", "loncoslat"]
    )
    def test_sph2_round_trips_through(self, mid):
        kw = {"usys": self._USYS}
        p = {"theta": 40.0, "phi": 70.0}
        back = cxc.pt_map(cxc.pt_map(p, cxc.sph2, mid, **kw), mid, cxc.sph2, **kw)
        assert float(back["theta"]) == pytest.approx(40.0, rel=1e-6)

    def test_a_bare_angle_comes_out_in_the_systems_unit(self):
        """The direct statement: 40 degrees of colatitude is 50 of latitude."""
        out = cxc.pt_map(
            {"r": 1.0, "theta": 40.0, "phi": 70.0},
            cxc.sph3d,
            cxc.lonlat_sph3d,
            usys=self._USYS,
        )
        assert float(out["lat"]) == pytest.approx(50.0)

    @pytest.mark.parametrize(
        ("usys", "expected"),
        [
            (None, math.pi / 4),
            (u.unitsystems.si, math.pi / 4),
            (u.unitsystem("m", "rad"), math.pi / 4),
            (u.unitsystem("m", "deg"), 45.0),
        ],
        ids=["no-usys", "si", "rad", "deg"],
    )
    def test_a_bare_angle_leaves_in_the_unit_it_would_be_read_in(self, usys, expected):
        """No unit system, or one whose angle is radians, means radians.

        The degrees case is the one this PR changes; the other three are the
        controls that say it did not disturb them.
        """
        kw = {} if usys is None else {"usys": usys}
        out = cxc.pt_map({"x": 1.0, "y": 1.0, "z": 0.0}, cxc.cart3d, cxc.sph3d, **kw)
        assert float(out["phi"]) == pytest.approx(expected, rel=1e-6)
