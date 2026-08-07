"""Tests for `guess_geometry_kind`, `guess_semantic_kind` and `guess_rep`.

The three functions read the *same* inputs -- a dimension, a tuple of
dimensions, a Quantity, or a CDict -- and answer three questions about them:
which geometry, which semantic kind, and the `Representation` combining both.
So there is one table of inputs with the two expected answers per row, and
`guess_rep` is checked to agree with the other two rather than being given its
own copy of the table.

That is what the file used to be: 91 functions, one per (input, function) pair,
with the input literals written out three times.
"""

__all__: tuple[str, ...] = ()

import pytest
from hypothesis import given

import unxt as u

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinaxs.hypothesis.representations as cxrst

# ===================================================================
# The table
#
# Each row is (id, input, expected geometry, expected semantic kind).

CASES = [
    # --- dimensions ---
    ("dim-length", u.dimension("length"), cxr.point_geom, cxr.loc),
    ("dim-angle", u.dimension("angle"), cxr.point_geom, cxr.loc),
    ("dim-speed", u.dimension("speed"), cxr.tangent_geom, cxr.vel),
    ("dim-angular-speed", u.dimension("angular speed"), cxr.tangent_geom, cxr.vel),
    ("dim-acceleration", u.dimension("acceleration"), cxr.tangent_geom, cxr.acc),
    (
        "dim-angular-acceleration",
        u.dimension("angular acceleration"),
        cxr.tangent_geom,
        cxr.acc,
    ),
    # --- tuples of dimensions, in both orders ---
    (
        "dims-speed-angular-speed",
        (u.dimension("speed"), u.dimension("angular speed")),
        cxr.tangent_geom,
        cxr.vel,
    ),
    (
        "dims-angular-speed-speed",
        (u.dimension("angular speed"), u.dimension("speed")),
        cxr.tangent_geom,
        cxr.vel,
    ),
    (
        "dims-acceleration-angular-acceleration",
        (u.dimension("acceleration"), u.dimension("angular acceleration")),
        cxr.tangent_geom,
        cxr.acc,
    ),
    (
        "dims-angular-acceleration-acceleration",
        (u.dimension("angular acceleration"), u.dimension("acceleration")),
        cxr.tangent_geom,
        cxr.acc,
    ),
    # --- quantities ---
    ("qty-length", u.Q(1, "m"), cxr.point_geom, cxr.loc),
    ("qty-angle", u.Q(0.5, "rad"), cxr.point_geom, cxr.loc),
    ("qty-speed", u.Q(1, "m / s"), cxr.tangent_geom, cxr.vel),
    ("qty-angular-speed", u.Q(1, "rad / s"), cxr.tangent_geom, cxr.vel),
    ("qty-acceleration", u.Q(1, "m / s**2"), cxr.tangent_geom, cxr.acc),
    ("qty-angular-acceleration", u.Q(1, "rad / s**2"), cxr.tangent_geom, cxr.acc),
    # --- cdicts ---
    (
        "cdict-cartesian",
        {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")},
        cxr.point_geom,
        cxr.loc,
    ),
    (
        "cdict-spherical-mixed-dims",
        {"r": u.Q(1, "m"), "theta": u.Q(0.5, "rad"), "phi": u.Q(1, "rad")},
        cxr.point_geom,
        cxr.loc,
    ),
    (
        "cdict-pure-angle",
        {"lon": u.Q(1, "deg"), "lat": u.Q(2, "deg")},
        cxr.point_geom,
        cxr.loc,
    ),
    (
        "cdict-speed",
        {"vx": u.Q(1, "m / s"), "vy": u.Q(2, "m / s")},
        cxr.tangent_geom,
        cxr.vel,
    ),
    (
        "cdict-angular-speed",
        {"vphi": u.Q(1, "rad / s"), "vtheta": u.Q(0.5, "rad / s")},
        cxr.tangent_geom,
        cxr.vel,
    ),
    (
        "cdict-mixed-speed-angular-speed",
        {"vr": u.Q(1, "m / s"), "vphi": u.Q(0.5, "rad / s")},
        cxr.tangent_geom,
        cxr.vel,
    ),
    (
        "cdict-acceleration",
        {"ax": u.Q(1, "m / s**2"), "ay": u.Q(2, "m / s**2")},
        cxr.tangent_geom,
        cxr.acc,
    ),
    (
        "cdict-angular-acceleration",
        {"aphi": u.Q(1, "rad / s**2"), "atheta": u.Q(0.5, "rad / s**2")},
        cxr.tangent_geom,
        cxr.acc,
    ),
    (
        "cdict-mixed-acceleration-angular-acceleration",
        {"ar": u.Q(1, "m / s**2"), "aphi": u.Q(0.5, "rad / s**2")},
        cxr.tangent_geom,
        cxr.acc,
    ),
]

CASE_PARAMS = [pytest.param(inp, geom, sem, id=name) for name, inp, geom, sem in CASES]

#: Inputs no `guess_*` function can classify.
UNGUESSABLE = [
    pytest.param(u.dimension("time"), id="dim-time"),
    pytest.param(u.Q(1, "s"), id="qty-time"),
    pytest.param({}, id="cdict-empty"),
    pytest.param({"t": u.Q(1, "s")}, id="cdict-time"),
]


# ===================================================================
# The three functions, over the one table


@pytest.mark.parametrize(("value", "geom", "sem"), CASE_PARAMS)
def test_guess_geometry_kind(value, geom, sem) -> None:
    assert cxr.guess_geometry_kind(value) == geom


@pytest.mark.parametrize(("value", "geom", "sem"), CASE_PARAMS)
def test_guess_semantic_kind(value, geom, sem) -> None:
    assert cxr.guess_semantic_kind(value) == sem


@pytest.mark.parametrize(("value", "geom", "sem"), CASE_PARAMS)
def test_guess_rep_agrees_with_both(value, geom, sem) -> None:
    """`guess_rep` combines exactly what the other two return."""
    rep = cxr.guess_rep(value)
    assert isinstance(rep, cxr.Representation)
    assert rep.geom_kind == geom
    assert rep.semantic_kind == sem


@pytest.mark.parametrize(
    "guess",
    [cxr.guess_geometry_kind, cxr.guess_semantic_kind, cxr.guess_rep],
    ids=["geometry_kind", "semantic_kind", "rep"],
)
@pytest.mark.parametrize("value", UNGUESSABLE)
def test_unguessable_input_raises(guess, value) -> None:
    with pytest.raises(ValueError, match="Cannot infer"):
        guess(value)


# ===================================================================
# Identity dispatch
#
# Each function returns its own kind of argument unchanged.


class TestIdentityDispatch:
    """Passing an already-resolved value through returns that same object."""

    @given(geom=cxrst.geometries())
    def test_geometry_kind_is_idempotent(self, geom: cxr.AbstractGeometry) -> None:
        assert cxr.guess_geometry_kind(geom) is geom

    @given(sem=cxrst.semantics())
    def test_semantic_kind_is_idempotent(self, sem: cxr.AbstractSemanticKind) -> None:
        assert cxr.guess_semantic_kind(sem) is sem

    @given(rep=cxrst.representations())
    def test_rep_is_idempotent(self, rep: cxr.Representation) -> None:
        assert cxr.guess_rep(rep) is rep

    @pytest.mark.parametrize(
        ("guess", "canonical"),
        [
            (cxr.guess_geometry_kind, cxr.point_geom),
            (cxr.guess_semantic_kind, cxr.loc),
            (cxr.guess_rep, cxr.point),
        ],
        ids=["geometry_kind", "semantic_kind", "rep"],
    )
    def test_canonical_instance_is_returned_by_identity(self, guess, canonical) -> None:
        """The canonical singletons come back as the same object, not a copy."""
        assert guess(canonical) is canonical

    def test_rep_accepts_a_geometry(self) -> None:
        """`guess_rep(point_geom)` resolves the remaining two fields."""
        assert cxr.guess_rep(cxr.point_geom) == cxr.point


# ===================================================================
# CDict + chart dispatch
#
# Only the geometry and rep functions take a chart alongside the data.


class TestWithChart:
    """The `(CDict, chart)` overloads."""

    @pytest.mark.parametrize(
        ("guess", "expected"),
        [(cxr.guess_geometry_kind, cxr.point_geom), (cxr.guess_rep, cxr.point)],
        ids=["geometry_kind", "rep"],
    )
    def test_matching_keys(self, guess, expected) -> None:
        """Each function returns its own whole answer, not just the geometry.

        `guess_rep` is compared against the canonical `point` rather than
        having its geometry picked out, so a wrong basis or semantic kind
        cannot slip through.
        """
        d = {"x": u.Q(1, "m"), "y": u.Q(2, "m")}
        assert guess(d, cxc.cart2d) == expected

    def test_wrong_keys_raise(self) -> None:
        d = {"a": u.Q(1, "m"), "b": u.Q(2, "m")}
        with pytest.raises(ValueError, match="Data keys do not match chart components"):
            cxr.guess_geometry_kind(d, cxc.cart2d)

    def test_prolate_spheroidal_area_and_angle(self) -> None:
        """A chart whose components are {area, area, angle} is still a point."""
        d = {"mu": u.Q(1, "km2"), "nu": u.Q(0.5, "km2"), "phi": u.Q(1, "deg")}
        chart = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(1, "km"))
        assert cxr.guess_geometry_kind(d, chart) == cxr.point_geom


# ===================================================================
# Return types


class TestReturnTypes:
    """Each function's return is an instance of the type it names."""

    @given(geom=cxrst.geometries())
    def test_geometry_kind(self, geom: cxr.AbstractGeometry) -> None:
        assert isinstance(cxr.guess_geometry_kind(geom), cxr.AbstractGeometry)

    @given(sem=cxrst.semantics())
    def test_semantic_kind(self, sem: cxr.AbstractSemanticKind) -> None:
        assert isinstance(cxr.guess_semantic_kind(sem), cxr.AbstractSemanticKind)

    @given(rep=cxrst.representations())
    def test_rep(self, rep: cxr.Representation) -> None:
        assert isinstance(cxr.guess_rep(rep), cxr.Representation)
