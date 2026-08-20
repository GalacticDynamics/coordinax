"""`metric_matrix`'s container is deliberate, and split by chart family.

Flat charts return a **bare array**; curvilinear and intrinsic charts return a
`QuantityMatrix`. That is a decision, not drift (#628), and these tests are
what stop it drifting.

Why not one container everywhere? Both are dimensionally correct -- a flat
`g = I` is dimensionless, so either would be honest -- and the tie is broken on
cost. The flat charts are the pure-JAX fast path, and boxing them measures:

    metric_matrix(cart3d) eager   196.6 us  ->  249.8 us   (+27%)
    tree_flatten(g)                 1.23 us ->    2.77 us  (2.3x)

which is exactly the per-call overhead a raw-array pipeline exists to avoid.

Why not bare everywhere? A curvilinear metric carries real units -- `sph3d` is
`m2/rad2`, and an *embedded* sphere's scales as `R**2` -- so stripping it would
discard information no consumer can recover.

A generic consumer therefore normalises with ``getattr(g, "value", g)``; the
alternative is paying the boxing cost on every flat call to save that.
"""

__all__: tuple[str, ...] = ()

import jax.numpy as jnp
import pytest

import unxt as u
import unxts.linalg as ul

import coordinax.charts as cxc
import coordinaxs.api.manifolds as cxmapi

_FLAT = [
    ("cart1d", cxc.cart1d, {"x": u.Q(1.0, "m")}),
    ("cart2d", cxc.cart2d, {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}),
    ("cart3d", cxc.cart3d, {k: u.Q(1.0, "m") for k in ("x", "y", "z")}),
    ("minkowskict", cxc.minkowskict, {k: u.Q(1.0, "m") for k in ("ct", "x", "y", "z")}),
    # `cartnd` has its own rule, keyed on the point's trailing axis rather than
    # on the chart's component count, so it can regress independently.
    ("cartnd", cxc.cartnd, {"q": u.Q(jnp.asarray([1.0, 2.0, 3.0]), "m")}),
]

_UNITED = [
    ("polar2d", cxc.polar2d, {"r": u.Q(2.0, "m"), "theta": u.Q(0.3, "rad")}),
    (
        "cyl3d",
        cxc.cyl3d,
        {"rho": u.Q(2.0, "m"), "phi": u.Q(0.3, "rad"), "z": u.Q(0.0, "m")},
    ),
    (
        "sph3d",
        cxc.sph3d,
        {"r": u.Q(2.0, "m"), "theta": u.Q(0.3, "rad"), "phi": u.Q(0.1, "rad")},
    ),
]

_INTRINSIC = [
    ("sph2", cxc.sph2, {"theta": u.Q(0.3, "rad"), "phi": u.Q(0.1, "rad")}),
    ("lonlat_sph2", cxc.lonlat_sph2, {"lon": u.Q(0.3, "rad"), "lat": u.Q(0.1, "rad")}),
]


def _diagonal(chart, point):
    return cxmapi.metric_matrix(chart.M, point, chart).diagonal


@pytest.mark.parametrize(("name", "chart", "point"), _FLAT, ids=[c[0] for c in _FLAT])
def test_flat_charts_return_a_bare_array(name, chart, point):
    """The fast path: no `Quantity` boxing on a metric that has no units."""
    del name
    assert not isinstance(_diagonal(chart, point), ul.QuantityMatrix)


@pytest.mark.parametrize(
    ("name", "chart", "point"), _UNITED, ids=[c[0] for c in _UNITED]
)
def test_curvilinear_charts_carry_their_units(name, chart, point):
    """`m2/rad2` is information a bare array could not carry."""
    del name
    d = _diagonal(chart, point)
    assert isinstance(d, ul.QuantityMatrix)
    assert any(d.unit[i] != u.unit("") for i in range(len(chart.components)))


@pytest.mark.parametrize(
    ("name", "chart", "point"), _INTRINSIC, ids=[c[0] for c in _INTRINSIC]
)
def test_intrinsic_charts_are_a_dimensionless_quantity_matrix(name, chart, point):
    """Dimensionless -- angles over angles -- but boxed like its siblings.

    The intrinsic sphere has no radius, so its metric is the *angular* one and
    carries no unit. It is still a `QuantityMatrix`, because it shares consumers
    with the curvilinear rules rather than with the flat fast path.
    """
    del name
    d = _diagonal(chart, point)
    assert isinstance(d, ul.QuantityMatrix)
    assert all(d.unit[i] == u.unit("") for i in range(len(chart.components)))


def test_the_embedded_sphere_scales_as_radius_squared():
    """Why the united branch cannot be flattened away: `R**2` is real content."""
    import coordinax.manifolds as cxm

    pt = {"theta": u.Q(0.3, "rad"), "phi": u.Q(0.1, "rad")}
    got = []
    for radius in (1.0, 2.0):
        M = cxm.EmbeddedManifold(
            intrinsic=cxm.S2, ambient=cxm.R3, embed_map=cxm.TwoSphereIn3D(radius=radius)
        )
        got.append(cxmapi.metric_matrix(M, pt, cxc.sph2).to_dense().matrix.value[0, 0])
    assert bool(jnp.isclose(got[1] / got[0], 4.0, atol=1e-6))
