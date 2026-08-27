"""Property test: analytic curvilinear metrics batch like ``vmap``.

Follow-up to #591, which fixed the analytic diagonal ``metric_matrix`` for the
orthogonal curvilinear Euclidean charts to be batch-safe. The regression tests
in ``test_metric_matrix_dispatch.py`` pin specific charts and values; this pins
the *general* invariant across the whole family with Hypothesis-drawn batches
of arbitrary shape:

    metric on a batched point  ==  metric evaluated element-by-element

i.e. leading axes are batch, components trail. A wrong axis ordering either
crashes on shape mismatch or produces per-row values that don't match the
unbatched evaluation, both of which this catches.
"""

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import array_shapes

import unxt as u

import coordinax.charts as cxc
import coordinax.manifolds as cxm
import coordinaxs.api.manifolds as cxmapi
import coordinaxs.hypothesis.main as cxst

# The analytic curvilinear diagonal metrics (manifold, chart), each reachable
# via the public ``metric_matrix``. This is exactly the family #591 touched.
CURVILINEAR = [
    (cxm.R2, cxc.polar2d),
    (cxm.R3, cxc.cyl3d),
    (cxm.R3, cxc.sph3d),
    (cxm.R3, cxc.math_sph3d),
    (cxm.R3, cxc.lonlat_sph3d),
    # The intrinsic hypersphere charts share the same cumulative-sine diagonal
    # and were missed by the #591 sweep: `jnp.cumprod` without an axis flattens,
    # so a batched point multiplied *across* the batch and returned shape
    # (1 + k*B,) instead of (*batch, k+1).
    (cxm.S1, cxc.sph1),
    (cxm.S2, cxc.sph2),
]


@given(data=st.data())
@settings(max_examples=50)
def test_curvilinear_metric_batches_like_elementwise(data):
    """Batched diagonal equals the per-element unbatched diagonals."""
    manifold, chart = data.draw(st.sampled_from(CURVILINEAR))
    shape = data.draw(array_shapes(min_dims=1, max_dims=2, min_side=1, max_side=3))
    # No `elements`: the chart's own domain now bounds each coordinate, and
    # the default magnitude keeps squared lengths clear of inf/nan noise.
    point = data.draw(cxst.cdicts(chart, shape=shape))

    n = len(chart.components)
    gb = cxmapi.metric_matrix(manifold, point, chart).diagonal
    assert gb.shape == (*shape, n)

    # Every chart parametrized here is curvilinear or intrinsic, so its
    # diagonal is a `QuantityMatrix`; the flat charts, which return a bare
    # array, are covered by `test_metric_container_convention`.
    gbv = np.asarray(gb.value)
    for idx in np.ndindex(shape):
        pt = {k: v[idx] for k, v in point.items()}
        gi = cxmapi.metric_matrix(manifold, pt, chart).diagonal
        # Same arithmetic on the same values: rows must match.
        np.testing.assert_allclose(gbv[idx], np.asarray(gi.value), rtol=1e-5)
        assert getattr(gb, "unit", None) == getattr(gi, "unit", None)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_sphere_diagonal_preserves_input_dtype(dtype):
    """The diagonal's dtype comes from the point, not the environment default.

    S¹ has no polar angles, so the diagonal is built from an *empty* stack. If
    that stack is created without an explicit dtype it picks up JAX's default
    and decides the result dtype, silently promoting (x64 on) or demoting
    (x64 off) the caller's angles.
    """
    at = {"phi": u.Q(np.asarray(0.7, dtype=dtype), "rad")}
    assert cxmapi.metric_matrix(cxm.S1, at, cxc.sph1).diagonal.dtype == dtype

    at2 = {
        "theta": u.Q(np.asarray(0.7, dtype=dtype), "rad"),
        "phi": u.Q(np.asarray(0.3, dtype=dtype), "rad"),
    }
    assert cxmapi.metric_matrix(cxm.S2, at2, cxc.sph2).diagonal.dtype == dtype


# ---------------------------------------------------------------------------
# The Jacobian-pullback family
#
# #591 pinned the analytic diagonal rules above. The other branch -- "all other
# charts", g = J^T J -- consumes `jac_pt_map`, which #782 changed from
# differentiating the batch as one function to mapping over its leading axes.
# Nothing pinned that branch batched, so this does.
# ---------------------------------------------------------------------------

_PULLBACK_CASES = [
    pytest.param(
        cxm.R3,
        cxc.loncoslat_sph3d,
        {
            "lon_coslat": u.Angle([0.4, 1.1, 0.2], "rad"),
            "lat": u.Angle([0.3, 0.5, 1.0], "rad"),
            "distance": u.Q([2.0, 3.0, 1.5], "m"),
        },
        id="loncoslat_sph3d",
    ),
    pytest.param(
        cxm.Sn(2),
        cxc.loncoslat_sph2,
        {
            "lon_coslat": u.Angle([0.4, 1.1, 0.2], "rad"),
            "lat": u.Angle([0.3, 0.5, 1.0], "rad"),
        },
        id="loncoslat_sph2",
    ),
    pytest.param(
        cxm.R3,
        cxc.ProlateSpheroidal3D(Delta=u.Q(1.0, "m")),
        {
            "mu": u.Q([2.0, 3.0, 1.5], "m2"),
            "nu": u.Q([0.5, 0.7, 0.2], "m2"),
            "phi": u.Angle([0.3, 1.0, 2.0], "rad"),
        },
        id="prolate_spheroidal",
    ),
]


def _dense(g):
    """Return the metric as a plain ``(..., n, n)`` array, whatever its type."""
    d = g.to_dense() if hasattr(g, "to_dense") else g
    m = d.matrix if hasattr(d, "matrix") else d
    return np.asarray(m.value if hasattr(m, "value") else m)


@pytest.mark.parametrize(("manifold", "chart", "at"), _PULLBACK_CASES)
def test_pullback_metric_batches_like_the_points(manifold, chart, at):
    """A batch of points gives a batch of metrics, element for element.

    Bit-exact: batched and unbatched run the same arithmetic per point, so a
    tolerance here would hide exactly the mispairing this guards against.
    """
    n = 3
    got = _dense(cxmapi.metric_matrix(manifold, at, chart))
    want = np.stack(
        [
            _dense(cxmapi.metric_matrix(manifold, {k: at[k][i] for k in at}, chart))
            for i in range(n)
        ]
    )
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=0, atol=0)
