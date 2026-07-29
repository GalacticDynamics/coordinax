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

# Bounded so squared lengths can't overflow into inf/nan noise; sign is
# irrelevant (the metric depends on r**2 and sin/cos), so a plain range is fine.
_coords = st.floats(
    min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False, width=32
)


@given(data=st.data())
@settings(deadline=None, max_examples=50)
def test_curvilinear_metric_batches_like_elementwise(data):
    """Batched diagonal equals the per-element unbatched diagonals."""
    manifold, chart = data.draw(st.sampled_from(CURVILINEAR))
    shape = data.draw(array_shapes(min_dims=1, max_dims=2, min_side=1, max_side=3))
    point = data.draw(cxst.cdicts(chart, shape=shape, elements=_coords))

    n = len(chart.components)
    gb = cxmapi.metric_matrix(manifold, point, chart).diagonal
    assert gb.shape == (*shape, n)

    # The Euclidean rules return a united QuantityMatrix; the intrinsic sphere
    # rules return a bare (dimensionless) Array. Compare whichever is carried.
    gbv = np.asarray(getattr(gb, "value", gb))
    for idx in np.ndindex(shape):
        pt = {k: v[idx] for k, v in point.items()}
        gi = cxmapi.metric_matrix(manifold, pt, chart).diagonal
        # Same arithmetic on the same values: rows must match, units unchanged.
        np.testing.assert_allclose(
            gbv[idx], np.asarray(getattr(gi, "value", gi)), rtol=1e-5
        )
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
