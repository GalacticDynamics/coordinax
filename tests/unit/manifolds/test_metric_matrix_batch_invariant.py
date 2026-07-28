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
from hypothesis import given, settings
from hypothesis.extra.numpy import array_shapes

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

    gbv = np.asarray(gb.value)
    for idx in np.ndindex(shape):
        pt = {k: v[idx] for k, v in point.items()}
        gi = cxmapi.metric_matrix(manifold, pt, chart).diagonal
        # Same arithmetic on the same values: rows must match, units unchanged.
        np.testing.assert_allclose(gbv[idx], np.asarray(gi.value), rtol=1e-5)
        assert gb.unit == gi.unit
