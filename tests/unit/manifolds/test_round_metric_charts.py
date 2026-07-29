"""The round metric on S^n is expressed in the *passed* chart's coordinates.

Every case pins to a textbook closed form derived independently of the
implementation.  For the unit two-sphere,
``ds2 = dtheta^2 + sin^2(theta) dphi^2``, so:

* ``sph2``      -> diag(1, sin^2(theta))
* ``lonlat``    -> diag(cos^2(lat), 1)                      [lat = pi/2 - theta]
* ``math``      -> diag(sin^2(phi), 1)                      [angle names swapped]
* ``loncoslat`` -> substituting ``lon = x / cos(y)`` gives
  ``dx^2 + 2 x tan(y) dx dy + (1 + x^2 tan^2(y)) dy^2`` -- *not* diagonal.

The orthogonal charts (``sph2``, ``lonlat``, ``math``) are exact; ``loncoslat``
is pulled back through a Jacobian, so its comparisons use a float tolerance.
"""

__all__: tuple[str, ...] = ()

import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.manifolds as cxm
from coordinax._src.metric.matrix import DenseMetric, DiagonalMetric

_ATOL = 1e-6  # loncoslat is a Jacobian pullback, not an analytic closed form

_X, _Y = 0.3, 0.5  # loncoslat base point
_XT = _X * np.tan(_Y)

# (chart, coords, closed-form metric, expected matrix representation)
_CASES = [
    (
        cxc.sph2,
        {"theta": 0.9, "phi": 0.6},
        np.diag([1.0, np.sin(0.9) ** 2]),
        DiagonalMetric,
    ),
    (
        cxc.lonlat_sph2,
        {"lon": 0.6, "lat": 0.7},
        np.diag([np.cos(0.7) ** 2, 1.0]),
        DiagonalMetric,
    ),
    (
        cxc.math_sph2,
        {"theta": 0.6, "phi": 0.9},
        np.diag([np.sin(0.9) ** 2, 1.0]),
        DiagonalMetric,
    ),
    (
        cxc.loncoslat_sph2,
        {"lon_coslat": _X, "lat": _Y},
        np.array([[1.0, _XT], [_XT, 1.0 + _XT**2]]),
        DenseMetric,
    ),
]


def _dense(g):
    """The metric as a plain ``(*batch, n, n)`` array, whatever its type.

    Built by hand rather than via ``to_dense()``, which is not batch-safe until
    GalacticDynamics/coordinax#613 lands.
    """
    if isinstance(g, DiagonalMetric):
        diag = np.asarray(getattr(g.diagonal, "value", g.diagonal))
        return diag[..., :, None] * np.eye(diag.shape[-1])
    return np.asarray(getattr(g.matrix, "value", g.matrix))


@pytest.mark.parametrize(("chart", "coords", "expected", "kind"), _CASES)
def test_metric_matches_closed_form(chart, coords, expected, kind) -> None:
    at = {k: u.Angle(v, "rad") for k, v in coords.items()}
    g = cxm.metric_matrix(cxm.S2, at, chart)
    assert isinstance(g, kind)
    np.testing.assert_allclose(_dense(g), expected, atol=_ATOL)


@pytest.mark.parametrize(("chart", "coords", "expected", "kind"), _CASES)
def test_metric_representation_matches_metric_matrix(
    chart, coords, expected, kind
) -> None:
    del coords, expected
    assert cxm.metric_representation(cxm.S2, chart) is kind


@pytest.mark.parametrize(("chart", "coords", "expected", "kind"), _CASES)
@pytest.mark.parametrize("batch", [(3,), (2, 3)])
def test_metric_is_batch_safe(chart, coords, expected, kind, batch) -> None:
    """A batched point gives (*batch, n, n), each equal to the scalar result."""
    del kind
    at = {
        k: u.Angle(jnp.full(batch, v), "rad")  # constant so `expected` still applies
        for k, v in coords.items()
    }
    got = _dense(cxm.metric_matrix(cxm.S2, at, chart))
    assert got.shape == (*batch, 2, 2)
    np.testing.assert_allclose(got, np.broadcast_to(expected, got.shape), atol=_ATOL)


def test_agrees_with_the_embedded_unit_sphere() -> None:
    """The intrinsic round metric matches the induced metric of the embedding."""
    emb = cxm.EmbeddedManifold(
        intrinsic=cxm.S2, ambient=cxm.R3, embed_map=cxm.TwoSphereIn3D(radius=1)
    )
    for chart, coords, _, _ in _CASES:
        at = {k: u.Angle(v, "rad") for k, v in coords.items()}
        intrinsic = _dense(cxm.metric_matrix(cxm.S2, at, chart))
        induced = np.asarray(cxm.metric_matrix(emb, at, chart).matrix.value)
        np.testing.assert_allclose(intrinsic, induced, atol=_ATOL)


def test_one_sphere_metric_is_unity() -> None:
    at = {"phi": u.Angle(0.6, "rad")}
    g = cxm.metric_matrix(cxm.Sn(1), at, cxc.sph1)
    np.testing.assert_allclose(_dense(g), np.eye(1), atol=_ATOL)
