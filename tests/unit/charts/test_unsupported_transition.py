"""An unsupported `pt_map` refuses, and keeps refusing under ``python -O``.

The rules take the manifolds explicitly as well as the charts, so a caller can
name one that is not the chart's own. That was guarded by bare ``assert``s,
which `python -O` strips: the call then proceeded on a mismatched pair and
returned a number computed from charts that do not belong to the manifolds
given.
"""

import subprocess
import sys

import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.manifolds as cxm
import coordinaxs.api.charts as cxcapi

P3 = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")}


class TestMismatchedManifoldIsRefused:
    """Naming a manifold that is not the chart's own is refused, not computed."""

    def test_wrong_target_manifold(self) -> None:
        with pytest.raises(cxc.ManifoldMismatchError, match="to_M"):
            cxcapi.pt_map(P3, cxm.R3, cxc.cart3d, cxm.R2, cxc.sph3d)

    def test_wrong_source_manifold(self) -> None:
        with pytest.raises(cxc.ManifoldMismatchError, match="from_M"):
            cxcapi.pt_map(P3, cxm.R2, cxc.cart3d, cxm.R3, cxc.sph3d)

    def test_incompatible_cartesian_charts(self) -> None:
        with pytest.raises(cxc.ManifoldMismatchError, match="Cartesian charts differ"):
            cxcapi.pt_map(P3, cxm.R3, cxc.cart3d, cxm.R2, cxc.cart2d)

    def test_cartnd_source_with_mismatched_target(self) -> None:
        """The `CartND` rules guard differently and need their own cases.

        Each accepts `RN` as well as the concrete manifold, so their condition
        is a membership test rather than the equality the paired guard uses --
        a shape no other rule exercises.
        """
        p = {"q": u.Q([3.0, 4.0], "m")}
        with pytest.raises(cxc.ManifoldMismatchError, match="do not line up"):
            cxcapi.pt_map(p, cxm.RN, cxc.cartnd, cxm.R3, cxc.polar2d)

    def test_cartnd_target_with_mismatched_source(self) -> None:
        p = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m")}
        with pytest.raises(cxc.ManifoldMismatchError, match="do not line up"):
            cxcapi.pt_map(p, cxm.R3, cxc.cart2d, cxm.R2, cxc.cartnd)

    def test_it_is_not_an_assertion(self) -> None:
        """`AssertionError` is what this used to raise, and is too broad.

        Anything else in the call raising one would have been indistinguishable
        from a refusal.
        """
        with pytest.raises(cxc.ManifoldMismatchError) as exc:
            cxcapi.pt_map(P3, cxm.R3, cxc.cart3d, cxm.R2, cxc.sph3d)
        assert not isinstance(exc.value, AssertionError)


def test_refusal_survives_optimised_mode() -> None:
    """Under ``-O`` the old bare `assert` vanished and the call returned a value.

    Run in a subprocess because `-O` is an interpreter flag: it strips
    assertions at compile time, so it cannot be toggled from inside this one.
    """
    script = (
        "import unxt as u, coordinax.charts as cxc, coordinax.manifolds as cxm\n"
        "import coordinaxs.api.charts as api\n"
        "p = {'x': u.Q(1.0,'m'), 'y': u.Q(2.0,'m'), 'z': u.Q(3.0,'m')}\n"
        "try:\n"
        "    api.pt_map(p, cxm.R3, cxc.cart3d, cxm.R2, cxc.sph3d)\n"
        "    print('RETURNED')\n"
        "except cxc.ManifoldMismatchError:\n"
        "    print('REFUSED')\n"
    )
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-O", "-c", script], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "REFUSED"
