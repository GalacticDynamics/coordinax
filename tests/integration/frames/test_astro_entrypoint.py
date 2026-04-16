"""Integration tests for optional frame entry-point loading."""

import pytest

import coordinax.frames as cxf

cxastro = pytest.importorskip("coordinax.astro")


def test_astro_icrs_is_exposed_in_coordinax_frames() -> None:
    """`coordinax.frames` exposes astro frames when entry-points are loaded."""
    icrs_cls = cxf.ICRS
    icrs_instance = cxf.icrs

    assert icrs_cls is cxastro.ICRS
    assert isinstance(icrs_instance, icrs_cls)
