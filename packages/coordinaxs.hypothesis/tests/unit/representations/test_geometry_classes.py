"""Tests for the geometry_classes and geometries strategies."""

import hypothesis.strategies as st
import pytest
from hypothesis import given

import coordinax.representations as cxr

import coordinaxs.hypothesis.main as cxst
import coordinaxs.hypothesis.representations as cxrst
from coordinaxs.hypothesis.utils import get_all_subclasses

# ============================================================================
# geometry_classes
# ============================================================================


class TestGeometryClasses:
    """Tests for the geometry_classes strategy."""

    @given(geom_cls=cxrst.geometry_classes())
    def test_returns_subclass_of_abstract_geometry(
        self, geom_cls: type[cxr.AbstractGeometry]
    ) -> None:
        """Generated class is always a subclass of AbstractGeometry."""
        assert issubclass(geom_cls, cxr.AbstractGeometry)

    @given(geom_cls=cxrst.geometry_classes())
    def test_never_returns_abstract_base(
        self, geom_cls: type[cxr.AbstractGeometry]
    ) -> None:
        """Generated class is never the abstract base AbstractGeometry itself."""
        assert geom_cls is not cxr.AbstractGeometry

    @given(geom_cls=cxrst.geometry_classes())
    def test_is_concrete_by_default(self, geom_cls: type[cxr.AbstractGeometry]) -> None:
        """Generated class is concrete and can be instantiated."""
        # Concrete classes should be instantiable with no args
        instance = geom_cls()
        assert isinstance(instance, cxr.AbstractGeometry)

    @given(geom_cls=cxrst.geometry_classes(include=(cxr.PointGeometry,)))
    def test_include_restricts_to_provided_classes(
        self, geom_cls: type[cxr.AbstractGeometry]
    ) -> None:
        """Include parameter restricts generation to provided classes."""
        assert geom_cls is cxr.PointGeometry

    @given(data=st.data())
    def test_empty_candidates_raises_value_error(self, data: st.DataObject) -> None:
        """Excluding all candidates raises ValueError."""
        all_geometries = get_all_subclasses(cxr.AbstractGeometry, exclude_abstract=True)
        with pytest.raises(ValueError, match="No role classes left after exclusions"):
            data.draw(cxrst.geometry_classes(exclude=tuple(all_geometries)))

    def test_also_accessible_via_main(self) -> None:
        """geometry_classes is re-exported from coordinaxs.hypothesis.main."""
        assert cxst.geometry_classes is cxrst.geometry_classes


# ============================================================================
# geometries
# ============================================================================


class TestGeometries:
    """Tests for the geometries strategy."""

    @given(geom=cxrst.geometries())
    def test_returns_abstract_geometry_instance(
        self, geom: cxr.AbstractGeometry
    ) -> None:
        """Generated value is an AbstractGeometry instance."""
        assert isinstance(geom, cxr.AbstractGeometry)

    @given(geom=cxrst.geometries(include=(cxr.PointGeometry,)))
    def test_include_restricts_to_point_geometry(
        self, geom: cxr.AbstractGeometry
    ) -> None:
        """Include parameter restricts instances to the provided classes."""
        assert isinstance(geom, cxr.PointGeometry)

    @given(geom=cxrst.geometries())
    def test_never_returns_abstract_class(self, geom: cxr.AbstractGeometry) -> None:
        """Generated value is never an instance of the abstract base itself (only concrete)."""
        # AbstractGeometry is abstract; so all instances are subclass instances
        assert type(geom) is not cxr.AbstractGeometry

    def test_also_accessible_via_main(self) -> None:
        """Geometries is re-exported from coordinaxs.hypothesis.main."""
        assert cxst.geometries is cxrst.geometries
