"""Tests for ``coordinax_hypothesis.utils.get_all_subclasses``."""

import abc
import warnings

import pytest

from coordinax_hypothesis.utils import get_all_subclasses

# ===========================================================================
# Test fixtures: class hierarchies


class Base:
    """Non-abstract base class for testing."""


class ChildA(Base):
    """Concrete child of Base."""


class ChildB(Base):
    """Another concrete child of Base."""


class GrandChild(ChildA):
    """Grandchild to test recursive discovery."""


class AbstractBase(abc.ABC):
    """Abstract base class for testing abstract exclusion."""

    @abc.abstractmethod
    def method(self) -> None: ...


class ConcreteFromAbstract(AbstractBase):
    """Concrete implementation of AbstractBase."""

    def method(self) -> None:
        pass


class AbstractNameOnly:
    """Not truly abstract, but name starts with 'Abstract'."""


class ConcreteFromAbstractName(AbstractNameOnly):
    """Concrete child of AbstractNameOnly."""


# Mixin for filter tests
class MixinA:
    """Mixin class for filter testing."""


class MixinB:
    """Another mixin class for filter testing."""


class FilterBase:
    """Base for filter tests."""


class FilterChildBothMixins(FilterBase, MixinA, MixinB):
    """Child that inherits both mixins."""


class FilterChildOneMixin(FilterBase, MixinA):
    """Child that inherits only MixinA."""


class FilterChildNoMixin(FilterBase):
    """Child with no mixins."""


# Exclude hierarchy
class ExcludeBase:
    """Base for exclude tests."""


class ExcludeChildA(ExcludeBase):
    """Child to be excluded."""


class ExcludeChildB(ExcludeBase):
    """Child to keep."""


class ExcludeGrandChild(ExcludeChildA):
    """Grandchild of excluded branch."""


# Because get_all_subclasses is ``@ft.cache-decorated``, we clear the cache
# before each test to avoid cross-test pollution.
@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the get_all_subclasses cache before each test."""
    get_all_subclasses.cache_clear()
    yield
    get_all_subclasses.cache_clear()


# ===========================================================================


class TestBasicDiscovery:
    """Test basic subclass discovery."""

    def test_finds_direct_children(self):
        result = get_all_subclasses(Base, exclude_abstract=False)
        assert ChildA in result
        assert ChildB in result

    def test_finds_grandchildren(self):
        result = get_all_subclasses(Base, exclude_abstract=False)
        assert GrandChild in result

    def test_returns_tuple(self):
        result = get_all_subclasses(Base, exclude_abstract=False)
        assert isinstance(result, tuple)

    def test_base_class_not_included(self):
        result = get_all_subclasses(Base, exclude_abstract=False)
        assert Base not in result

    def test_no_duplicates(self):
        result = get_all_subclasses(Base, exclude_abstract=False)
        assert len(result) == len(set(result))


class TestExcludeAbstract:
    """Test the ``exclude_abstract`` parameter."""

    def test_excludes_abc_abstract(self):
        result = get_all_subclasses(AbstractBase, exclude_abstract=True)
        assert AbstractBase not in result
        assert ConcreteFromAbstract in result

    def test_excludes_name_based_abstract(self):
        """Classes whose name starts with 'Abstract' are excluded."""
        result = get_all_subclasses(AbstractNameOnly, exclude_abstract=True)
        assert AbstractNameOnly not in result
        assert ConcreteFromAbstractName in result

    def test_includes_abstract_when_disabled(self):
        result = get_all_subclasses(AbstractBase, exclude_abstract=False)
        # AbstractBase itself is the root, not a subclass of itself
        # but its abstract subclasses (if any) would be included.
        # ConcreteFromAbstract should still be present either way.
        assert ConcreteFromAbstract in result


class TestFilter:
    """Test the ``filter`` parameter."""

    def test_single_filter(self):
        result = get_all_subclasses(FilterBase, filter=MixinA)
        assert FilterChildBothMixins in result
        assert FilterChildOneMixin in result
        assert FilterChildNoMixin not in result

    def test_tuple_filter_all_must_match(self):
        result = get_all_subclasses(FilterBase, filter=(MixinA, MixinB))
        assert FilterChildBothMixins in result
        assert FilterChildOneMixin not in result
        assert FilterChildNoMixin not in result

    def test_default_filter_is_object(self):
        """Default filter=object matches everything."""
        result = get_all_subclasses(Base, exclude_abstract=False)
        assert ChildA in result
        assert ChildB in result
        assert GrandChild in result


class TestExclude:
    """Test the ``exclude`` parameter."""

    def test_excludes_specified_class(self):
        result = get_all_subclasses(ExcludeBase, exclude=(ExcludeChildA,))
        assert ExcludeChildA not in result
        assert ExcludeChildB in result

    def test_excludes_subclasses_of_excluded(self):
        """Subclasses of excluded classes are also excluded (covariant)."""
        result = get_all_subclasses(ExcludeBase, exclude=(ExcludeChildA,))
        assert ExcludeGrandChild not in result

    def test_empty_exclude(self):
        result = get_all_subclasses(ExcludeBase, exclude=())
        assert ExcludeChildA in result
        assert ExcludeChildB in result
        assert ExcludeGrandChild in result


class TestWarning:
    """Test that a warning is emitted when no subclasses are found."""

    def test_warns_when_no_subclasses(self):
        class Isolated:
            """A class with no subclasses."""

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = get_all_subclasses(Isolated)

        assert result == ()
        assert len(caught) == 1
        assert issubclass(caught[0].category, UserWarning)
        assert "No subclasses found" in str(caught[0].message)


class TestCaching:
    """Test that the result is cached via functools.cache."""

    def test_same_object_returned(self):
        r1 = get_all_subclasses(Base, exclude_abstract=False)
        r2 = get_all_subclasses(Base, exclude_abstract=False)
        assert r1 is r2

    def test_cache_clear_gives_fresh_result(self):
        r1 = get_all_subclasses(Base, exclude_abstract=False)
        get_all_subclasses.cache_clear()
        r2 = get_all_subclasses(Base, exclude_abstract=False)
        # After clearing, the content should be the same but identity may differ
        assert set(r1) == set(r2)
