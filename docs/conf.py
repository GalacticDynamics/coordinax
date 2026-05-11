"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import importlib.metadata
from datetime import datetime

from typing import Any

import pytz
from docutils.nodes import Element, Node, reference
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment

# -- Project information -----------------------------------------------------

author = "Coordinax Developers"
project = "coordinax"
copyright = f"{datetime.now(pytz.timezone('UTC')).year}, {author}"
version = importlib.metadata.version("coordinax")

master_doc = "index"
language = "en"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",  # Jupyter notebook support via MyST (includes myst_parser)
    "sphinx_design",
    "sphinx.ext.autodoc",  # TODO: replace with autodoc2
    "sphinx.ext.autosummary",  # TODO: replace with autodoc2
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx-prompt",
    "sphinxext.opengraph",
    # "sphinxext.rediraffe",  # Add redirects
    "sphinx_togglebutton",
    "sphinx_tippy",
]

# Wikipedia's REST API requires a User-Agent header; sphinx_tippy doesn't set
# one, so requests get a 403 with non-JSON body, causing a build warning.
# TODO: periodically check if this is fixed
tippy_enable_wikitips = False

python_use_unqualified_type_names = True

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
    "guides/perf.md",  # Excluded: converted to perf.ipynb by jupytext
]

source_suffix = [".md", ".rst", ".ipynb"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    # Canonical URL (jax.readthedocs.io now redirects here)
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "jaxtyping": ("https://docs.kidger.site/jaxtyping/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "equinox": ("https://docs.kidger.site/equinox/", None),
    "plum": ("https://beartype.github.io/plum/", None),
    "quax": ("https://docs.kidger.site/quax/", None),
    "unxt": ("https://unxt.readthedocs.io/en/latest/", None),
}

# -- Autodoc settings ---------------------------------------------------

autodoc_typehints = "description"
autodoc_typehints_format = "short"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

always_document_param_types = True
typehints_use_signature = True


nitpick_ignore = [
    # Keep this ignore: Sphinx emits unresolved typing.Union references from
    # generated type signatures with <unknown> source locations, so there is no
    # stable doc target to fix directly yet.
    # TODO: Revisit after upgrading Sphinx and/or sphinx-autodoc-typehints.
    ("py:data", "typing.Union"),
    # ArrayLike is documented as py:data in JAX (it's a type alias), but
    # sphinx_autodoc_typehints emits it as py:class — the mismatch cannot be
    # resolved via intersphinx regardless of URL.
    ("py:class", "ArrayLike"),
    ("py:class", "jax.typing.ArrayLike"),
    ("py:class", "unxt.Angle"),
    ("py:class", "coordinax.distances._src.base.AbstractDistance"),
    # Private internal helper class from unxt with no public docs
    ("py:class", "unxt._src.quantity.base._QuantityIndexUpdateHelper"),
    ("py:class", "dataclassish._src.converters.PassThroughTs"),
    ("py:class", "dataclassish._src.converters.ArgT"),
    ("py:class", "unxt._src.quantity.quantity.Quantity[PhysicalType('length')]"),
    ("py:class", "coordinax.vectors._src.core.Point"),
    ("py:class", "coordinax.representations._src.semantics.AbstractSemanticKind"),
    ("py:class", "coordinax.representations._src.geom.PointGeometry"),
    ("py:class", "coordinax.representations._src.basis.AbstractBasis"),
    ("py:class", "coordinax._src.charts.d3.LonLatSpherical3D"),
]

# TypedNdArray is a JAX-private type (jax._src.basearray) with no public docs.
# jax._src.* are private JAX implementation paths never in the public inventory.
nitpick_ignore_regex = [
    (r"py:class", r"jaxtyping\..*"),  # TODO: remove
    (r"py:class", r".*TypedNdArray.*"),
    (r"py:class", r"jax\._src\..*"),
]

# -- MyST Setting -------------------------------------------------

myst_enable_extensions = [
    "amsmath",  # for direct LaTeX math
    "attrs_block",  # enable parsing of block attributes
    "attrs_inline",  # apply syntax highlighting to inline code
    "colon_fence",
    "deflist",
    "dollarmath",  # for $, $$
    # "linkify",  # identify “bare” web URLs and add hyperlinks:
    "smartquotes",  # convert straight quotes to curly quotes
    "substitution",  # substitution definitions
]
myst_heading_anchors = 3

# -- MyST-NB settings (Jupyter notebook support) --------------------------

nb_execution_mode = "cache"
nb_execution_cache_path = "_build/.jupyter_cache"
nb_execution_raise_on_error = True
nb_execution_timeout = 100

# myst_substitutions = {
#     "ArrayLike": "{obj}`jaxtyping.ArrayLike`",
#     "Any": "{obj}`typing.Any`",
# }


rst_prolog = """
.. py:module:: coordinax
.. py:module:: astropy
.. py:module:: plum
.. py:module:: JAX
.. py:module:: unxt-hypothesis
"""


# -- HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "coordinax"
html_logo = "_static/favicon.png"  # TODO: an svg
html_copy_source = True
html_favicon = "_static/favicon.png"

html_static_path = ["_static"]
html_css_files = ["custom_toc.css", "custom_tooltip.css"]

html_theme_options: dict[str, Any] = {
    "home_page_in_toc": True,
    "repository_url": "https://github.com/GalacticDynamics/coordinax",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": False,
    "use_issues_button": True,
    "show_toc_level": 2,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/GalacticDynamics/coordinax",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/coordinax/",
            "icon": "https://img.shields.io/pypi/v/coordinax",
            "type": "url",
        },
        {
            "name": "Zenodo",
            "url": "https://zenodo.org/doi/10.5281/zenodo.10850557",
            "icon": "fa fa-quote-right",
        },
    ],
}

# -- Missing-reference handler ----------------------------------------
# Plum's combined __doc__ emits bare short names (e.g. `PhysicalType`) inside
# ``.. py:function::`` RST directives. Intersphinx cannot match short names —
# only fully-qualified keys are in its inventory. This handler intercepts those
# unresolved references and returns real hyperlinks instead of suppressing them.

# Map bare short name → canonical documentation URL
_SHORT_NAME_URLS: dict[str, str] = {
    # astropy: not in Sphinx inventory under short name
    "PhysicalType": (
        "https://docs.astropy.org/en/stable/api/astropy.units.PhysicalType.html"
    ),
    # types.NoneType (Python 3.10+): intersphinx key is qualified; bare name
    # won't resolve
    "NoneType": "https://docs.python.org/3/library/types.html#types.NoneType",
    # numpy.ndarray: intersphinx key is qualified; bare name won't resolve
    "ndarray": "https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html",
    # jaxtyping.Array is an alias for jax.Array; jaxtyping has no Sphinx
    # inventory (MkDocs)
    "Array": "https://docs.jax.dev/en/latest/_autosummary/jax.Array.html",
}


def _resolve_short_names(
    app: Sphinx,
    env: BuildEnvironment,
    node: Element,
    contnode: Element,
) -> Node | None:
    """Resolve known bare short names to external documentation links."""
    if node.get("refdomain") != "py":
        return None
    target = node.get("reftarget", "")
    url = _SHORT_NAME_URLS.get(target)
    if url is None:
        return None
    return reference("", "", contnode, internal=False, refuri=url)


def setup(app: Sphinx, /) -> None:
    app.connect("missing-reference", _resolve_short_names)
