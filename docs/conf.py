"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import importlib.metadata
from datetime import datetime

from typing import TYPE_CHECKING, Any, cast

import pytz
from docutils.nodes import Element, Node, reference
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.util.nodes import make_refnode

if TYPE_CHECKING:
    from sphinx.domains.python import PythonDomain

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

# -- Napoleon settings ---------------------------------------------------
# Render class ``Attributes`` sections as ``:ivar:`` info fields rather than
# standalone ``.. attribute::`` directives. With ``automodule``/``autoclass``
# already documenting the real attributes and properties, the directive form
# emits a second object description for the same name ("duplicate object
# description"); the ``:ivar:`` form keeps the curated prose without a clashing
# target.
napoleon_use_ivar = True


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
    ("py:class", "coordinax.vectors._src.core.Point"),
    ("py:class", "coordinax.representations._src.semantics.AbstractSemanticKind"),
    ("py:class", "coordinax.representations._src.geom.PointGeometry"),
    ("py:class", "coordinax.representations._src.basis.AbstractBasis"),
    ("py:class", "coordinax._src.charts.d3.LonLatSpherical3D"),
    # --- External types with no resolvable Sphinx inventory entry ---
    # unxt's objects.inv does not expose these public names under these keys
    # (it uses different qualified paths internally), so intersphinx cannot
    # resolve them. The bare / import-alias forms leak in from hand-written
    # docstrings and generated type signatures.
    ("py:class", "AbstractQuantity"),
    ("py:class", "Quantity"),
    ("py:class", "unxt.AbstractQuantity"),
    ("py:class", "unxt.Quantity"),
    ("py:class", "u.AbstractUnit"),
    ("py:class", "jnp.ndarray"),
    # typing.TypeIs is only in the CPython inventory from 3.13; the pinned
    # Python intersphinx target predates it.
    ("py:class", "typing.TypeIs"),
    # plum does not publish a Sphinx inventory entry for these names.
    ("py:exc", "plum.NotFoundLookupError"),
    ("py:func", "plum.dispatch"),
    # TypeVars / jaxtyping shape fragments that leak verbatim into
    # sphinx-autodoc-typehints-generated signatures (source ``<unknown>``);
    # they are not documentable objects.
    ("py:class", "Ts"),
    ("py:class", "Rep"),
    ("py:class", "StaticValue"),
    ("py:class", "3"),
    ("py:class", "'N N'"),
    # ``ChartT`` is a TypeVar; ``TypeIs`` leaks in bare and from typing_extensions
    # (the pinned intersphinx Python target predates ``typing.TypeIs``).
    ("py:class", "ChartT"),
    ("py:class", "TypeIs"),
    ("py:class", "typing_extensions.TypeIs"),
    # AbstractVector's docstring references the ``AbstractCoordinate`` bundle base
    # by a public path it is not (re-)exported under; not a documentable target.
    ("py:obj", "coordinax.vectors.AbstractCoordinate"),
    # A napoleon-misparsed prose word ("... the same ...") in an aggregated
    # plum docstring; nothing is legitimately named ``same``.
    ("py:obj", "same"),
]

# TypedNdArray is a JAX-private type (jax._src.basearray) with no public docs.
# jax._src.* are private JAX implementation paths never in the public inventory.
nitpick_ignore_regex = [
    (r"py:class", r"jaxtyping\..*"),  # TODO: remove
    (r"py:class", r".*TypedNdArray.*"),
    (r"py:class", r"jax\._src\..*"),
    # Private ``coordinax``/``coordinaxs`` ``_src`` implementation paths (and
    # their TypeVars) leak into type signatures; the public re-exports (e.g.
    # ``coordinax.vectors.Point``) are the documented targets. Ignore the
    # private forms rather than the public API. Fully anchored (``^…$``) and
    # built from dotted-segment groups (no greedy ``.*``) so third-party
    # ``._src.`` targets are unaffected. ``py:.*`` covers every Python role
    # (class/data/obj/…) since ``_src`` symbols are never intentionally
    # documented in any role. Removes ~940 warnings.
    (r"py:.*", r"^coordinaxs?(\.\w+)*\._src(\.\w+)+$"),
    # External libraries without a resolvable Sphinx inventory (MkDocs sites or
    # no published objects.inv): render their type references as plain text.
    (r"py:.*", r"wadler_lindig\..*"),
    (r"py:.*", r"unxt_hypothesis\..*"),
    (r"py:.*", r"optype\..*"),
    # quax-blocks mixins are private (``_src``) implementation details with no
    # published inventory; they leak into base-class signatures.
    (r"py:.*", r"quax_blocks\._src\..*"),
    # Parametrized unxt Quantity aliases (``Quantity[PhysicalType('length')]``,
    # ``['angle']``, …) are emitted verbatim into signatures but are not
    # resolvable inventory targets.
    (r"py:.*", r"unxt\._src\.quantity\.quantity\.Quantity\[PhysicalType\(.*"),
    # beartype-validator annotations (``typing.Annotated[..., beartype.vale.Is
    # [...]]``) are emitted verbatim into signatures and are not doc targets.
    (r"py:.*", r"typing\.Annotated\[.*"),
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


def _py_suffix_index(
    env: BuildEnvironment, pydomain: "PythonDomain"
) -> dict[str, list[str]]:
    """Map every dotted suffix of each py object name to the full names.

    Built once per build and cached on ``env`` (missing-reference resolution runs
    after reading, when the object inventory is complete and stable). A lookup by
    ``target`` then returns exactly the names for which
    ``name == target or name.endswith(f".{target}")``.
    """
    index: dict[str, list[str]] | None = getattr(env, "_coordinax_suffix_index", None)
    if index is None:
        index = {}
        for name in pydomain.objects:
            parts = name.split(".")
            for i in range(len(parts)):
                index.setdefault(".".join(parts[i:]), []).append(name)
        env._coordinax_suffix_index = index  # ty: ignore[unresolved-attribute]
    return index


def _resolve_internal_short_names(
    app: Sphinx,
    env: BuildEnvironment,
    node: Element,
    contnode: Element,
) -> Node | None:
    """Resolve a bare short name to a coordinax object of the same name.

    Plum's combined ``__doc__`` and hand-written Napoleon ``Parameters`` /
    ``Returns`` / ``See Also`` sections reference project objects by their short
    name (e.g. ``Representation``, ``pt_map``, ``minkowski4d``) rather than the
    fully-qualified path Sphinx indexes them under. Look the target up in this
    project's own Python object inventory and, when it maps unambiguously to a
    single public object, return a real internal cross-reference. External
    names never appear in this inventory, so they fall through to the normal
    (nitpick) handling.
    """
    if node.get("refdomain") != "py":
        return None
    target = node.get("reftarget", "")
    if not target:
        return None
    pydomain = cast("PythonDomain", env.get_domain("py"))
    # ``target`` matches an object when it equals the full name or a trailing
    # dotted suffix of it (``name == target or name.endswith(f".{target}")``).
    # Look that up in a per-build suffix index so each reference is O(1) rather
    # than an O(N) scan of every documented object.
    matches = _py_suffix_index(env, pydomain).get(target, ())
    # Prefer public paths over private ``._src.`` implementation paths.
    public = [name for name in matches if "._src." not in name]
    candidates = public or list(matches)
    if len(candidates) != 1:
        return None  # unknown or ambiguous → leave for normal handling
    name = candidates[0]
    obj = pydomain.objects[name]
    return make_refnode(
        app.builder, node["refdoc"], obj.docname, obj.node_id, contnode, name
    )


# -- Dollar-math in docstrings ----------------------------------------
# The Markdown (MyST) pages render maths with the ``dollarmath`` extension, so
# docstrings are written with the same ``$...$`` / ``$$...$$`` convention for
# consistency. Autodoc, however, feeds docstrings to the *reStructuredText*
# parser, which has no dollar-math support: inside ``$...$`` a LaTeX subscript
# such as ``h_\theta`` is misread as an RST reference (``Unknown target name:
# "h"``), ``|\nu|`` as a substitution, ``**`` as strong emphasis, and multi-line
# ``$$``-blocks as block quotes (``Unexpected indentation``). Convert dollar-math
# to the RST ``:math:`` role / ``.. math::`` directive before parsing so the
# maths both renders correctly *and* stops emitting spurious warnings. Doctest
# blocks in these docstrings never contain ``$``, so this is safe.
import re  # noqa: E402

_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
# Inline maths may wrap across a line break (but not a blank line), so match any
# run of non-``$`` characters that does not contain a blank line.
_INLINE_MATH_RE = re.compile(r"\$(?!\$)((?:[^$]|\n(?!\s*\n))+?)\$")


def _display_math_repl(match: "re.Match[str]") -> str:
    """Replace a ``$$...$$`` block with an ``.. math::`` directive."""
    body = "\n".join(
        "   " + line.strip() for line in match.group(1).strip().splitlines()
    )
    return f"\n\n.. math::\n\n{body}\n\n"


def _convert_dollar_math(
    app: Sphinx,
    what: str,
    name: str,
    obj: object,
    options: object,
    lines: list[str],
) -> None:
    """Rewrite ``$...$`` / ``$$...$$`` maths in docstrings to RST math."""
    # Restrict to this project's own docstrings: ``$`` reliably means math only
    # in coordinax's controlled docstrings, so an external package entering the
    # build (where ``$`` might be currency, a shell var, etc.) is left untouched.
    if not name.startswith(("coordinax", "coordinaxs")):
        return
    if not any("$" in line for line in lines):
        return
    text = "\n".join(lines)
    text = _DISPLAY_MATH_RE.sub(_display_math_repl, text)
    text = _INLINE_MATH_RE.sub(
        lambda m: ":math:`" + " ".join(m.group(1).split()) + "`", text
    )
    lines[:] = text.split("\n")


# Members inherited from the external ``quax.Value`` materialisation protocol
# (defined on ``quax.Value`` and/or overridden without a docstring on
# ``unxt.AbstractQuantity``). Their upstream docstrings are written in MkDocs
# Markdown (autorefs, ``!!!`` admonitions, code fences) that Sphinx's RST parser
# renders as raw text, so we document these inherited members with a concise RST
# summary instead. See the quax docs for the full details.
_QUAX_VALUE_DOCS: dict[str, str] = {
    "aval": "Return the ``jax.core.AbstractValue`` this value presents to JAX.",
    "default": (
        "Default multiple-dispatch rule used when no rule is registered for a "
        "primitive."
    ),
    "materialise": (
        "Materialise this value into a concrete JAX type (usually an array)."
    ),
    "enable_materialise": "Whether this value may be materialised into a JAX type.",
}
_QUAX_VALUE_SOURCES = ("quax", "unxt")


def _clean_quax_value_docstrings(
    app: Sphinx,
    what: str,
    name: str,
    obj: object,
    options: object,
    lines: list[str],
) -> None:
    """Replace the quax-materialisation members' MkDocs docstrings with clean RST.

    Only rewrites the external inherited forms (defining module ``quax``/``unxt``,
    or a docstring still carrying MkDocs markup); a coordinax-authored override of
    the same name keeps its own docstring.
    """
    summary = _QUAX_VALUE_DOCS.get(name.rsplit(".", 1)[-1])
    if summary is None:
        return
    is_external = (getattr(obj, "__module__", "") or "").startswith(_QUAX_VALUE_SOURCES)
    has_mkdocs = any("!!!" in ln or "[`" in ln for ln in lines)
    if is_external or has_mkdocs:
        lines[:] = [summary, ""]


def setup(app: Sphinx, /) -> None:
    app.connect("missing-reference", _resolve_short_names)
    app.connect("missing-reference", _resolve_internal_short_names)
    app.connect("autodoc-process-docstring", _convert_dollar_math)
    app.connect("autodoc-process-docstring", _clean_quax_value_docstrings)
