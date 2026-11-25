See the [Scientific Python Developer Guide][spc-dev-intro] for a detailed description of best practices for developing scientific packages.

[spc-dev-intro]: https://learn.scientific-python.org/development/

# Quick development

The fastest way to start with development is to use nox. If you don't have nox, you can use `pipx run nox` to run it without installing, or `pipx install nox`. If you don't have pipx (pip for applications), then you can install with `pip install pipx` (the only case were installing an application with regular pip is reasonable). If you use macOS, then pipx and nox are both in brew, use `brew install pipx nox`.

To use, run `nox`. This will lint and test using every installed version of Python on your system, skipping ones that are not installed. You can also run specific jobs:

```console
nox -s lint  # Lint only
nox -s tests  # Python tests
nox -s docs -- --serve  # Build and serve the docs
nox -s build  # Make an SDist and wheel
```

Nox handles everything for you, including setting up an temporary virtual environment for each run.

# Setting up a development environment manually

You can set up a development environment by running:

```bash
python3 -m venv .venv
source ./.venv/bin/activate
pip install -v -e .[dev]
```

If you have the [Python Launcher for Unix](https://github.com/brettcannon/python-launcher), you can instead do:

```bash
py -m venv .venv
py -m install -v -e .[dev]
```

# Post setup

You should prepare prek, which will help you by checking that commits pass required checks:

```bash
pip install prek # or brew install prek on macOS
prek install # Will install a prek hook into the git repo
```

You can also/alternatively run `prek run` (changes only) or `prek run --all-files` to check even without installing the hook.

# Testing

Use pytest to run the unit checks:

```bash
pytest
```

# Coverage

Use pytest-cov to generate coverage reports:

```bash
pytest --cov=coordinax
```

# Building docs

You can build the docs using:

```bash
nox -s docs
```

You can see a preview with:

```bash
nox -s docs -- --serve
```

# Pre-commit

This project uses prek for all style checking. While you can run it with nox, this is such an important tool that it deserves to be installed on its own. Install prek and run:

```bash
prek run --all-files
```

to check all files.

# The code graph

`graphify-out/graph.json` is a [graphify](https://github.com/Graphify-Labs/graphify) map of this repository: its symbols and the relationships between them. AI assistants query it instead of grepping, and you can too. Extraction is local tree-sitter parsing — no model call, and nothing leaves your machine — and every edge is marked `EXTRACTED` (explicit in the source) or `INFERRED` (resolved by graphify), so you can tell what was read from what was deduced.

It covers the library an agent asks questions about, not the whole tree. [`.graphifyignore`](../.graphifyignore) excludes `tests/` (60% of the nodes on its own), the repository tooling, the `coordinaxs.hypothesis` strategies and all YAML, Markdown and images — 13.7 MB of graph narrowed to 4.4 MB, with the arithmetic written down in that file. `tests/` mirrors the source layout, so the tests for a symbol are where the symbol is.

The graph is committed, and the commit checks rebuild it. Install the pinned release and register the merge driver once per clone, so two branches that both changed the graph merge cleanly:

```bash
uv tool install "graphifyy==0.9.119"
PY="$(sed -n '1s/^#!//p' "$(command -v graphify)")"
git config merge.graphify.name "graphify graph.json union merge"
git config merge.graphify.driver "\"$PY\" -m graphify merge-driver %O %A %B"
```

Git will not take a merge-driver command from a tracked file, so that `git config` pair cannot be committed — it is the one piece of this setup every clone runs by hand.

Then query it:

```bash
graphify query "how are vectors converted between representations"   # a scoped subgraph
graphify explain "AbstractChart"                                     # one symbol and its neighbours
graphify path "AbstractVector" "AbstractChart"                       # how two things connect
open graphify-out/graph.html                                         # the clickable graph
```

Day to day:

- The rebuild runs with your other commit checks. Like a formatter, it rebuilds and stops the commit; `git add graphify-out/graph.json` and commit again.
- Commit without rebuilding: `SKIP=graphify-update git commit ...`.
- If a rebuild is ever refused because the graph would shrink — you deleted code — use `GRAPHIFY_FORCE=1 git commit ...`. It usually just shrinks and says nothing.
- If the graph ever conflicts in a merge, rebuild rather than edit it: `rm graphify-out/graph.json && graphify update . --force && git add graphify-out/graph.json`.
- A changed `graph.json` in a pull request is generated output. It is one line, collapsed by GitHub, and is not reviewed.
- The rebuild will rewrite the graph the first time you commit from a working environment, and the weekly `Refresh code graph` job may rewrite it back. That is expected, not a bug: graphify's output depends on which files are on disk, so a built checkout and a fresh clone disagree about community numbering even though every symbol is identical. A generated `src/coordinax/_version.py` alone does it. Stage it and commit.
- The graph the repository carries is the fresh-clone flavour, because that is what CI and the weekly job reproduce. To rebuild that flavour deliberately — before a release, say, or after a rebase — clone the repository somewhere outside this tree and build there, since anything extra on disk changes the result:

  ```bash
  git clone --no-hardlinks . ../coordinax-graph-build
  cd ../coordinax-graph-build && git checkout <branch>
  PYTHONHASHSEED=0 graphify update . --force
  cp graphify-out/graph.json <this checkout>/graphify-out/graph.json
  ```
