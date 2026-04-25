---
name: gitmoji-commit
description: "Commit staged changes using cz_gitmoji format. Use when: staging is done and ready to commit, make a commit, commit these changes, ready to commit, write a commit message, commit staged changes."
argument-hint: "Optional: describe the intent if you want to guide type selection"
---

# Gitmoji Commit

## When to Use

Load this skill when the user says any of:

- "I have staged changes, commit them"
- "ready to commit"
- "make a commit"
- "commit these changes"
- "write a commit message"
- "commit staged changes"

## Commit Format

```
{icon} {type}({scope}): {subject}

{body}
```

Or, when scope is omitted:

```
{icon} {type}: {subject}

{body}
```

**Example:** `✨ feat(distances): add parallax-to-distance conversion`

- Scope is optional. Omit parentheses if no clear module scope.
- There IS a space between icon and type: prefix is `"{icon} {type}"` (e.g. `"✨ feat"`).
- Subject: imperative mood, lowercase, no trailing period, ≤72 chars.
- Body: explain _why_, not _what_. Optional but recommended for non-trivial changes.

## Procedure

### Step 0 — Sync the gitmoji type reference when needed

Use this step when the user asks to refresh the type reference, or before committing if the plugin version may have changed.

1. Check whether the reference is stale:

```bash
uv run --group dev python .github/skills/gitmoji-commit/scripts/update_types_reference.py --check
```

2. If the check reports stale output, regenerate the reference file and version metadata:

```bash
uv run --group dev python .github/skills/gitmoji-commit/scripts/update_types_reference.py --write
```

3. This update process keeps [the full type reference](./references/types.md) aligned with:

- `dependency-groups.dev` in `pyproject.toml` (required specifier)
- `uv.lock` (locked concrete version)
- the installed `cz-conventional-gitmoji` metadata used to generate the table

### Step 1 — Inspect staged changes

Run both commands to understand what's staged:

```bash
git diff --staged --stat
git diff --staged
```

### Step 2 — Choose type and scope

Use [the full type reference](./references/types.md) to pick the most specific type.

**Quick-pick for this project (most common):**

| Icon | Type | When |
| --- | --- | --- |
| ✨ | `feat` | New public API, new class, new function |
| 🐛 | `fix` | Bug fix, wrong result |
| 🩹 | `fix-simple` | Non-critical fix (typo in code, minor correction) |
| 📝 | `docs` | Docstrings, README, guides, tutorials |
| 💬 | `text` | Update text/literals in existing docs (not structural) |
| ♻️ | `refactor` | Internal restructure, no behaviour change |
| ✅ | `test` | Add or update tests |
| 🧹 | `chore` | Miscellaneous maintenance |
| 🔧 | `config` | pyproject.toml, noxfile, pre-commit, config files |
| 💚 | `ci` | GitHub Actions, CI workflows |
| 👷 | `build` | Build system changes |
| ⬆️ | `dep-bump` | Upgrade dependencies |
| ➕ | `dep-add` | Add a dependency |
| ➖ | `dep-rm` | Remove a dependency |
| 🏷️ | `types` | Type annotations only |
| 🗑️ | `deprecation` | Mark something deprecated |
| ⚰️ | `dead` | Remove dead/unused code |
| 💥 | `boom` | Breaking change |
| 🔨 | `script` | Scripts in `scripts/` directory |
| 🧱 | `infra` | Infrastructure (packaging, workspace setup) |
| 🧑‍💻 | `devxp` | Developer experience improvements |

**Scope** = the module or package being changed. Examples:

- `distances`, `angles`, `vectors`, `frames`, `manifolds`, `charts`
- `ci`, `tests`, `docs`
- Omit scope if the change spans multiple unrelated modules.

### Step 3 — Write the message

Construct:

- **Subject line (with scope)**: `{icon} {type}({scope}): {short imperative description}`
- **Subject line (without scope)**: `{icon} {type}: {short imperative description}`
- **Body** (if needed): one paragraph explaining _why_ the change was made, not _what_ changed (git diff shows what).

### Step 4 — Execute the commit

With scope:

```bash
git commit -m "{icon} {type}({scope}): {subject}" -m "{body}"
```

Without scope:

```bash
git commit -m "{icon} {type}: {subject}" -m "{body}"
```

Omit the second `-m` if no body is needed.

**Do NOT use** `cz commit` — it requires interactive input via arrow-key selection.

### Step 5 — Confirm

Show the user the result of `git log --oneline -1` to confirm the commit was recorded correctly.

## Notes

- Never stage additional files; only commit what's already staged.
- If `git diff --staged` returns nothing, tell the user nothing is staged.
- For breaking changes, append `!` after the type+scope: `💥 boom(vectors)!: remove deprecated CartesianPos2D`
- The body should be plain prose, no markdown bullets (keep it simple for git log readability).
