# No Comments

**CRITICAL:** Do NOT add comments to code. None. The code itself must be the documentation.

This applies to all comment syntaxes across all languages: `#`, `//`, `/* */`, `<!-- -->`, `"""..."""` used as inline commentary, etc.

## What NOT to do

```python
# BAD — narrative comment describing what the code does
def fetch_user(user_id: int) -> User:
    # Look up the user in the database
    user = db.get_user(user_id)
    # Return the user
    return user

# BAD — explaining the change you just made
# Updated to use the new API
result = new_api.call()

# BAD — TODO/FIXME/NOTE comments
# TODO: handle the empty case
# NOTE: this was refactored from the old version
# FIXME: temporary workaround

# BAD — commented-out code
# old_logic()
new_logic()

# BAD — section header comments
# === Helpers ===
def helper(): ...
```

## What TO do instead

Make the code self-documenting through clear names and structure:

```python
# GOOD — name says what the function does, no narration needed
def fetch_user(user_id: int) -> User:
    return db.get_user(user_id)

# GOOD — extract a well-named helper instead of commenting
def _is_within_business_hours(timestamp: datetime) -> bool:
    return 9 <= timestamp.hour < 17

if _is_within_business_hours(now):
    schedule_call()
```

## Allowed (these are NOT comments)

- **Docstrings** (`"""..."""` directly under a `def`/`class`/module) — these are documentation strings, not comments. Use them for public APIs when the name and signature aren't enough.
- **Shebang lines** (`#!/usr/bin/env python3`) — required for executable scripts.
- **Encoding/type-checker directives required by tooling** (e.g. `# type: ignore` only when explicitly authorized per `.claude/rules/agent-guidelines.md`).
- **License/copyright headers** when the project already uses them in similar files.

## When tempted to write a comment

1. **Rename** the variable/function/class so the name conveys the intent.
2. **Extract** a helper with a descriptive name.
3. **Restructure** the code so the flow is obvious.
4. **Delete** the comment — git history and the code already tell the story.

If after all of that the intent is still genuinely non-obvious (rare: a subtle algorithmic invariant, a workaround for an external bug, a non-obvious performance trade-off), put it in a **docstring** on the enclosing function — not an inline comment.

## When editing existing code

- Do not add new comments.
- If you touch a block that has narrative/transitional/TODO comments, remove them.
- Leave pre-existing docstrings, shebangs, and required directives alone.
