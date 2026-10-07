"""Generate / verify every copy of the local model-group tables.

The group names live in five places (see local_models.py's docstring). Four of
them are derived at import time; two are not:

  * config.yaml   router_settings.context_window_fallbacks  (YAML text)
  * the DB row    "router_settings"                         (JSON text)

This script owns those two. Run with no args to print the canonical YAML
block; --check to fail if config.yaml has drifted; --emit-sql to print the
UPDATE that pushes the same value into the DB row.

    python3 scripts/gen-model-group-config.py --check
    python3 scripts/gen-model-group-config.py --emit-sql | podman exec -i ...

``--emit-sql`` merges the LIVE row so it cannot silently drop keys this table
does not own (``model_group_alias``, and the non-local fallbacks). Set
``LITELLM_ROUTER_SETTINGS_ROW`` to JSON text or a path to a JSON file to supply
that row offline -- that is how the hermetic guard exercises the merge without a
container. ``--emit-sql-standalone`` skips the merge (fresh DB only).

Exit codes: 0 ok, 1 drift/mismatch, 2 usage error.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import local_models as m

CONFIG = ROOT / "config.yaml"

BEGIN = "  context_window_fallbacks:\n"
END = "\nguardrails:"


def render_block() -> str:
    """The exact YAML text that must appear in config.yaml."""
    out = [BEGIN.rstrip("\n")]
    for group, targets in m.CONTEXT_WINDOW_FALLBACKS.items():
        out.append(f"    - {group}:")
        for t in targets:
            out.append(f"        - {t}")
    return "\n".join(out) + "\n"


def current_block(text: str) -> str | None:
    i = text.find(BEGIN)
    if i < 0:
        return None
    j = text.find(END, i)
    if j < 0:
        return None
    return text[i:j]


def cmd_check() -> int:
    text = CONFIG.read_text()
    want, have = render_block(), current_block(text)
    if have is None:
        print("FAIL: no context_window_fallbacks block found in config.yaml")
        return 1
    if have != want:
        print("FAIL: config.yaml context_window_fallbacks has drifted")
        sys.stdout.writelines(
            difflib.unified_diff(
                have.splitlines(True), want.splitlines(True),
                "config.yaml(current)", "local_models.py(want)",
            )
        )
        return 1
    missing = [
        g for g in m.COMPRESS_GROUPS
        if g not in m.CONTEXT_WINDOW_FALLBACKS and g != "local"
    ]
    if missing:
        print(f"FAIL: primary groups with no escalation entry: {missing}")
        return 1
    print(f"ok: {len(m.CONTEXT_WINDOW_FALLBACKS)} groups in sync "
          f"({', '.join(m.CONTEXT_WINDOW_FALLBACKS)})")
    return 0


def _merge_fallbacks(existing: list) -> list:
    """Our local entries override same-key existing rows, keep the rest.

    The DB row is shared with non-local groups (free, nv-coding-best, ...).
    Replacing it wholesale would delete them, so merge key-by-key.
    """
    ours = {k: v for k, v in m.FALLBACKS.items()}
    out, seen = [], set()
    for item in existing:
        if not isinstance(item, dict) or len(item) != 1:
            out.append(item)
            continue
        (key, _), = item.items()
        if key in ours:
            out.append({key: ours[key]})
            seen.add(key)
        else:
            out.append(item)
    for key, val in ours.items():
        if key not in seen:
            out.append({key: val})
    return out


def db_value(existing: dict | None = None) -> str:
    """The exact JSON to store in LiteLLM_Config.router_settings.

    With ``existing`` (the current row) we merge and preserve every key we do
    not own -- notably ``model_group_alias`` and the non-local fallbacks.
    Without it we emit the standalone local-only payload.
    """
    existing = existing or {}
    payload = dict(existing)
    payload["fallbacks"] = _merge_fallbacks(existing.get("fallbacks") or [])
    payload["context_window_fallbacks"] = [
        {k: v} for k, v in m.CONTEXT_WINDOW_FALLBACKS.items()
    ]
    return json.dumps(payload, separators=(",", ":"))


def _read_db_row() -> dict | None:
    """Current router_settings row, or None if unavailable.

    ``LITELLM_ROUTER_SETTINGS_ROW`` overrides the live read: either the JSON
    text itself, or a path to a JSON file. This exists so the merge below can be
    tested WITHOUT a container/DB -- this repo's guard claims to be hermetic, and
    without an override it silently degrades to the local-only payload and looks
    like a data-loss regression. Real runs leave it unset and read the DB.
    """
    override = os.environ.get("LITELLM_ROUTER_SETTINGS_ROW")
    if override:
        try:
            text = override.lstrip()
            if not text.startswith("{"):
                text = Path(override).read_text()
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError("override is not a JSON object")
            print("note: using LITELLM_ROUTER_SETTINGS_ROW; the live DB row was "
                  "NOT read", file=sys.stderr)
            return row
        except (OSError, ValueError) as exc:
            print(f"warning: LITELLM_ROUTER_SETTINGS_ROW unusable ({exc}); "
                  "falling back to the live read", file=sys.stderr)
            return None

    import shutil
    import subprocess

    if not shutil.which("podman"):
        return None
    try:
        cp = subprocess.run(
            ["podman", "exec", "litellm-db-ai-host", "psql", "-U", "litellm",
             "-d", "litellm", "-tAc",
             'SELECT param_value FROM "LiteLLM_Config" '
             "WHERE param_name='router_settings'"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if cp.returncode != 0 or not cp.stdout.strip():
        return None
    try:
        return json.loads(cp.stdout.strip())
    except json.JSONDecodeError:
        return None


def cmd_emit_sql() -> int:
    existing = _read_db_row()
    if existing is None:
        print(
            "warning: could not read the live row; emitting a LOCAL-ONLY "
            "payload. Applying it would DROP model_group_alias and the "
            "non-local fallbacks. Apply it only on a fresh DB.",
            file=sys.stderr,
        )
    else:
        kept = sorted(set(existing) - {"fallbacks", "context_window_fallbacks"})
        if kept:
            print(f"merging: preserving {', '.join(kept)}", file=sys.stderr)
    val = db_value(existing).replace("'", "''")
    print(
        'UPDATE "LiteLLM_Config" SET param_value = \'{}\'::jsonb '
        "WHERE param_name = 'router_settings';".format(val)
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="verify config.yaml")
    g.add_argument("--emit-yaml", action="store_true", help="print the YAML block")
    g.add_argument("--emit-sql", action="store_true", help="print the DB UPDATE")
    g.add_argument(
        "--emit-sql-standalone", action="store_true",
        help="print the DB UPDATE without merging the live row",
    )
    a = ap.parse_args()
    if a.check:
        return cmd_check()
    if a.emit_sql:
        return cmd_emit_sql()
    if a.emit_sql_standalone:
        print(
            'UPDATE "LiteLLM_Config" SET param_value = \'{}\'::jsonb '
            "WHERE param_name = 'router_settings';".format(
                db_value().replace("'", "''")
            )
        )
        return 0
    print(render_block(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
