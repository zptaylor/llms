#!/usr/bin/env bash


set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "$HERE/.venv/bin/python" "$HERE/scripts/quality-gate.py"
