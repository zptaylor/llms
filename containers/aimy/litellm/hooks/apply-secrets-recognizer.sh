#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENC_FILE="$SCRIPT_DIR/secrets-recognizer.json.enc"
JSON_FILE="$SCRIPT_DIR/secrets-recognizer.json"

if [[ ! -f "$ENC_FILE" ]]; then
  echo "Error: Encrypted file not found: $ENC_FILE"
  echo "Create it first with:"
  echo "  sops encrypt --age <AGE_PUBLIC_KEY> secrets-recognizer.json > $ENC_FILE"
  exit 1
fi

if ! command -v sops &> /dev/null; then
  echo "Error: 'sops' command not found. Install it (e.g., nixpkgs#sops) and retry."
  exit 1
fi

echo "Decrypting $ENC_FILE -> $JSON_FILE"
sops decrypt "$ENC_FILE" > "$JSON_FILE"

if ! python3 -c "import json; json.load(open('$JSON_FILE'))" &> /dev/null; then
  echo "Error: Decrypted file is not valid JSON!"
  exit 1
fi

echo "Success: $JSON_FILE updated with real secrets."
echo "Remember to run your rebuild command (e.g., sudo nixos-rebuild switch --flake .#ai)."
