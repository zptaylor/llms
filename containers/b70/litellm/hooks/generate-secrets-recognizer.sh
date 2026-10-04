#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <secrets-file>"
  exit 1
fi

SECRETS_FILE="$1"

if [[ ! -f "$SECRETS_FILE" ]]; then
  echo "Error: Secrets file '$SECRETS_FILE' not found."
  exit 1
fi

mapfile -t SECRETS < <(grep -v '^#' "$SECRETS_FILE" | grep -v '^$' | sed 's/[\"]/\\&/g')

if [[ ${#SECRETS[@]} -eq 0 ]]; then
  echo "[]"
else
  jq -n \
    --argjson deny_list "$(printf '%s\n' "${SECRETS[@]}" | jq -R . | jq -s .)" \
    '[{name: "Custom Secrets Deny List", supported_language: "en", deny_list: $deny_list, supported_entity: "CUSTOM_SECRET"}]'
fi
