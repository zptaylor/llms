# Presidio Ad-Hoc Recognizer for Custom Secrets

This directory contains the Presidio ad‑hoc recognizer used by the LiteLLM proxy to mask or block arbitrary secrets (passwords, API tokens, etc.) that are not covered by the built‑in patterns (AWS keys, GitHub tokens, etc.).

## Files

- `secrets-recognizer.json`  
  Tracked in git. Contains **harmless placeholder** deny‑list entries (e.g., `"REPLACE_WITH_YOUR_SECRET_1"`).  
  This file is mounted read‑only into the LiteLLM container and guarantees the container always starts with a valid JSON recognizer (so the proxy never crashes on missing file).

- `secrets-recognizer.json.template`  
  Tracked in git. Same as above; kept as a backup/template.

- `secrets-recognizer.json.enc`  
  **NOT tracked** (gitignored). SOPS‑encrypted file that holds your real deny‑list strings.

- `apply-secrets-recognizer.sh`  
  Script that decrypts `.enc` over the tracked `.json` file, giving you a local copy with real secrets.

- `generate-secrets-recognizer.sh`  
  Helper to build the recognizer JSON from a line‑by‑line `secrets.txt` file (one secret per line).

## Workflow — Adding or Updating Your Secret Deny‑List

1. **Prepare your secret list**  
   Create a plaintext file `secrets.txt` (one secret per line, `#` for comments).  
   Example:
   ```
   # my‑secrets.txt
   CHANGE_ME
   <GITHUB_TOKEN>
   sk_redacted
   ```

2. **(Optional) Generate the recognizer JSON**  
   ```bash
   ./generate-secrets-recognizer.sh secrets.txt > secrets-recognizer.json
   ```
   This produces a JSON file with a single `DenyListRecognizer` entity named `CUSTOM_SECRET`.

3. **Encrypt with SOPS** (using your age public key)  
   ```bash
   sops encrypt --age <AGE_PUBLIC_KEY> secrets-recognizer.json \
     > secrets-recognizer.json.enc
   ```
   - Replace `<AGE_PUBLIC_KEY>` with your age public key (e.g., `age1xxx...`).  
   - If you don’t have one yet, see the project’s `.sops.yaml` (gitignored) for the cluster keys (`ai-host`, `HOST-E`, `HOST-D`, `HOST-Z`).  
   - Commit `secrets-recognizer.json.enc`? **NO** — it is gitignored; keep it local or in your password store.

4. **Decrypt locally for use**  
   ```bash
   ./apply-secrets-recognizer.sh
   ```
   This overwrites the tracked `secrets-recognizer.json` with your real secrets.  
   Because the file is tracked, you should tell git to ignore further local changes:
   ```bash
   git update-index --skip-worktree containers/ai-host/litellm/hooks/secrets-recognizer.json
   ```
   (Now your local real‑secret edits won’t appear in `git status` and won’t be accidentally committed.)

5. **Rebuild and redeploy**  
   ```bash
   sudo nixos-rebuild switch --flake .#ai
   ```
   (or your target host). The LiteLLM proxy will restart, mount the updated recognizer, and begin masking/matching your deny‑list.

## How It Works in the Proxy

- The `presidio-phi` guardrail in `config.yaml` includes:
  ```yaml
    presidio_ad_hoc_recognizers: "./hooks/secrets-recognizer.json"
    pii_entities_config:
      # ... existing ...
      CUSTOM_SECRET: "MASK"   # or "BLOCK"
  ```
- On every request (pre‑call), the presidio analyzer:
  1. Runs its standard NER (for SSN, email, etc.).
  2. Loads the ad‑hoc recognizer from the mounted JSON.
  3. If any of your deny‑list strings appear, they are tagged as `CUSTOM_SECRET`.
  4. Per `pii_entities_config`, the tag is either:
     - **MASK** → replaced with `<CUSTOM_SECRET_1>` before the LLM sees it; after the LLM call, the original value is re‑hydrated (and wrapped with `🛡️` by the custom callback).
     - **BLOCK** → the request is rejected with a 4xx error before reaching the LLM.

## Security Notes

- **Never commit real secrets**. The only tracked file (`secrets-recognizer.json`) contains harmless placeholders that will not match real traffic.
- The SOPS‑encrypted file (`secrets-recognizer.json.enc`) may be committed if you wish, but it is safer to keep it off‑grid (e.g., in a password manager) and only decrypt locally when needed.
- If you lose the age private key needed to decrypt, you can always regenerate the encrypted file from your plaintext `secrets.txt`.
- The DenyListRecognizer does literal, case‑sensitive string matching. For pattern‑based matching (e.g., “any 32‑hex token”), use a `PatternRecognizer` with a regex instead (same JSON format, just change `deny_list` to `patterns`).

---
**Tip**: To rotate secrets, repeat steps 1‑4; the proxy will pick up the new values on the next restart.