#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$PROJECT_ROOT/bin"
WRAPPER_PATH="$BIN_DIR/solidcue"
PATH_EXPORT="export PATH=\"$BIN_DIR:\$PATH\""

mkdir -p "$BIN_DIR"

cat > "$WRAPPER_PATH" <<'WRAP'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
uv run --project "$PROJECT_ROOT" cli "$@"
WRAP

chmod +x "$WRAPPER_PATH"

append_if_missing() {
  local file="$1"
  local line="$2"

  touch "$file"
  if ! grep -F "$line" "$file" >/dev/null 2>&1; then
    printf "\n%s\n" "$line" >> "$file"
    echo "Updated $file"
  else
    echo "Already configured in $file"
  fi
}

append_if_missing "$HOME/.zshrc" "$PATH_EXPORT"
append_if_missing "$HOME/.zprofile" "$PATH_EXPORT"

echo
echo "Installed: $WRAPPER_PATH"
echo "Run this once in your current shell:"
echo "  source ~/.zshrc && source ~/.zprofile"
echo "Then verify:"
echo "  which solidcue"
echo "  solidcue --help"
