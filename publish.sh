#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
BRANCH="published"
PYTHON="/opt/homebrew/bin/python3"
LOG_FILE="$REPO_DIR/logs/publish_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$REPO_DIR/logs"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== publish.sh started at $(date) ==="

cd "$REPO_DIR"

# Ensure we're on the published branch
current=$(git branch --show-current)
if [ "$current" != "$BRANCH" ]; then
  echo "Switching to $BRANCH branch..."
  git checkout "$BRANCH"
fi

git pull --ff-only origin "$BRANCH" || true

# Ensure Ollama is running
if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Ollama is not running. Starting Ollama..."
  ollama serve &
  # Wait for Ollama to be ready
  for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
      echo "Ollama is ready."
      break
    fi
    if [ "$i" -eq 30 ]; then
      echo "ERROR: Ollama failed to start within 30 seconds." >&2
      exit 1
    fi
    sleep 1
  done
else
  echo "Ollama is already running."
fi

# Run main.py
echo "Running main.py..."
$PYTHON main.py

# Commit and push if there are changes in docs/
if git diff --quiet docs/ && git diff --cached --quiet docs/; then
  echo "No changes in docs/ — skipping commit."
else
  git add docs/
  git commit -m "Update articles $(date +%Y-%m-%d)"
  git push origin "$BRANCH"
  echo "Pushed to origin/$BRANCH"
fi

echo "=== publish.sh finished at $(date) ==="
