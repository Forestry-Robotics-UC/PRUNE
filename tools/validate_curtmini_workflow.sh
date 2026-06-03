#!/bin/bash
# Backward-compatible wrapper for the canonical validation workflow script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/validation/validate_curtmini_workflow.sh" "$@"
