#!/bin/bash

# Quick TiddlyWiki cleanup script - can be run from project root
# Alias for the main cleanup script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEANUP_SCRIPT="$SCRIPT_DIR/docs/wiki/cleanup_storylist.sh"

if [ -f "$CLEANUP_SCRIPT" ]; then
    echo "🚀 Running TiddlyWiki cleanup..."
    "$CLEANUP_SCRIPT"
else
    echo "❌ Error: Cleanup script not found at $CLEANUP_SCRIPT"
    exit 1
fi
