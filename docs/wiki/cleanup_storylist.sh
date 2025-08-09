#!/bin/bash

# TiddlyWiki StoryList Cleanup Script
# This script removes temporary StoryList files that accumulate during development

echo "🧹 TiddlyWiki StoryList Cleanup"
echo "================================"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIDDLERS_DIR="$SCRIPT_DIR/tiddlers"

# Check if tiddlers directory exists
if [ ! -d "$TIDDLERS_DIR" ]; then
    echo "❌ Error: tiddlers directory not found at $TIDDLERS_DIR"
    exit 1
fi

echo "📁 Working in: $TIDDLERS_DIR"

# Count StoryList files before cleanup
BEFORE_COUNT=$(find "$TIDDLERS_DIR" -name "\$__StoryList_*.tid" 2>/dev/null | wc -l)
MAIN_STORYLIST="$TIDDLERS_DIR/\$__StoryList.tid"

echo "📊 Found $BEFORE_COUNT numbered StoryList files"

if [ $BEFORE_COUNT -eq 0 ]; then
    echo "✅ No numbered StoryList files to clean up!"
else
    echo "🗑️  Removing numbered StoryList files..."
    
    # Remove numbered StoryList files (keep the main one)
    find "$TIDDLERS_DIR" -name "\$__StoryList_*.tid" -delete 2>/dev/null
    
    # Count files after cleanup
    AFTER_COUNT=$(find "$TIDDLERS_DIR" -name "\$__StoryList_*.tid" 2>/dev/null | wc -l)
    REMOVED_COUNT=$((BEFORE_COUNT - AFTER_COUNT))
    
    echo "✅ Removed $REMOVED_COUNT StoryList files"
fi

# Check if main StoryList exists
if [ -f "$MAIN_STORYLIST" ]; then
    echo "📝 Main StoryList file preserved: \$__StoryList.tid"
else
    echo "ℹ️  No main StoryList file found (this is normal)"
fi

echo ""
echo "🎉 Cleanup complete!"
echo "💡 Tip: Run this script anytime StoryList files accumulate during development"
