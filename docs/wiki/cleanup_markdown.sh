#!/bin/bash

# PMU Instance Sampah Fuzzy - Markdown Cleanup Script

echo "🧹 Markdown Cleanup Options"
echo ""
echo "This script helps you manage the original .md files after converting to wiki format."
echo ""
echo "Options:"
echo "1. Move to archive folder (recommended)"
echo "2. Move to backup folder"
echo "3. Delete original files (dangerous!)"
echo "4. Just list what would be affected"
echo ""

read -p "Choose option (1-4): " choice

case $choice in
    1)
        echo "📁 Creating archive folder..."
        mkdir -p docs/archive
        mkdir -p docs/cursor/archive
        
        echo "📋 Moving docs/*.md to docs/archive/"
        mv docs/*.md docs/archive/ 2>/dev/null || echo "No .md files in docs/"
        
        echo "📋 Moving docs/cursor/*.md to docs/cursor/archive/"
        mv docs/cursor/*.md docs/cursor/archive/ 2>/dev/null || echo "No .md files in cursor/"
        
        echo "✅ Original markdown files moved to archive folders"
        echo "📁 Archive locations:"
        echo "   - docs/archive/"
        echo "   - docs/cursor/archive/"
        ;;
    2)
        echo "📁 Creating backup folder..."
        mkdir -p docs/backup
        mkdir -p docs/cursor/backup
        
        echo "📋 Copying docs/*.md to docs/backup/"
        cp docs/*.md docs/backup/ 2>/dev/null || echo "No .md files in docs/"
        
        echo "📋 Copying docs/cursor/*.md to docs/cursor/backup/"
        cp docs/cursor/*.md docs/cursor/backup/ 2>/dev/null || echo "No .md files in cursor/"
        
        echo "✅ Original markdown files backed up"
        echo "📁 Backup locations:"
        echo "   - docs/backup/"
        echo "   - docs/cursor/backup/"
        ;;
    3)
        echo "⚠️  WARNING: This will DELETE original markdown files!"
        read -p "Are you sure? Type 'yes' to confirm: " confirm
        if [ "$confirm" = "yes" ]; then
            echo "🗑️  Deleting docs/*.md..."
            rm docs/*.md 2>/dev/null || echo "No .md files in docs/"
            
            echo "🗑️  Deleting docs/cursor/*.md..."
            rm docs/cursor/*.md 2>/dev/null || echo "No .md files in cursor/"
            
            echo "✅ Original markdown files deleted"
        else
            echo "❌ Operation cancelled"
        fi
        ;;
    4)
        echo "📋 Files that would be affected:"
        echo ""
        echo "docs/*.md:"
        ls docs/*.md 2>/dev/null || echo "No .md files found"
        echo ""
        echo "docs/cursor/*.md:"
        ls docs/cursor/*.md 2>/dev/null || echo "No .md files found"
        ;;
    *)
        echo "❌ Invalid option"
        ;;
esac

echo ""
echo "🎯 Next steps:"
echo "1. Use the wiki: open docs/wiki/tiddler_viewer.html"
echo "2. Set up TiddlyWiki: ./docs/wiki/setup_wiki.sh"
echo "3. Edit tiddler files in docs/wiki/tiddlers/"
