#!/bin/bash

# PMU Instance Sampah Fuzzy - Fix Tiddler Types for Markdown

echo "🔧 Fixing tiddler types for markdown formatting..."

# Check if we're in the right directory
if [ ! -d "docs/wiki" ]; then
    echo "❌ Please run this script from the project root"
    exit 1
fi

# Go to wiki directory
cd docs/wiki

echo "📁 Working in: $(pwd)"

# Count total tiddler files
TOTAL_FILES=$(find tiddlers -name "*.tid" | wc -l)
echo "📋 Found $TOTAL_FILES tiddler files"

# Fix each tiddler file
FIXED_COUNT=0
for file in tiddlers/*.tid; do
    if [ -f "$file" ]; then
        echo "🔧 Fixing: $file"
        
        # Create a temporary file
        temp_file=$(mktemp)
        
        # Process the file: change type from text/vnd.tiddlywiki to text/markdown
        sed 's/type: text\/vnd\.tiddlywiki/type: text\/markdown/' "$file" > "$temp_file"
        
        # Move the fixed file back
        mv "$temp_file" "$file"
        
        FIXED_COUNT=$((FIXED_COUNT + 1))
    fi
done

echo "✅ Fixed $FIXED_COUNT tiddler files"

# Update tiddlywiki.info to use text/markdown
echo "📝 Updating tiddlywiki.info..."
cat > tiddlywiki.info << 'EOF'
{
    "description": "Basic client-server edition",
    "plugins": [
        "tiddlywiki/tiddlyweb",
        "tiddlywiki/filesystem",
        "tiddlywiki/highlight",
        "tiddlywiki/markdown"
    ],
    "themes": [
        "tiddlywiki/vanilla",
        "tiddlywiki/snowwhite"
    ],
    "tiddlers": [
        {
            "file": "tiddlers/pipeline_scripts_guide.tid",
            "fields": {
                "title": "Pipeline Scripts Guide",
                "type": "text/markdown",
                "tags": "pipeline documentation"
            }
        },
        {
            "file": "tiddlers/pipeline_differences_and_folder_issues_guide.tid",
            "fields": {
                "title": "Pipeline Differences Guide",
                "type": "text/markdown",
                "tags": "pipeline documentation"
            }
        },
        {
            "file": "tiddlers/pipeline_quick_reference.tid",
            "fields": {
                "title": "Pipeline Quick Reference",
                "type": "text/markdown",
                "tags": "pipeline documentation"
            }
        },
        {
            "file": "tiddlers/deployment_guide.tid",
            "fields": {
                "title": "Deployment Guide",
                "type": "text/markdown",
                "tags": "deployment documentation"
            }
        },
        {
            "file": "tiddlers/onnx_and_rknn_troubleshooting_guide.tid",
            "fields": {
                "title": "ONNX/RKNN Troubleshooting Guide",
                "type": "text/markdown",
                "tags": "troubleshooting onnx rknn"
            }
        },
        {
            "file": "tiddlers/comprehensive_testing_guide.tid",
            "fields": {
                "title": "Comprehensive Testing Guide",
                "type": "text/markdown",
                "tags": "testing documentation"
            }
        },
        {
            "file": "tiddlers/api_reference.tid",
            "fields": {
                "title": "API Reference",
                "type": "text/markdown",
                "tags": "api documentation"
            }
        }
    ],
    "build": {
        "index": [
            "--render",
            "$:/plugins/tiddlywiki/tiddlyweb/save/offline",
            "index.html",
            "text/plain"
        ],
        "static": [
            "--render",
            "$:/core/templates/static.template.html",
            "static.html",
            "text/plain",
            "--render",
            "$:/core/templates/alltiddlers.template.html",
            "alltiddlers.html",
            "text/plain",
            "--render",
            "[!is[system]]",
            "[encodeuricomponent[]addprefix[static/]addsuffix[.html]]",
            "text/plain",
            "$:/core/templates/static.tiddler.html",
            "--render",
            "$:/core/templates/static.template.css",
            "static/static.css",
            "text/plain"
        ]
    }
}
EOF

echo "✅ Tiddler types fixed for markdown!"
echo ""
echo "🎯 Next steps:"
echo "1. Restart the server: ./docs/wiki/restart_with_markdown.sh"
echo "2. Open: http://127.0.0.1:8080"
echo "3. Your headings should now look beautiful!"
echo ""
echo "📋 All tiddler files now use text/markdown type"
