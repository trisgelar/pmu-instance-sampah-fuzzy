#!/bin/bash

# PMU Instance Sampah Fuzzy - Simple Wiki Setup Script

echo "🚀 Setting up TiddlyWiki..."

# Check if we're in the right directory
if [ ! -d "docs/wiki" ]; then
    echo "❌ Please run this script from the project root"
    exit 1
fi

# Go to wiki directory
cd docs/wiki

echo "📁 Working in: $(pwd)"

# Install TiddlyWiki globally if not already installed
echo "📦 Checking TiddlyWiki installation..."
if ! command -v tiddlywiki &> /dev/null; then
    echo "📦 Installing TiddlyWiki globally..."
    cd /home/balaplumpat/Projects/pmu-instance-sampah-fuzzy
    source run_with_node.sh npm install -g tiddlywiki
    cd docs/wiki
fi

# Initialize TiddlyWiki if tiddlywiki.info doesn't exist
if [ ! -f "tiddlywiki.info" ]; then
    echo "🔧 Initializing TiddlyWiki..."
    cd /home/balaplumpat/Projects/pmu-instance-sampah-fuzzy
    mkdir -p temp_wiki
    source run_with_node.sh tiddlywiki temp_wiki --init server
    cp temp_wiki/tiddlywiki.info docs/wiki/
    rm -rf temp_wiki
    cd docs/wiki
fi

echo "✅ Setup complete!"
echo ""
echo "🎯 To start the wiki:"
echo "   ./docs/wiki/start.sh"
echo ""
echo "🌐 Then open: http://127.0.0.1:8080"
