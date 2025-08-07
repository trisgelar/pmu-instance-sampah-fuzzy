#!/bin/bash

# PMU Instance Sampah Fuzzy - Simple Wiki Start Script

echo "🚀 Starting TiddlyWiki server..."

# Check if TiddlyWiki is installed
if ! command -v tiddlywiki &> /dev/null; then
    echo "❌ TiddlyWiki not found. Please install it first:"
    echo "   npm install -g tiddlywiki"
    exit 1
fi

# Change to wiki directory
cd docs/wiki

echo "📁 Starting server from: $(pwd)"
echo "🌐 Server will be available at: http://127.0.0.1:8080"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Start TiddlyWiki server
cd /home/balaplumpat/Projects/pmu-instance-sampah-fuzzy
source run_with_node.sh tiddlywiki docs/wiki --listen
