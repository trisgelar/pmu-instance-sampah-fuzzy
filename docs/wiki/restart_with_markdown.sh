#!/bin/bash

# PMU Instance Sampah Fuzzy - Restart with Markdown Plugin

echo "🔄 Restarting TiddlyWiki server with markdown plugin..."

# Check if server is running and stop it
echo "🛑 Stopping current server..."
pkill -f "tiddlywiki.*--listen" 2>/dev/null || true

# Wait a moment for the server to stop
sleep 2

# Start the server with the new configuration
echo "🚀 Starting server with markdown plugin..."
cd /home/balaplumpat/Projects/pmu-instance-sampah-fuzzy
source run_with_node.sh tiddlywiki docs/wiki --listen

echo "✅ Server restarted with markdown plugin!"
echo "🌐 Open: http://127.0.0.1:8080"
echo "📝 Your headings should now look much better!"
