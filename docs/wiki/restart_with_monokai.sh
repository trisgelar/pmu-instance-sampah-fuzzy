#!/bin/bash

# PMU Instance Sampah Fuzzy - Restart with Monokai Pro Theme

echo "🎨 Restarting TiddlyWiki server with Monokai Pro theme..."

# Check if server is running and stop it
echo "🛑 Stopping current server..."
pkill -f "tiddlywiki.*--listen" 2>/dev/null || true

# Wait a moment for the server to stop
sleep 2

# Start the server with the new theme
echo "🚀 Starting server with Monokai Pro theme..."
cd /home/balaplumpat/Projects/pmu-instance-sampah-fuzzy
source run_with_node.sh tiddlywiki docs/wiki --listen

echo "✅ Server restarted with Monokai Pro theme!"
echo "🌐 Open: http://127.0.0.1:8080"
echo "🎨 To switch themes, open the 'Theme Switcher' tiddler"
echo "📝 Your documentation now has beautiful Monokai Pro colors!"
