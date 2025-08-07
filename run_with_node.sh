#!/bin/bash

# Wrapper script to properly use Node.js and npm in Cursor IDE terminal
# Similar to run_with_venv.sh for Python
PROJECT_ROOT="/home/balaplumpat/Projects/pmu-instance-sampah-fuzzy"

# Set Node.js paths directly since we know where they are
NODE_PATH="/home/balaplumpat/.nvm/versions/node/v22.18.0/bin"
export PATH="$NODE_PATH:$PATH"

echo "🔍 Using Node.js from: $NODE_PATH"
echo "✅ Node.js version: $(node --version)"
echo "✅ npm version: $(npm --version)"

# Check if nvm is available and load it
if [ -s "$HOME/.nvm/nvm.sh" ]; then
    echo "📦 Loading nvm..."
    source "$HOME/.nvm/nvm.sh"
    nvm use default 2>/dev/null || nvm use node 2>/dev/null
fi

# Verify Node.js and npm are available
echo "🔧 Checking Node.js installation..."
if command -v node &> /dev/null; then
    echo "✅ Node.js found: $(node --version)"
else
    echo "❌ Node.js not found in PATH"
    exit 1
fi

if command -v npm &> /dev/null; then
    echo "✅ npm found: $(npm --version)"
else
    echo "❌ npm not found in PATH"
    exit 1
fi

echo "🚀 Executing command: $@"
exec "$@"
