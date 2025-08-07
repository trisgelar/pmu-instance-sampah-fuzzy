#!/bin/bash

# PMU Instance Sampah Fuzzy - Wiki Setup Script

echo "🚀 Setting up TiddlyWiki for PMU Instance Sampah Fuzzy documentation..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    echo "   Visit: https://nodejs.org/"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

# Install TiddlyWiki globally
echo "📦 Installing TiddlyWiki..."
npm install -g tiddlywiki

# Create wiki directory
echo "📁 Creating wiki directory..."
mkdir -p mywiki

# Initialize TiddlyWiki
echo "🔧 Initializing TiddlyWiki..."
tiddlywiki mywiki --init

# Copy documentation files
echo "📋 Copying documentation files..."
if [ -d "docs/wiki/tiddlers" ]; then
    cp docs/wiki/tiddlers/*.tid mywiki/tiddlers/ 2>/dev/null || echo "No tiddler files found"
else
    echo "⚠️  No tiddlers directory found"
fi

# Copy configuration
if [ -f "docs/wiki/tiddlywiki.info" ]; then
    cp docs/wiki/tiddlywiki.info mywiki/
else
    echo "⚠️  No tiddlywiki.info found"
fi

# Copy images
if [ -d "docs/wiki/images" ]; then
    cp -r docs/wiki/images/* mywiki/files/ 2>/dev/null || mkdir -p mywiki/files && cp -r docs/wiki/images/* mywiki/files/
fi

echo "✅ Wiki setup complete!"
echo ""
echo "🎯 To start the wiki server:"
echo "   cd mywiki"
echo "   tiddlywiki --listen"
echo ""
echo "🌐 Then open your browser to: http://127.0.0.1:8080"
echo ""
echo "📝 To build a static version:"
echo "   tiddlywiki --build index"
echo ""
echo "📁 The static files will be in: mywiki/output/"
echo ""
echo "📚 Navigation guide: docs/wiki/NAVIGATION.md"
