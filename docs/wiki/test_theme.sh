#!/bin/bash

# Test Theme Functionality
echo "🧪 Testing Monokai Pro Theme..."

# Check if theme files exist
echo "📁 Checking theme files..."
if [ -f "tiddlers/monokai_pro_theme.tid" ]; then
    echo "✅ Monokai Pro theme found"
else
    echo "❌ Monokai Pro theme missing"
fi

if [ -f "tiddlers/theme_switcher.tid" ]; then
    echo "✅ Theme switcher found"
else
    echo "❌ Theme switcher missing"
fi

# Check if server is running
echo "🌐 Checking server status..."
if curl -s http://127.0.0.1:8080 > /dev/null; then
    echo "✅ Server is running"
else
    echo "❌ Server is not running"
fi

echo ""
echo "🎯 To test the theme:"
echo "1. Open: http://127.0.0.1:8080"
echo "2. Search for 'Theme Switcher'"
echo "3. Click the theme buttons"
echo "4. Check browser console for any errors"
echo ""
echo "🔧 If theme doesn't work:"
echo "- Open browser developer tools (F12)"
echo "- Check Console tab for JavaScript errors"
echo "- Check Network tab for missing files"
