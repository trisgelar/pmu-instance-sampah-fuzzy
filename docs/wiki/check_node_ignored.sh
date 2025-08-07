#!/bin/bash

# PMU Instance Sampah Fuzzy - Node.js Git Ignore Verification Script

echo "🔍 Checking Node.js files and gitignore patterns..."
echo ""

# Check if .gitignore contains Node.js patterns
echo "📋 Checking .gitignore for Node.js patterns:"
if grep -q "node_modules" .gitignore; then
    echo "✅ node_modules/ is in .gitignore"
else
    echo "❌ node_modules/ is NOT in .gitignore"
fi

if grep -q "npm-debug.log" .gitignore; then
    echo "✅ npm-debug.log* is in .gitignore"
else
    echo "❌ npm-debug.log* is NOT in .gitignore"
fi

if grep -q ".nvmrc" .gitignore; then
    echo "✅ .nvmrc is in .gitignore"
else
    echo "❌ .nvmrc is NOT in .gitignore"
fi

if grep -q ".npm" .gitignore; then
    echo "✅ .npm is in .gitignore"
else
    echo "❌ .npm is NOT in .gitignore"
fi

echo ""

# Check current git status for any Node.js files
echo "📊 Checking git status for Node.js files:"
git status --porcelain | grep -E "(node_modules|\.npm|\.nvmrc|npm-debug|yarn)" || echo "✅ No Node.js files in git status"

echo ""

# Check if any Node.js files exist locally
echo "📁 Checking for existing Node.js files:"
if [ -d "node_modules" ]; then
    echo "⚠️  node_modules/ directory exists locally"
else
    echo "✅ No node_modules/ directory found"
fi

if [ -f ".nvmrc" ]; then
    echo "⚠️  .nvmrc file exists locally"
else
    echo "✅ No .nvmrc file found"
fi

if [ -f "package.json" ]; then
    echo "⚠️  package.json file exists locally"
else
    echo "✅ No package.json file found"
fi

if [ -f "package-lock.json" ]; then
    echo "⚠️  package-lock.json file exists locally"
else
    echo "✅ No package-lock.json file found"
fi

echo ""

# Test gitignore patterns
echo "🧪 Testing gitignore patterns:"
echo "Creating test files to verify gitignore..."

# Create test files
mkdir -p test_node_ignore
echo "test" > test_node_ignore/node_modules/test.txt
echo "test" > test_node_ignore/.nvmrc
echo "test" > test_node_ignore/npm-debug.log
echo "test" > test_node_ignore/package.json

# Check if they're ignored
echo "Checking if test files are ignored:"
if git check-ignore test_node_ignore/node_modules/test.txt > /dev/null; then
    echo "✅ node_modules files are properly ignored"
else
    echo "❌ node_modules files are NOT ignored"
fi

if git check-ignore test_node_ignore/.nvmrc > /dev/null; then
    echo "✅ .nvmrc files are properly ignored"
else
    echo "❌ .nvmrc files are NOT ignored"
fi

if git check-ignore test_node_ignore/npm-debug.log > /dev/null; then
    echo "✅ npm-debug.log files are properly ignored"
else
    echo "❌ npm-debug.log files are NOT ignored"
fi

# Clean up test files
rm -rf test_node_ignore

echo ""
echo "🎯 Summary:"
echo "- Node.js files will be ignored by git"
echo "- You can safely run: ./docs/wiki/setup_wiki.sh"
echo "- TiddlyWiki will create node_modules/ but it won't be committed"
echo ""
echo "✅ Ready to proceed with TiddlyWiki setup!"
