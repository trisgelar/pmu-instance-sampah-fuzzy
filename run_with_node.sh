#!/bin/bash

# Enhanced Node.js wrapper script with TiddlyWiki management
# Handles TiddlyWiki installation, version checking, and updates
PROJECT_ROOT="/home/balaplumpat/Projects/pmu-instance-sampah-fuzzy"

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${2}${1}${NC}"
}

print_status "🚀 Node.js & TiddlyWiki Environment Manager" "$CYAN"
print_status "==========================================" "$CYAN"

# Set Node.js paths directly since we know where they are
NODE_PATH="/home/balaplumpat/.nvm/versions/node/v22.18.0/bin"
export PATH="$NODE_PATH:$PATH"

print_status "🔍 Using Node.js from: $NODE_PATH" "$BLUE"

# Check if nvm is available and load it
if [ -s "$HOME/.nvm/nvm.sh" ]; then
    print_status "📦 Loading nvm..." "$YELLOW"
    source "$HOME/.nvm/nvm.sh"
    nvm use default 2>/dev/null || nvm use node 2>/dev/null
fi

# Verify Node.js and npm are available
print_status "🔧 Checking Node.js installation..." "$BLUE"
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_status "✅ Node.js found: $NODE_VERSION" "$GREEN"
else
    print_status "❌ Node.js not found in PATH" "$RED"
    exit 1
fi

if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    print_status "✅ npm found: $NPM_VERSION" "$GREEN"
else
    print_status "❌ npm not found in PATH" "$RED"
    exit 1
fi

# TiddlyWiki Management
print_status "🎯 TiddlyWiki Management" "$PURPLE"
print_status "========================" "$PURPLE"

# Function to check TiddlyWiki installation
check_tiddlywiki() {
    if command -v tiddlywiki &> /dev/null; then
        TW_VERSION=$(tiddlywiki --version 2>/dev/null || echo "unknown")
        print_status "✅ TiddlyWiki found: $TW_VERSION" "$GREEN"
        return 0
    else
        print_status "❌ TiddlyWiki not found" "$RED"
        return 1
    fi
}

# Function to install TiddlyWiki globally
install_tiddlywiki() {
    print_status "📦 Installing TiddlyWiki globally..." "$YELLOW"
    
    if npm install -g tiddlywiki; then
        print_status "✅ TiddlyWiki installed successfully!" "$GREEN"
        # Verify installation
        if command -v tiddlywiki &> /dev/null; then
            TW_VERSION=$(tiddlywiki --version 2>/dev/null || echo "unknown")
            print_status "✅ Installed version: $TW_VERSION" "$GREEN"
        fi
    else
        print_status "❌ Failed to install TiddlyWiki" "$RED"
        print_status "💡 You may need to run with sudo or check npm permissions" "$YELLOW"
        return 1
    fi
}

# Function to update TiddlyWiki
update_tiddlywiki() {
    print_status "🔄 Updating TiddlyWiki to latest version..." "$YELLOW"
    
    if npm update -g tiddlywiki; then
        print_status "✅ TiddlyWiki updated successfully!" "$GREEN"
        NEW_VERSION=$(tiddlywiki --version 2>/dev/null || echo "unknown")
        print_status "✅ Current version: $NEW_VERSION" "$GREEN"
    else
        print_status "❌ Failed to update TiddlyWiki" "$RED"
        return 1
    fi
}

# Function to show TiddlyWiki info
show_tiddlywiki_info() {
    print_status "📊 TiddlyWiki Information:" "$CYAN"
    echo
    
    if command -v tiddlywiki &> /dev/null; then
        TW_VERSION=$(tiddlywiki --version 2>/dev/null || echo "unknown")
        TW_PATH=$(which tiddlywiki 2>/dev/null || echo "not found")
        
        echo "  📌 Version: $TW_VERSION"
        echo "  📁 Path: $TW_PATH"
        
        # Check for latest version
        print_status "🔍 Checking for latest version..." "$YELLOW"
        LATEST_VERSION=$(npm view tiddlywiki version 2>/dev/null || echo "unknown")
        echo "  🌐 Latest available: $LATEST_VERSION"
        
        # Compare versions
        if [ "$TW_VERSION" != "$LATEST_VERSION" ] && [ "$LATEST_VERSION" != "unknown" ]; then
            print_status "⚠️  Update available: $TW_VERSION → $LATEST_VERSION" "$YELLOW"
        else
            print_status "✅ You have the latest version!" "$GREEN"
        fi
    else
        print_status "❌ TiddlyWiki not installed" "$RED"
    fi
    echo
}

# Main TiddlyWiki logic
if ! check_tiddlywiki; then
    print_status "🤔 TiddlyWiki not found. Would you like to install it?" "$YELLOW"
    print_status "💡 Installing automatically..." "$BLUE"
    install_tiddlywiki
else
    show_tiddlywiki_info
fi

# Special commands handling
if [ $# -eq 0 ]; then
    print_status "🎯 Available Commands:" "$CYAN"
    echo "  • tiddlywiki --help        - Show TiddlyWiki help"
    echo "  • tiddlywiki --version     - Show TiddlyWiki version"
    echo "  • ./docs/wiki/start.sh     - Start TiddlyWiki server"
    echo ""
    print_status "💡 Usage: $0 <command>" "$YELLOW"
    print_status "Example: $0 tiddlywiki --version" "$YELLOW"
    exit 0
fi

# Handle special management commands
case "$1" in
    "install-tiddlywiki")
        install_tiddlywiki
        exit $?
        ;;
    "update-tiddlywiki")
        update_tiddlywiki
        exit $?
        ;;
    "tiddlywiki-info")
        show_tiddlywiki_info
        exit 0
        ;;
    "check-tiddlywiki")
        check_tiddlywiki
        show_tiddlywiki_info
        exit $?
        ;;
esac

print_status "🚀 Executing command: $@" "$GREEN"
print_status "========================================" "$GREEN"
exec "$@"
