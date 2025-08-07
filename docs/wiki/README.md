# PMU Instance Sampah Fuzzy - Documentation Wiki

## 🚀 Quick Start

### First Time Setup
```bash
./docs/wiki/setup.sh
```

### Start Wiki Server
```bash
./docs/wiki/start.sh
```

### Restart with Markdown Plugin (Better Formatting)
```bash
./docs/wiki/restart_with_markdown.sh
```

### Restart with Monokai Pro Theme (Beautiful Colors)
```bash
./docs/wiki/restart_with_monokai.sh
```

### Access Wiki
Open your browser to: http://127.0.0.1:8080

## 📁 Directory Structure

```
docs/wiki/
├── tiddlers/          # Your documentation files (.tid)
├── images/            # Images for the wiki
├── tiddlywiki.info    # TiddlyWiki configuration
├── start.sh           # Start the wiki server
├── setup.sh           # Setup the wiki
├── restart_with_markdown.sh  # Restart with markdown plugin
├── restart_with_monokai.sh   # Restart with Monokai Pro theme
├── monokai_pro_theme.tid     # Monokai Pro theme definition
├── theme_switcher.tid        # Theme switcher interface
└── README.md          # This file
```

## 📋 Available Commands

- `./docs/wiki/setup.sh` - Setup TiddlyWiki (first time only)
- `./docs/wiki/start.sh` - Start the wiki server
- `./docs/wiki/restart_with_markdown.sh` - Restart with markdown plugin
- `./docs/wiki/restart_with_monokai.sh` - Restart with Monokai Pro theme
- Press `Ctrl+C` to stop the server

## 📚 Your Documentation

All your documentation is stored in `docs/wiki/tiddlers/` as `.tid` files.
The wiki automatically loads these files when started.

## 🎨 Monokai Pro Theme

Based on the [Monokai Pro color scheme](https://monokai.pro/), this theme provides:

### 🌙 Dark Theme
- **Background**: Dark gray (`#2d2a2e`)
- **Text**: Yellow (`#fce566`)
- **Headings**: Green (`#a6e22e`)
- **Code**: Orange (`#fd971f`)
- **Links**: Light green (`#a8ff60`)

### ☀️ Light Theme
- **Background**: Light gray (`#fafafa`)
- **Text**: Dark gray (`#2d2a2e`)
- **Headings**: Purple (`#6f42c1`)
- **Code**: Red (`#d73a49`)
- **Links**: Dark green (`#718c00`)

### 🎯 Theme Features
- **Beautiful syntax highlighting** for code blocks
- **Professional aesthetics** optimized for developer workflow
- **Focus-oriented design** with non-distractive interface
- **Persistent theme selection** (remembers your choice)
- **Easy theme switching** via the Theme Switcher tiddler

### 🔄 How to Switch Themes
1. Open the wiki at http://127.0.0.1:8080
2. Search for "Theme Switcher" or navigate to it
3. Click the theme buttons to switch between Dark and Light
4. Your choice will be remembered for future visits

## 📝 Markdown Plugin

The markdown plugin is enabled for better formatting:
- **Better headings** - Proper heading styles
- **Code blocks** - Syntax highlighting
- **Lists** - Better list formatting
- **Links** - Improved link styling
- **Tables** - Better table formatting

## 🔧 Troubleshooting

If you see plugin warnings, run the setup script again:
```bash
./docs/wiki/setup.sh
```

To enable markdown formatting, restart with:
```bash
./docs/wiki/restart_with_markdown.sh
```

To enable Monokai Pro theme, restart with:
```bash
./docs/wiki/restart_with_monokai.sh
```
