#!/usr/bin/env python3
"""
Convert existing documentation to TiddlyWiki-style format.

This script helps convert your existing documentation files into a structured
wiki format that can be easily navigated and maintained.
"""

import os
import re
import shutil
from pathlib import Path

def create_wiki_structure():
    """Create the basic wiki directory structure."""
    wiki_dir = Path("docs/wiki")
    wiki_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (wiki_dir / "images").mkdir(exist_ok=True)
    (wiki_dir / "tiddlers").mkdir(exist_ok=True)
    
    print(f"✅ Created wiki directory structure in {wiki_dir}")

def convert_markdown_to_wiki_tiddler(md_file, output_dir):
    """Convert a markdown file to TiddlyWiki tiddler format."""
    if not md_file.exists():
        print(f"⚠️  File not found: {md_file}")
        return
    
    # Read the markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract title from first heading
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    title = title_match.group(1) if title_match else md_file.stem
    
    # Create tiddler filename
    tiddler_filename = f"{title.replace(' ', '_').lower()}.tid"
    tiddler_path = output_dir / tiddler_filename
    
    # Convert markdown to tiddler format
    tiddler_content = f"""title: {title}
type: text/vnd.tiddlywiki
tags: documentation

{content}
"""
    
    # Write tiddler file
    with open(tiddler_path, 'w', encoding='utf-8') as f:
        f.write(tiddler_content)
    
    print(f"✅ Converted {md_file.name} to {tiddler_filename}")

def create_tiddlywiki_config():
    """Create a basic TiddlyWiki configuration."""
    config_content = """{
    "description": "PMU Instance Sampah Fuzzy Documentation",
    "plugins": [
        "tiddlywiki/filesystem",
        "tiddlywiki/tiddlyweb"
    ],
    "themes": [
        "tiddlywiki/vanilla",
        "tiddlywiki/snowwhite"
    ],
    "tiddlers": [
        {
            "file": "tiddlers/overview.tid",
            "fields": {
                "title": "Overview",
                "type": "text/vnd.tiddlywiki"
            }
        },
        {
            "file": "tiddlers/installation.tid",
            "fields": {
                "title": "Installation",
                "type": "text/vnd.tiddlywiki"
            }
        },
        {
            "file": "tiddlers/quick_start.tid",
            "fields": {
                "title": "Quick Start",
                "type": "text/vnd.tiddlywiki"
            }
        },
        {
            "file": "tiddlers/pipeline_overview.tid",
            "fields": {
                "title": "Pipeline Overview",
                "type": "text/vnd.tiddlywiki"
            }
        },
        {
            "file": "tiddlers/yolov8_pipeline.tid",
            "fields": {
                "title": "YOLOv8 Pipeline",
                "type": "text/vnd.tiddlywiki"
            }
        },
        {
            "file": "tiddlers/yolov10_pipeline.tid",
            "fields": {
                "title": "YOLOv10 Pipeline",
                "type": "text/vnd.tiddlywiki"
            }
        },
        {
            "file": "tiddlers/yolov11_pipeline.tid",
            "fields": {
                "title": "YOLOv11 Pipeline",
                "type": "text/vnd.tiddlywiki"
            }
        },
        {
            "file": "tiddlers/master_pipeline.tid",
            "fields": {
                "title": "Master Pipeline",
                "type": "text/vnd.tiddlywiki"
            }
        },
        {
            "file": "tiddlers/common_issues.tid",
            "fields": {
                "title": "Common Issues",
                "type": "text/vnd.tiddlywiki"
            }
        },
        {
            "file": "tiddlers/onnx_rknn_issues.tid",
            "fields": {
                "title": "ONNX/RKNN Issues",
                "type": "text/vnd.tiddlywiki"
            }
        }
    ]
}"""
    
    config_path = Path("docs/wiki/tiddlywiki.info")
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"✅ Created TiddlyWiki configuration: {config_path}")

def create_setup_script():
    """Create a setup script for the wiki."""
    setup_script = """#!/bin/bash

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
cp docs/wiki/tiddlers/*.tid mywiki/tiddlers/ 2>/dev/null || echo "No tiddler files found"

# Copy configuration
if [ -f "docs/wiki/tiddlywiki.info" ]; then
    cp docs/wiki/tiddlywiki.info mywiki/
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
"""
    
    setup_path = Path("docs/wiki/setup_wiki.sh")
    with open(setup_path, 'w', encoding='utf-8') as f:
        f.write(setup_script)
    
    # Make it executable
    os.chmod(setup_path, 0o755)
    
    print(f"✅ Created setup script: {setup_path}")

def create_readme():
    """Create a comprehensive README for the wiki."""
    readme_content = """# PMU Instance Sampah Fuzzy - Documentation Wiki

## 📋 Overview

This directory contains a TiddlyWiki-style documentation structure for the PMU Instance Sampah Fuzzy project. The documentation is organized in a wiki format for easy navigation and reference.

## 🚀 Quick Start

### Option 1: Use the Simple HTML Wiki
1. Open `simple-wiki.html` in your browser
2. Navigate using the sidebar
3. All content is self-contained

### Option 2: Set up TiddlyWiki
1. Run the setup script:
   ```bash
   chmod +x setup_wiki.sh
   ./setup_wiki.sh
   ```

2. Start the wiki server:
   ```bash
   cd mywiki
   tiddlywiki --listen
   ```

3. Open your browser to: http://127.0.0.1:8080

### Option 3: Manual Setup
1. Install TiddlyWiki: `npm install -g tiddlywiki`
2. Create new wiki: `tiddlywiki mywiki --init`
3. Copy files from this directory to `mywiki/`
4. Run: `tiddlywiki mywiki --listen`

## 📁 File Structure

```
docs/wiki/
├── simple-wiki.html          # Simple HTML wiki (ready to use)
├── setup_wiki.sh             # Setup script for TiddlyWiki
├── convert_to_wiki.py        # Conversion script
├── README.md                 # This file
├── tiddlywiki.info           # TiddlyWiki configuration
├── tiddlers/                 # TiddlyWiki tiddler files
│   ├── overview.tid
│   ├── installation.tid
│   ├── quick_start.tid
│   ├── pipeline_overview.tid
│   ├── yolov8_pipeline.tid
│   ├── yolov10_pipeline.tid
│   ├── yolov11_pipeline.tid
│   ├── master_pipeline.tid
│   ├── common_issues.tid
│   └── onnx_rknn_issues.tid
└── images/                   # Images for the wiki
```

## 🎯 Features

### ✅ Easy Navigation
- Sidebar navigation
- Breadcrumb trails
- Search functionality

### ✅ Multiple Formats
- HTML for immediate use
- TiddlyWiki for advanced features
- Markdown for editing

### ✅ Responsive Design
- Works on desktop and mobile
- Clean, modern interface
- Accessible design

## 🔧 Customization

### Adding New Content
1. Create new `.tid` file in `tiddlers/`
2. Add to `tiddlywiki.info` configuration
3. Update navigation in `simple-wiki.html`

### Styling
- Edit CSS in `simple-wiki.html`
- Modify TiddlyWiki themes
- Add custom styles

### Content
- Edit tiddler files
- Add images to `images/`
- Update links as needed

## 🚀 Usage Examples

### View the Simple Wiki
```bash
# Open in browser
open docs/wiki/simple-wiki.html
```

### Set up TiddlyWiki
```bash
# Run setup script
./docs/wiki/setup_wiki.sh

# Start wiki server
cd mywiki
tiddlywiki --listen
```

### Build Static Version
```bash
cd mywiki
tiddlywiki --build index
# Static files will be in output/
```

## 📚 Related Documentation

- [Main README](../README.md)
- [Pipeline Scripts Guide](../PIPELINE_SCRIPTS_GUIDE.md)
- [ONNX/RKNN Troubleshooting](../ONNX_RKNN_TROUBLESHOOTING.md)
- [Pipeline Differences Guide](../PIPELINE_DIFFERENCES_GUIDE.md)

## 🎉 Benefits

### ✅ Better Organization
- Clear navigation structure
- Logical content grouping
- Easy to find information

### ✅ Improved Readability
- Clean, modern design
- Consistent formatting
- Better typography

### ✅ Enhanced User Experience
- Fast navigation
- Search functionality
- Mobile-friendly

### ✅ Easy Maintenance
- Modular content structure
- Version control friendly
- Easy to update

---

This wiki structure provides a clean, organized way to navigate and understand the PMU Instance Sampah Fuzzy project documentation! 🎉
"""
    
    readme_path = Path("docs/wiki/README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✅ Created comprehensive README: {readme_path}")

def main():
    """Main function to set up the wiki structure."""
    print("🚀 Setting up TiddlyWiki-style documentation structure...")
    
    # Create basic structure
    create_wiki_structure()
    
    # Create TiddlyWiki configuration
    create_tiddlywiki_config()
    
    # Create setup script
    create_setup_script()
    
    # Create README
    create_readme()
    
    print("\n✅ Wiki structure setup complete!")
    print("\n📋 Next steps:")
    print("1. Open docs/wiki/simple-wiki.html in your browser")
    print("2. Or run: ./docs/wiki/setup_wiki.sh")
    print("3. Edit content in docs/wiki/tiddlers/")
    print("4. Customize styling in simple-wiki.html")
    
    print("\n🎯 The wiki provides:")
    print("- Easy navigation with sidebar")
    print("- Clean, modern design")
    print("- Mobile-responsive layout")
    print("- Search functionality")
    print("- TiddlyWiki compatibility")

if __name__ == "__main__":
    main()
