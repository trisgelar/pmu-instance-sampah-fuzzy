#!/usr/bin/env python3
"""
Convert ALL markdown files in docs/ and docs/cursor/ to TiddlyWiki tiddler format.

This script converts all existing .md files to TiddlyWiki tiddler files for a modular wiki structure.
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime

def extract_title_from_markdown(content):
    """Extract title from markdown content."""
    # Look for first heading
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if title_match:
        return title_match.group(1).strip()
    
    # Look for second level heading if no first level
    title_match = re.search(r'^##\s+(.+)$', content, re.MULTILINE)
    if title_match:
        return title_match.group(1).strip()
    
    return None

def extract_tags_from_content(content, file_path):
    """Extract potential tags from content."""
    tags = []
    
    # Look for common keywords
    keywords = [
        'pipeline', 'yolo', 'training', 'inference', 'onnx', 'rknn',
        'troubleshooting', 'installation', 'deployment', 'testing',
        'configuration', 'documentation', 'guide', 'reference',
        'git', 'cuda', 'python', 'cursor', 'unit', 'structure'
    ]
    
    content_lower = content.lower()
    for keyword in keywords:
        if keyword in content_lower:
            tags.append(keyword)
    
    # Add category tag based on file path
    if 'cursor' in str(file_path):
        tags.append('cursor')
    else:
        tags.append('docs')
    
    return tags

def convert_markdown_to_tiddler(md_file, output_dir):
    """Convert a markdown file to TiddlyWiki tiddler format."""
    if not md_file.exists():
        print(f"⚠️  File not found: {md_file}")
        return None
    
    # Read the markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract title
    title = extract_title_from_markdown(content)
    if not title:
        title = md_file.stem.replace('_', ' ').title()
    
    # Extract tags
    tags = extract_tags_from_content(content, md_file)
    
    # Create tiddler filename
    tiddler_filename = f"{title.replace(' ', '_').lower()}.tid"
    tiddler_path = output_dir / tiddler_filename
    
    # Convert markdown to tiddler format
    tiddler_content = f"""title: {title}
type: text/vnd.tiddlywiki
tags: {" ".join(tags)} documentation
created: {datetime.now().strftime("%Y%m%d%H%M%S")}
modified: {datetime.now().strftime("%Y%m%d%H%M%S")}
source: {md_file}

{content}
"""
    
    # Write tiddler file
    with open(tiddler_path, 'w', encoding='utf-8') as f:
        f.write(tiddler_content)
    
    print(f"✅ Converted {md_file.name} to {tiddler_filename}")
    return tiddler_filename

def create_tiddlywiki_config(tiddler_files):
    """Create TiddlyWiki configuration with all tiddler files."""
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
    "tiddlers": ["""
    
    for tiddler_file in tiddler_files:
        config_content += f"""
        {{
            "file": "tiddlers/{tiddler_file}",
            "fields": {{
                "title": "{tiddler_file.replace('.tid', '').replace('_', ' ').title()}",
                "type": "text/vnd.tiddlywiki"
            }}
        }},"""
    
    config_content += """
    ]
}"""
    
    config_path = Path("docs/wiki/tiddlywiki.info")
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"✅ Created TiddlyWiki configuration: {config_path}")

def create_wiki_navigation(tiddler_files):
    """Create a navigation structure for the wiki."""
    navigation_content = """# PMU Instance Sampah Fuzzy - Complete Documentation Wiki

## 📋 Overview

This wiki contains ALL documentation for the PMU Instance Sampah Fuzzy project, converted from markdown files to TiddlyWiki format for better organization and navigation.

## 🚀 Quick Start

### Option 1: Use TiddlyWiki
1. Install TiddlyWiki: `npm install -g tiddlywiki`
2. Create new wiki: `tiddlywiki mywiki --init`
3. Copy files: `cp docs/wiki/tiddlers/*.tid mywiki/tiddlers/`
4. Copy config: `cp docs/wiki/tiddlywiki.info mywiki/`
5. Start server: `cd mywiki && tiddlywiki --listen`
6. Open: http://127.0.0.1:8080

### Option 2: Use Setup Script
```bash
chmod +x docs/wiki/setup_wiki.sh
./docs/wiki/setup_wiki.sh
```

## 📁 Documentation Structure

### Getting Started
- [Overview](overview) - Project overview and introduction
- [Installation](installation) - Setup and installation guide
- [Quick Start](quick_start) - Quick start guide

### Pipeline Documentation
- [Pipeline Scripts Guide](pipeline_scripts_guide) - Comprehensive pipeline documentation
- [Pipeline Differences Guide](pipeline_differences_guide) - Differences between pipeline types
- [Pipeline Quick Reference](pipeline_quick_reference) - Quick reference for pipelines

### Configuration & Setup
- [Environment Setup](environment_setup) - Environment configuration
- [Model Configuration](model_configuration) - Model settings and parameters
- [Dataset Configuration](dataset_configuration) - Dataset setup and management

### Troubleshooting
- [Common Issues](common_issues) - Frequently encountered problems
- [ONNX/RKNN Issues](onnx_rknn_troubleshooting) - ONNX and RKNN specific issues
- [Pipeline Issues](pipeline_issues) - Pipeline-related problems

### Advanced Topics
- [Testing](testing_guide) - Testing procedures and guidelines
- [Deployment](deployment_guide) - Deployment instructions
- [API Reference](api_reference) - API documentation

### Development & Maintenance
- [Code Structure Memory](code_structure_memory) - Code organization notes
- [Code Restructure Summary](code_restructure_summary) - Restructuring documentation
- [Naming Convention Reorganization](naming_convention_reorganization) - Naming conventions
- [Root Folder Cleanup Summary](root_folder_cleanup_summary) - Cleanup documentation

### Project History
- [Git Push Summary](git_push_summary) - Git operations summary
- [Import Warnings Fix Summary](import_warnings_fix_summary) - Import fixes
- [Compressed Folder Organization](compressed_folder_organization) - File organization
- [Gitignore Compressed Folder](gitignore_compressed_folder) - Gitignore updates

### External References
- [COCO Structure Compliance](coco_structure_compliance) - COCO format compliance
- [Colab Execution Guide](colab_execution_guide) - Google Colab usage
- [Dataset Fix Integration](dataset_fix_integration) - Dataset fixes

### Cursor IDE Documentation
- [Enhanced Training Guide](enhanced_training_guide) - Enhanced training documentation
- [Git Conflict Resolution](git_conflict_resolution) - Git conflict resolution guide
- [Git Quick Reference](git_quick_reference) - Git quick reference
- [Python 3.11 Compatibility](python_311_compatibility) - Python 3.11 compatibility
- [Setup Structure Improvements](setup_structure_improvements) - Setup improvements
- [Structural Changes Case Study](structural_changes_case_study) - Structural changes
- [Test Structure Improvements](test_structure_improvements) - Test structure improvements
- [Unit Testing Improvements](unit_testing_improvements) - Unit testing documentation
- [CUDA Version Management](cuda_version_management) - CUDA management

## 🎯 Features

### ✅ Modular Structure
- Each document is a separate tiddler
- Easy to navigate and search
- Version control friendly

### ✅ TiddlyWiki Benefits
- Advanced search functionality
- Tag-based organization
- Cross-references between documents
- Multiple themes available

### ✅ Easy Maintenance
- Edit individual tiddler files
- Add new documents easily
- Update existing content

## 🔧 Customization

### Adding New Documents
1. Create new `.tid` file in `docs/wiki/tiddlers/`
2. Add to `tiddlywiki.info` configuration
3. Update this navigation

### Styling
- Modify TiddlyWiki themes
- Add custom CSS
- Change colors and layout

### Content
- Edit tiddler files directly
- Add images to `docs/wiki/images/`
- Update links and references

## 📚 Related Resources

- [Main README](../README.md)
- [Pipeline Scripts](../PIPELINE_SCRIPTS_GUIDE.md)
- [Testing Guide](../TESTING_GUIDE.md)
- [Deployment Guide](../DEPLOYMENT_GUIDE.md)

---

This wiki provides a comprehensive, organized way to navigate all project documentation! 🎉
"""
    
    nav_path = Path("docs/wiki/NAVIGATION.md")
    with open(nav_path, 'w', encoding='utf-8') as f:
        f.write(navigation_content)
    
    print(f"✅ Created navigation guide: {nav_path}")

def create_cleanup_script():
    """Create a script to optionally move or archive original markdown files."""
    cleanup_script = """#!/bin/bash

# PMU Instance Sampah Fuzzy - Markdown Cleanup Script

echo "🧹 Markdown Cleanup Options"
echo ""
echo "This script helps you manage the original .md files after converting to wiki format."
echo ""
echo "Options:"
echo "1. Move to archive folder (recommended)"
echo "2. Move to backup folder"
echo "3. Delete original files (dangerous!)"
echo "4. Just list what would be affected"
echo ""

read -p "Choose option (1-4): " choice

case $choice in
    1)
        echo "📁 Creating archive folder..."
        mkdir -p docs/archive
        mkdir -p docs/cursor/archive
        
        echo "📋 Moving docs/*.md to docs/archive/"
        mv docs/*.md docs/archive/ 2>/dev/null || echo "No .md files in docs/"
        
        echo "📋 Moving docs/cursor/*.md to docs/cursor/archive/"
        mv docs/cursor/*.md docs/cursor/archive/ 2>/dev/null || echo "No .md files in cursor/"
        
        echo "✅ Original markdown files moved to archive folders"
        echo "📁 Archive locations:"
        echo "   - docs/archive/"
        echo "   - docs/cursor/archive/"
        ;;
    2)
        echo "📁 Creating backup folder..."
        mkdir -p docs/backup
        mkdir -p docs/cursor/backup
        
        echo "📋 Copying docs/*.md to docs/backup/"
        cp docs/*.md docs/backup/ 2>/dev/null || echo "No .md files in docs/"
        
        echo "📋 Copying docs/cursor/*.md to docs/cursor/backup/"
        cp docs/cursor/*.md docs/cursor/backup/ 2>/dev/null || echo "No .md files in cursor/"
        
        echo "✅ Original markdown files backed up"
        echo "📁 Backup locations:"
        echo "   - docs/backup/"
        echo "   - docs/cursor/backup/"
        ;;
    3)
        echo "⚠️  WARNING: This will DELETE original markdown files!"
        read -p "Are you sure? Type 'yes' to confirm: " confirm
        if [ "$confirm" = "yes" ]; then
            echo "🗑️  Deleting docs/*.md..."
            rm docs/*.md 2>/dev/null || echo "No .md files in docs/"
            
            echo "🗑️  Deleting docs/cursor/*.md..."
            rm docs/cursor/*.md 2>/dev/null || echo "No .md files in cursor/"
            
            echo "✅ Original markdown files deleted"
        else
            echo "❌ Operation cancelled"
        fi
        ;;
    4)
        echo "📋 Files that would be affected:"
        echo ""
        echo "docs/*.md:"
        ls docs/*.md 2>/dev/null || echo "No .md files found"
        echo ""
        echo "docs/cursor/*.md:"
        ls docs/cursor/*.md 2>/dev/null || echo "No .md files found"
        ;;
    *)
        echo "❌ Invalid option"
        ;;
esac

echo ""
echo "🎯 Next steps:"
echo "1. Use the wiki: open docs/wiki/tiddler_viewer.html"
echo "2. Set up TiddlyWiki: ./docs/wiki/setup_wiki.sh"
echo "3. Edit tiddler files in docs/wiki/tiddlers/"
"""
    
    cleanup_path = Path("docs/wiki/cleanup_markdown.sh")
    with open(cleanup_path, 'w', encoding='utf-8') as f:
        f.write(cleanup_script)
    
    # Make it executable
    os.chmod(cleanup_path, 0o755)
    
    print(f"✅ Created cleanup script: {cleanup_path}")

def main():
    """Main function to convert all markdown files to wiki format."""
    print("🚀 Converting ALL markdown files to TiddlyWiki format...")
    
    # Create tiddlers directory
    tiddlers_dir = Path("docs/wiki/tiddlers")
    tiddlers_dir.mkdir(exist_ok=True)
    
    # Get all markdown files from docs and cursor directories
    docs_dir = Path("docs")
    cursor_dir = Path("docs/cursor")
    
    md_files = []
    
    # Add docs/*.md files
    if docs_dir.exists():
        md_files.extend(list(docs_dir.glob("*.md")))
    
    # Add docs/cursor/*.md files
    if cursor_dir.exists():
        md_files.extend(list(cursor_dir.glob("*.md")))
    
    print(f"📋 Found {len(md_files)} markdown files to convert:")
    for md_file in md_files:
        print(f"   - {md_file}")
    
    # Convert each markdown file
    tiddler_files = []
    for md_file in md_files:
        tiddler_filename = convert_markdown_to_tiddler(md_file, tiddlers_dir)
        if tiddler_filename:
            tiddler_files.append(tiddler_filename)
    
    # Create TiddlyWiki configuration
    create_tiddlywiki_config(tiddler_files)
    
    # Create navigation guide
    create_wiki_navigation(tiddler_files)
    
    # Create cleanup script
    create_cleanup_script()
    
    print(f"\n✅ Conversion complete!")
    print(f"📁 Tiddler files created in: {tiddlers_dir}")
    print(f"📋 Total files converted: {len(tiddler_files)}")
    
    print("\n📋 Next steps:")
    print("1. View documents: open docs/wiki/tiddler_viewer.html")
    print("2. Set up TiddlyWiki: ./docs/wiki/setup_wiki.sh")
    print("3. Clean up original files: ./docs/wiki/cleanup_markdown.sh")
    
    print("\n🎯 Benefits of this structure:")
    print("- Modular: Each document is separate")
    print("- Searchable: Advanced search in TiddlyWiki")
    print("- Navigable: Easy to find information")
    print("- Maintainable: Easy to update individual documents")
    print("- Version controlled: Each tiddler can be tracked separately")

if __name__ == "__main__":
    main()
