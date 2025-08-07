# PMU Instance Sampah Fuzzy - Modular Documentation Wiki

## 📋 Overview

This directory contains a **modular TiddlyWiki-style documentation structure** for the PMU Instance Sampah Fuzzy project. All existing markdown files from the `docs/` directory have been converted to individual TiddlyWiki tiddler files for better organization and navigation.

## 🚀 Quick Start

### Option 1: Use the Tiddler Viewer (Immediate)
```bash
# Open the HTML viewer in your browser
open docs/wiki/tiddler_viewer.html
# or
firefox docs/wiki/tiddler_viewer.html
# or
google-chrome docs/wiki/tiddler_viewer.html
```

### Option 2: Set up Full TiddlyWiki
```bash
# Run the automated setup script
chmod +x docs/wiki/setup_wiki.sh
./docs/wiki/setup_wiki.sh

# Start the wiki server
cd mywiki
tiddlywiki --listen
```

### Option 3: Manual TiddlyWiki Setup
```bash
# Install TiddlyWiki
npm install -g tiddlywiki

# Create new wiki
tiddlywiki mywiki --init

# Copy all tiddler files
cp docs/wiki/tiddlers/*.tid mywiki/tiddlers/

# Copy configuration
cp docs/wiki/tiddlywiki.info mywiki/

# Start server
cd mywiki
tiddlywiki --listen
```

## 📁 File Structure

```
docs/wiki/
├── tiddler_viewer.html           # HTML viewer for browsing tiddlers
├── convert_md_to_wiki.py         # Script to convert markdown to tiddlers
├── setup_wiki.sh                 # Automated setup script
├── README.md                     # This file
├── NAVIGATION.md                 # Navigation guide
├── tiddlywiki.info               # TiddlyWiki configuration
├── tiddlers/                     # All converted tiddler files
│   ├── pipeline_scripts_guide.tid
│   ├── pipeline_differences_and_folder_issues_guide.tid
│   ├── pipeline_quick_reference.tid
│   ├── deployment_guide.tid
│   ├── google_colab_execution_guide_for_pmu_instance_sampah_fuzzy.tid
│   ├── coco_structure_compliance.tid
│   ├── onnx_and_rknn_troubleshooting_guide.tid
│   ├── dataset_fix_integration_guide.tid
│   ├── import_warnings_fix_summary.tid
│   ├── comprehensive_testing_guide.tid
│   ├── api_reference.tid
│   ├── code_structure_and_testing_organization_memory.tid
│   ├── code_restructure_and_testing_organization_summary.tid
│   ├── test_folder_naming_convention_reorganization.tid
│   ├── naming_convention_reorganization_-_completed.tid
│   ├── root_folder_cleanup_summary.tid
│   ├── git_push_summary_-_successful_deployment.tid
│   ├── compressed_folder_organization.tid
│   └── gitignore_update_-_compressed_folder.tid
└── images/                       # Images for the wiki
```

## 🎯 Features

### ✅ Modular Structure
- **19 separate tiddler files** - Each document is independent
- **Easy navigation** - Sidebar with categorized sections
- **Search functionality** - Find content quickly
- **Version control friendly** - Each tiddler can be tracked separately

### ✅ TiddlyWiki Benefits
- **Advanced search** - Full-text search across all documents
- **Tag-based organization** - Automatic tagging based on content
- **Cross-references** - Link between related documents
- **Multiple themes** - Choose different visual styles
- **Export capabilities** - Build static HTML versions

### ✅ Easy Maintenance
- **Edit individual files** - Update specific documents without affecting others
- **Add new documents** - Create new `.tid` files easily
- **Update existing content** - Modify tiddler files directly
- **Backup and restore** - Simple file-based storage

## 📋 Document Categories

### 🚀 Getting Started
- Overview and introduction
- Installation guides
- Quick start tutorials

### 📊 Pipeline Documentation
- **Pipeline Scripts Guide** - Comprehensive pipeline documentation
- **Pipeline Differences Guide** - Differences between pipeline types
- **Pipeline Quick Reference** - Quick reference for pipelines

### ⚙️ Configuration & Setup
- **Deployment Guide** - Complete deployment instructions
- **Colab Execution Guide** - Google Colab usage
- **COCO Structure Compliance** - COCO format compliance

### 🔧 Troubleshooting
- **ONNX/RKNN Troubleshooting** - ONNX and RKNN specific issues
- **Dataset Fix Integration** - Dataset fixes and integration
- **Import Warnings Fix** - Import warnings and solutions

### 📚 Advanced Topics
- **Testing Guide** - Comprehensive testing procedures
- **API Reference** - Complete API documentation
- **Code Structure Memory** - Code organization notes

### 🛠️ Development & Maintenance
- **Code Restructure Summary** - Restructuring documentation
- **Naming Convention Reorganization** - Naming conventions
- **Root Folder Cleanup** - Cleanup documentation

### 📈 Project History
- **Git Push Summary** - Git operations summary
- **Compressed Folder Organization** - File organization
- **Gitignore Updates** - Gitignore configuration

## 🔧 Customization

### Adding New Documents
1. Create new `.tid` file in `docs/wiki/tiddlers/`
2. Add to `tiddlywiki.info` configuration
3. Update navigation in `tiddler_viewer.html`

### Editing Existing Documents
1. Edit the `.tid` file directly
2. Update metadata if needed (title, tags, etc.)
3. Test in TiddlyWiki server

### Styling
- Modify TiddlyWiki themes
- Edit CSS in `tiddler_viewer.html`
- Add custom styles

### Content
- Edit tiddler files directly
- Add images to `docs/wiki/images/`
- Update links and references

## 🚀 Usage Examples

### View Documents Immediately
```bash
# Open HTML viewer
open docs/wiki/tiddler_viewer.html
```

### Set up Full TiddlyWiki
```bash
# Automated setup
./docs/wiki/setup_wiki.sh

# Manual setup
npm install -g tiddlywiki
tiddlywiki mywiki --init
cp docs/wiki/tiddlers/*.tid mywiki/tiddlers/
cd mywiki && tiddlywiki --listen
```

### Build Static Version
```bash
cd mywiki
tiddlywiki --build index
# Static files will be in output/
```

### Search and Navigate
- Use the search box in the HTML viewer
- Browse by category in the sidebar
- Use TiddlyWiki's advanced search features

## 📊 Benefits Over Original Structure

### ✅ Better Organization
- **Categorized navigation** instead of scattered files
- **Logical grouping** of related documents
- **Easy to find** specific information

### ✅ Improved Readability
- **Clean, modern design**
- **Consistent formatting**
- **Better typography** and spacing

### ✅ Enhanced User Experience
- **Fast navigation** with sidebar
- **Search functionality**
- **Mobile-friendly** interface

### ✅ Easy Maintenance
- **Modular content structure**
- **Version control friendly**
- **Easy to update** individual sections

## 🔍 Navigation Guide

### Quick Access
- **HTML Viewer**: `docs/wiki/tiddler_viewer.html`
- **Tiddler Files**: `docs/wiki/tiddlers/`
- **Setup Script**: `docs/wiki/setup_wiki.sh`
- **Navigation Guide**: `docs/wiki/NAVIGATION.md`

### Document Categories
1. **Pipeline Documentation** - Scripts and guides
2. **Configuration & Setup** - Deployment and setup
3. **Troubleshooting** - Common issues and fixes
4. **Advanced Topics** - Testing and API reference
5. **Development & Maintenance** - Code organization
6. **Project History** - Git operations and file management

## 📚 Related Resources

- [Main README](../README.md)
- [Pipeline Scripts Guide](../PIPELINE_SCRIPTS_GUIDE.md)
- [Testing Guide](../TESTING_GUIDE.md)
- [Deployment Guide](../DEPLOYMENT_GUIDE.md)

## 🎉 Summary

This modular wiki structure provides:

- ✅ **19 separate documents** converted from markdown
- ✅ **Easy navigation** with categorized sidebar
- ✅ **Search functionality** for quick access
- ✅ **TiddlyWiki compatibility** for advanced features
- ✅ **Version control friendly** structure
- ✅ **Mobile-responsive** design
- ✅ **Immediate use** with HTML viewer
- ✅ **Full TiddlyWiki** setup available

The wiki makes it much easier to navigate and find information compared to the original scattered markdown files! 🚀
