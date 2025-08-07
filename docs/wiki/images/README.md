# Wiki Images Directory

This directory is for storing images that will be used in the TiddlyWiki documentation.

## Usage

Place any images (PNG, JPG, SVG, etc.) in this directory and they will be automatically copied to the TiddlyWiki `files/` directory when you run the setup script.

## Supported Formats

- PNG (.png)
- JPEG (.jpg, .jpeg)
- GIF (.gif)
- SVG (.svg)
- WebP (.webp)
- ICO (.ico)

## How it works

When you run `./docs/wiki/setup_wiki.sh`, the script will:

1. Check if this `docs/wiki/images/` directory exists
2. If it exists, copy all files to `mywiki/files/`
3. The images will then be available in your TiddlyWiki

## Adding Images

1. Place your image files in this directory
2. Run the setup script: `./docs/wiki/setup_wiki.sh`
3. The images will be available in your wiki

## Example

```
docs/wiki/images/
├── README.md
├── pipeline-diagram.png
├── setup-screenshot.jpg
└── logo.svg
```

## Notes

- This directory is optional - the wiki will work fine without any images
- Large images may slow down the wiki loading
- Consider optimizing images for web use
- The setup script will create this directory if it doesn't exist
