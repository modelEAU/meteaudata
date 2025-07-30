#!/usr/bin/env python3
"""
Asset copier for meteaudata documentation.

This script copies generated HTML plot files to locations that MkDocs will serve properly.
It registers the files with mkdocs_gen_files so they're included in the build.
"""

import os
import shutil
import sys
from pathlib import Path
import mkdocs_gen_files

def copy_generated_assets():
    """Copy generated HTML and PNG assets to be served by MkDocs."""
    
    # Source directory with generated assets
    assets_dir = Path('docs/assets/generated')
    
    if not assets_dir.exists():
        print("No generated assets directory found")
        return
    
    # Find all HTML and PNG files in the assets directory
    html_files = list(assets_dir.glob('*.html'))
    png_files = list(assets_dir.glob('*.png'))
    
    all_files = html_files + png_files
    
    if not all_files:
        print("No HTML or PNG files found in assets directory")
        return
    
    print(f"Found {len(html_files)} HTML files and {len(png_files)} PNG files to copy:")
    
    # Copy each file using mkdocs_gen_files
    for asset_file in all_files:
        # Create the target path in the built site
        # This will be served at the root level of the site
        target_path = f"assets/generated/{asset_file.name}"
        
        print(f"  Copying {asset_file.name} -> {target_path}")
        
        if asset_file.suffix == '.html':
            # Read HTML files as text
            content = asset_file.read_text(encoding='utf-8')
            with mkdocs_gen_files.open(target_path, "w") as f:
                f.write(content)
        elif asset_file.suffix == '.png':
            # Read PNG files as binary
            content = asset_file.read_bytes()
            with mkdocs_gen_files.open(target_path, "wb") as f:
                f.write(content)
    
    print(f"Copied {len(all_files)} asset files for MkDocs serving")

if __name__ == "__main__":
    copy_generated_assets()

# This is called by mkdocs-gen-files
copy_generated_assets()