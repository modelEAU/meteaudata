#!/usr/bin/env python3
"""
Template preprocessor for meteaudata documentation.

This script processes *_template.md files and generates the corresponding .md files
with executable code blocks processed and outputs embedded.
"""

import os
import sys
from pathlib import Path
import mkdocs_gen_files

# Add the scripts directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from exec_processor import CodeExecutor
    print("=== STARTING TEMPLATE PROCESSING ===")
    print(f"Current working directory: {os.getcwd()}")
except ImportError as e:
    print(f"ERROR: Could not import exec_processor: {e}")
    print("Make sure exec_processor.py is in the scripts directory")
    exit(1)


def find_template_files():
    """Find all *_template.md files in the docs directory."""
    docs_dir = Path('docs')
    template_files = []
    
    for template_file in docs_dir.rglob('*_template.md'):
        # Skip files in certain directories
        if any(part.startswith('.') for part in template_file.parts):
            continue
        if 'site' in template_file.parts:
            continue
            
        template_files.append(template_file)
    
    return sorted(template_files)


def get_output_path(template_path):
    """Convert template path to output path by removing '_template' suffix."""
    # Convert /path/to/file_template.md -> /path/to/file.md
    stem = template_path.stem.replace('_template', '')
    return template_path.parent / f"{stem}.md"


def process_template_file(template_path, output_path):
    """Process a single template file."""
    print(f"Processing template: {template_path} -> {output_path}")
    
    try:
        # Read template content
        template_content = template_path.read_text(encoding='utf-8')
        
        # Check if it has executable code blocks
        if 'python exec' in template_content:
            print(f"  Found executable code blocks in {template_path}")
            
            # Use the existing CodeExecutor to process executable blocks
            executor = CodeExecutor()
            
            # Create a temporary output file path for processing
            temp_output = template_path.parent / f"temp_{output_path.name}"
            
            # Process the template (this modifies the file in place)
            executor.process_markdown_file(template_path, temp_output)
            
            # Read the processed content
            processed_content = temp_output.read_text(encoding='utf-8')
            
            # Clean up temp file
            temp_output.unlink()
            
        else:
            print(f"  No executable code blocks in {template_path}, copying as-is")
            processed_content = template_content
        
        # Write the result using mkdocs_gen_files
        relative_output_path = output_path.relative_to(Path('docs'))
        with mkdocs_gen_files.open(str(relative_output_path), "w") as f:
            f.write(processed_content)
            
        print(f"  ✓ Generated {relative_output_path}")
        
    except Exception as e:
        print(f"  ERROR processing {template_path}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point for template processing."""
    
    # Find all template files
    template_files = find_template_files()
    
    if not template_files:
        print("No template files (*_template.md) found")
        return
    
    print(f"Found {len(template_files)} template files:")
    for template_file in template_files:
        print(f"  {template_file}")
    
    print("\nProcessing templates...")
    
    # Process each template file
    for template_file in template_files:
        output_path = get_output_path(template_file)
        process_template_file(template_file, output_path)
    
    print(f"\n=== COMPLETED TEMPLATE PROCESSING ===")



main()

