#!/usr/bin/env python3
"""
Script to process executable code blocks in documentation.

This script is run by mkdocs-gen-files during documentation build to:
1. Find all markdown files with executable code blocks
2. Execute the code and capture outputs
3. Generate processed versions with embedded results

This script integrates the executable code system with the existing MkDocs workflow.
"""

import os
import sys
from pathlib import Path
import mkdocs_gen_files

# Add the scripts directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from exec_processor import CodeExecutor
    from exec_contexts import list_contexts, get_context_description
    print("=== STARTING EXECUTABLE CODE PROCESSING ===")
    print(f"Current working directory: {os.getcwd()}")
except ImportError as e:
    print(f"ERROR: Could not import exec_processor: {e}")
    print("Make sure exec_processor.py is in the scripts directory")
    exit(1)


def process_executable_documentation():
    """Main function to process all documentation with executable code blocks."""
    docs_dir = Path('docs')
    executor = CodeExecutor()
    
    print(f"Looking for executable markdown files in: {docs_dir.absolute()}")
    
    # Find all markdown files
    md_files = list(docs_dir.rglob('*.md'))
    executable_files = []
    
    # Check which files have executable code blocks
    for md_file in md_files:
        # Skip certain directories and files
        if any(part.startswith('.') for part in md_file.parts):
            continue
        if 'site' in md_file.parts:
            continue
        if md_file.name.startswith('test_'):  # Skip our test files
            continue
            
        try:
            content = md_file.read_text(encoding='utf-8')
            if 'python exec' in content:
                executable_files.append(md_file)
                print(f"Found executable code in: {md_file}")
        except Exception as e:
            print(f"Error reading {md_file}: {e}")
    
    if not executable_files:
        print("No executable code blocks found in documentation")
        return
    
    print(f"Processing {len(executable_files)} files with executable code...")
    
    # Process each file with executable code
    for md_file in executable_files:
        try:
            print(f"Processing: {md_file}")
            
            # Process the file in place - executor will modify the content
            executor.process_markdown_file(md_file, md_file)
            
        except Exception as e:
            print(f"Error processing {md_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print("=== COMPLETED EXECUTABLE CODE PROCESSING ===")


def generate_exec_contexts_documentation():
    """Generate documentation for available execution contexts."""
    print("Generating execution contexts documentation...")
    
    contexts = list_contexts()
    
    content = [
        "# Execution Contexts",
        "",
        "This page documents the available execution contexts for code snippets.",
        "",
        "## Available Contexts",
        "",
        "The following contexts are available for use with `python exec=\"setup:context_name\"`:",
        ""
    ]
    
    for context_name in sorted(contexts):
        description = get_context_description(context_name)
        content.extend([
            f"### `{context_name}`",
            "",
            description,
            ""
        ])
    
    content.extend([
        "## Usage Examples",
        "",
        "### Using a Context",
        "",
        "```python exec=\"setup:simple_signal\"",
        "# This code will have access to a pre-created signal",
        "print(f\"Signal name: {signal.name}\")",
        "```",
        "",
        "### Chaining Code Blocks",
        "",
        "```python exec",
        "x = 42",
        "print(f\"Initial value: {x}\")",
        "```",
        "",
        "```python exec=\"continue\"",
        "y = x * 2",
        "print(f\"Doubled: {y}\")",
        "```",
        ""
    ])
    
    # Write using mkdocs_gen_files
    with mkdocs_gen_files.open("development/execution-contexts.md", "w") as f:
        f.write("\\n".join(content))
    
    print("Generated execution contexts documentation")


def main():
    """Main entry point for the gen-files script."""
    try:
        # First, process all executable documentation
        process_executable_documentation()
        
        # Then generate documentation about the exec system
        generate_exec_contexts_documentation()
        
    except Exception as e:
        print(f"Error in executable docs processing: {e}")
        import traceback
        traceback.print_exc()
        # Don't exit with error - let the build continue
        print("Continuing with documentation build despite exec processing error...")


if __name__ == "__main__":
    main()


# This is called by mkdocs-gen-files
main()