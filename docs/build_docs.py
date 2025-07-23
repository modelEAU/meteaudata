import os
import re
from pathlib import Path
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

def insert_code_snippet(markdown_content: str, base_path: Path) -> str:
    """Insert code snippets into markdown content."""
    pattern = r"<!-- INSERT CODE: (.+?) -->"
    
    def replace_with_code(match):
        code_file_path = match.group(1).strip()
        
        # Handle different path formats
        if code_file_path.startswith('./docs/snippets/'):
            code_file_path = code_file_path.replace('./docs/snippets/', '')
        elif code_file_path.startswith('docs/snippets/'):
            code_file_path = code_file_path.replace('docs/snippets/', '')
        
        full_path = base_path / code_file_path
        try:
            with open(full_path, "r") as code_file:
                code_content = code_file.read()
            return f"```python\n{code_content}\n```"
        except FileNotFoundError:
            return f"<!-- ERROR: File not found: {code_file_path} at {full_path} -->"
    
    return re.sub(pattern, replace_with_code, markdown_content)

def build_readme():
    """Build the main README from template and snippets."""
    project_root = Path(__file__).parent.parent
    template_path = project_root / "docs" / "README_template.md"
    snippets_path = project_root / "docs" / "snippets"
    output_path = project_root / "README.md"
    
    print(f"Project root: {project_root}")
    print(f"Template path: {template_path}")
    print(f"Snippets path: {snippets_path}")
    print(f"Output path: {output_path}")
    
    if not template_path.exists():
        print(f"Template not found at {template_path}")
        return
    
    if not snippets_path.exists():
        print(f"Snippets directory not found at {snippets_path}")
        return
    
    with open(template_path, "r") as f:
        template_content = f.read()
    
    # Process the template to insert code snippets
    updated_content = insert_code_snippet(template_content, snippets_path)
    
    with open(output_path, "w") as f:
        f.write(updated_content)
    
    print(f"README.md generated at {output_path}")

class CustomBuildHook(BuildHookInterface):
    """Custom build hook to generate documentation."""
    
    def initialize(self, version, build_data):
        """Called before the build starts."""
        print("Generating documentation...")
        build_readme()
        print("Documentation generation complete.")

if __name__ == "__main__":
    build_readme()