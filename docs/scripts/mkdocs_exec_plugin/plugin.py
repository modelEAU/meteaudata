"""
MkDocs plugin for executing Python code snippets and injecting outputs.

This plugin processes markdown files looking for Python code blocks marked with 
'exec' and executes them, capturing outputs and injecting them into the documentation.

Phase 1: Basic execution for complete code snippets.
"""

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mkdocs.config import config_options
from mkdocs.plugins import BasePlugin
from mkdocs.structure.files import File
from mkdocs.structure.pages import Page


class ExecConfig(config_options.Config):
    """Configuration for the exec plugin."""
    enabled = config_options.Type(bool, default=True)
    timeout = config_options.Type(int, default=30)
    cache_dir = config_options.Type(str, default='.mkdocs_exec_cache')
    assets_dir = config_options.Type(str, default='docs/assets/generated')
    show_source = config_options.Type(bool, default=True)


class CodeExecutor:
    """Handles execution of Python code snippets."""
    
    def __init__(self, timeout: int = 30, assets_dir: str = 'docs/assets/generated'):
        self.timeout = timeout
        self.assets_dir = Path(assets_dir)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
    
    def execute_code(self, code: str, code_hash: str) -> Tuple[str, str, List[str]]:
        """
        Execute Python code and capture outputs.
        
        Args:
            code: Python code to execute
            code_hash: Hash of the code for asset naming
            
        Returns:
            Tuple of (stdout, stderr, generated_files)
        """
        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Set up the execution environment
            setup_code = self._get_setup_code(code_hash)
            full_code = setup_code + '\n' + code
            f.write(full_code)
            temp_file = f.name
        
        try:
            # Execute using the same Python environment (uv run)
            result = subprocess.run(
                ['uv', 'run', 'python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=os.getcwd()
            )
            
            stdout = result.stdout
            stderr = result.stderr
            
            # Find any generated files
            generated_files = self._find_generated_files(code_hash)
            
            return stdout, stderr, generated_files
            
        except subprocess.TimeoutExpired:
            return "", f"Code execution timed out after {self.timeout} seconds", []
        except Exception as e:
            return "", f"Execution error: {str(e)}", []
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except OSError:
                pass
    
    def _get_setup_code(self, code_hash: str) -> str:
        """Generate setup code for the execution environment."""
        return f"""
import sys
import os
import warnings
from pathlib import Path

# Set up matplotlib for non-interactive use
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set up plotly for file output
import plotly.io as pio
import plotly.graph_objects as go

# Configure output directory
OUTPUT_DIR = Path(r"{self.assets_dir.absolute()}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Override show() methods to save files instead
def _save_plot(fig, filename_prefix="plot"):
    filename = OUTPUT_DIR / f"{filename_prefix}_{code_hash[:8]}.png"
    fig.write_image(str(filename))
    print(f"[GENERATED_FILE]{{filename}}")
    return filename

# Monkey patch plotly show
original_plotly_show = go.Figure.show
def plotly_show_override(self, *args, **kwargs):
    filename = _save_plot(self, "plotly_plot")
    print(f"Plot saved to {{filename}}")

go.Figure.show = plotly_show_override

# Monkey patch matplotlib show  
original_plt_show = plt.show
def plt_show_override(*args, **kwargs):
    filename = OUTPUT_DIR / f"matplotlib_plot_{code_hash[:8]}.png"
    plt.savefig(str(filename), dpi=150, bbox_inches='tight')
    print(f"[GENERATED_FILE]{{filename}}")
    print(f"Plot saved to {{filename}}")
    plt.close()

plt.show = plt_show_override

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
"""
    
    def _find_generated_files(self, code_hash: str) -> List[str]:
        """Find files generated during code execution."""
        # For now, look for files with the code hash in the name
        generated = []
        if self.assets_dir.exists():
            for file in self.assets_dir.glob(f"*{code_hash[:8]}*"):
                generated.append(str(file.relative_to(Path.cwd())))
        return generated


class MkDocsExecPlugin(BasePlugin[ExecConfig]):
    """MkDocs plugin for executing code snippets."""
    
    def __init__(self):
        super().__init__()
        self.executor = None
        self.cache = {}
    
    def on_config(self, config):
        """Initialize the plugin with configuration."""
        if not self.config.enabled:
            return config
            
        self.executor = CodeExecutor(
            timeout=self.config.timeout,
            assets_dir=self.config.assets_dir
        )
        
        # Create cache directory
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        return config
    
    def on_page_markdown(self, markdown: str, page: Page, config, files) -> str:
        """Process markdown to execute code blocks."""
        if not self.config.enabled or not self.executor:
            return markdown
        
        # Find all python exec code blocks
        pattern = r'```python exec(?:\s*=\s*"[^"]*")?\s*\n(.*?)\n```'
        
        def replace_code_block(match):
            code = match.group(1)
            return self._process_code_block(code, page.file.src_path)
        
        # Process all matches
        processed = re.sub(pattern, replace_code_block, markdown, flags=re.DOTALL)
        return processed
    
    def _process_code_block(self, code: str, source_path: str) -> str:
        """Process a single code block."""
        # Generate hash for caching
        code_hash = hashlib.md5(f"{source_path}:{code}".encode()).hexdigest()
        
        # Check cache first (simple implementation for now)
        if code_hash in self.cache:
            return self.cache[code_hash]
        
        # Execute the code
        stdout, stderr, generated_files = self.executor.execute_code(code, code_hash)
        
        # Build the output
        result_parts = []
        
        # Show original code if configured
        if self.config.show_source:
            result_parts.append(f"```python\n{code}\n```")
        
        # Add output section
        if stdout or stderr:
            result_parts.append("\n**Output:**")
            
            if stdout:
                # Clean up stdout (remove our internal markers)
                clean_stdout = re.sub(r'\[GENERATED_FILE\][^\n]*\n?', '', stdout)
                if clean_stdout.strip():
                    result_parts.append(f"```\n{clean_stdout.strip()}\n```")
            
            if stderr:
                result_parts.append(f"\n**Errors:**\n```\n{stderr.strip()}\n```")
        
        # Add generated files (images, plots, etc.)
        for file_path in generated_files:
            if file_path.endswith(('.png', '.jpg', '.jpeg', '.svg')):
                # Convert to relative path from docs root
                rel_path = file_path.replace('docs/', '../')
                result_parts.append(f'\n<img src="{rel_path}" alt="Generated plot" style="max-width: 100%;">')
        
        result = '\n'.join(result_parts)
        self.cache[code_hash] = result
        return result


def makeExtension(**kwargs):
    """Required for MkDocs plugin system."""
    return MkDocsExecPlugin(**kwargs)