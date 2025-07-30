"""
Standalone code execution processor for MkDocs gen-files.

This processes markdown files to execute Python code blocks and inject outputs.
Works with the existing gen-files setup rather than as a separate plugin.
"""

import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from exec_contexts import get_context, list_contexts


class CodeExecutor:
    """Handles execution of Python code snippets."""
    
    def __init__(self, timeout: int = 30, assets_dir: str = 'docs/assets/generated'):
        self.timeout = timeout
        self.assets_dir = Path(assets_dir)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        # For testing, let's not use caching initially
        self.use_cache = False
        # Store execution state for continue blocks
        self.page_state: Dict[str, str] = {}  # page_path -> accumulated code
    
    def process_markdown_file(self, input_path: Path, output_path: Path) -> None:
        """Process a markdown file, executing code blocks and writing the result."""
        if not input_path.exists():
            return
            
        content = input_path.read_text(encoding='utf-8')
        
        # Reset page state for new file
        self.page_state[str(input_path)] = ""
        
        # Find all python exec code blocks with optional parameters
        pattern = r'```python exec(?:\s*=\s*"([^"]*)")?\s*\n(.*?)\n```'
        
        def replace_code_block(match):
            options = match.group(1) or ""  # exec options (e.g., "setup:simple_signal", "continue") 
            code = match.group(2)
            return self._process_code_block(code, str(input_path), options)
        
        # Process all matches
        processed_content = re.sub(pattern, replace_code_block, content, flags=re.DOTALL)
        
        # Always write for testing
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(processed_content, encoding='utf-8')
        if processed_content != content:
            print(f"Processed executable code in: {input_path}")
        else:
            print(f"No changes needed in: {input_path}")
    
    def _process_code_block(self, code: str, source_path: str, options: str = "") -> str:
        """Process a single code block with context support."""
        # Parse options
        context_name = None
        is_continue = False
        is_silent = False
        
        if options:
            option_parts = [opt.strip() for opt in options.split(',')]
            for opt in option_parts:
                if opt.startswith('setup:'):
                    # Legacy format: setup:context_name
                    context_name = opt[6:]  # Remove 'setup:' prefix
                elif opt == 'continue':
                    is_continue = True
                elif opt == 'silent':
                    is_silent = True
                else:
                    # If it's not a known special option, treat it as a context name
                    # This handles the new format: exec="context_name"
                    from exec_contexts import list_contexts
                    available_contexts = list_contexts()
                    if opt in available_contexts:
                        context_name = opt
                    else:
                        # If it's not a known context, it might be some other option
                        # You could add logging here if needed
                        pass
        
        # Build the full code to execute
        full_code = self._build_full_code(code, source_path, context_name, is_continue)
        
        # Generate hash for caching (include context in hash)
        cache_key = f"{source_path}:{options}:{code}"
        code_hash = hashlib.md5(cache_key.encode()).hexdigest()
        
        # Check cache
        if self.use_cache and code_hash in self.cache:
            return self.cache[code_hash]
        
        # Execute the code
        stdout, stderr, generated_files = self._execute_code(full_code, code_hash)
        
        # Update page state for continue blocks
        if is_continue:
            # For continue blocks, accumulate the new code
            if source_path not in self.page_state:
                self.page_state[source_path] = ""
            # Add just the new code to the accumulated state
            self.page_state[source_path] += "\n\n" + code
        else:
            # For setup/context blocks, store the complete executable code
            # This includes the context + the actual code
            self.page_state[source_path] = full_code
        
        # Build the output
        result_parts = []
        
        # Show original code (unless silent)
        if not is_silent:
            result_parts.append(f"```python\n{code}\n```")
        
        # Add output section
        if stdout or stderr:
            result_parts.append("\n**Output:**")
            
            if stdout:
                # Clean up stdout (remove our internal markers and plot messages)
                clean_stdout = re.sub(r'\[GENERATED_FILE\][^\n]*\n?', '', stdout)
                clean_stdout = re.sub(r'Plot saved as HTML:.*?\n', '', clean_stdout)
                clean_stdout = re.sub(r'Plot saved as PNG:.*?\n', '', clean_stdout)
                clean_stdout = re.sub(r'meteaudata [a-z_]+ saved to.*?\n', '', clean_stdout)
                clean_stdout = re.sub(r'Captured HTML display:.*?\n', '', clean_stdout)
                clean_stdout = re.sub(r'<IPython\.core\.display\.HTML object>\n?', '', clean_stdout)
                clean_stdout = re.sub(r'\(PNG export failed:.*?\)\n', '', clean_stdout, flags=re.DOTALL)
                clean_stdout = re.sub(r'Image export using.*?\n.*?pip install.*?\n.*?\)\n', '', clean_stdout, flags=re.DOTALL)
                
                if clean_stdout.strip():
                    result_parts.append(f"```\n{clean_stdout.strip()}\n```")
            
            if stderr:
                result_parts.append(f"\n**Errors:**\n```\n{stderr.strip()}\n```")
        
        # Add generated files (images, plots, etc.)
        for file_path in generated_files:
            if file_path.endswith(('.png', '.jpg', '.jpeg', '.svg')):
                # Handle image files - use relative path that works with MkDocs subpath
                filename = os.path.basename(file_path)
                # Use relative path that works regardless of site base URL
                img_src = f"../../assets/generated/{filename}"
                result_parts.append(f'\n<img src="{img_src}" alt="Generated plot" style="max-width: 100%; height: auto;">')
            elif file_path.endswith('.html'):
                # Handle HTML files - use relative path that works with MkDocs subpath
                filename = os.path.basename(file_path)
                # Use relative path that works regardless of site base URL
                iframe_src = f"../../assets/generated/{filename}"
                result_parts.append(f'\n<iframe src="{iframe_src}" width="100%" height="500" style="border: none; display: block; margin: 1em 0;"></iframe>')
        
        result = '\n'.join(result_parts)
        self.cache[code_hash] = result
        return result
    
    def _build_full_code(self, code: str, source_path: str, context_name: Optional[str], is_continue: bool) -> str:
        parts = []
        
        if is_continue:
            # For continue blocks, always start with previous state
            if source_path in self.page_state and self.page_state[source_path]:
                # Silence previous output but keep all variables
                previous_code = self.page_state[source_path]
                silenced_previous = f"""
import sys
from io import StringIO

# Capture and discard output from previous code blocks
old_stdout = sys.stdout
sys.stdout = StringIO()

try:
    {self._indent_code(previous_code, "    ")}
finally:
    sys.stdout = old_stdout
"""
                parts.append(silenced_previous)
        elif context_name:
            # For context blocks, start fresh with the context
            context_code = get_context(context_name)
            parts.append(context_code)
        
        # Add the actual code
        parts.append(code)
        
        return '\n\n'.join(parts)
    
    def _indent_code(self, code: str, indent: str) -> str:
        """Indent each line of code."""
        return '\n'.join(indent + line for line in code.split('\n'))
    
    def _execute_code(self, code: str, code_hash: str) -> Tuple[str, str, List[str]]:
        """Execute Python code and capture outputs."""
        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Set up the execution environment
            setup_code = self._get_setup_code(code_hash)
            
            # Add HTML capture setup to the code - using direct _build_html_content method
            html_capture_code = f'''
# Set up HTML capture for meteaudata display() methods
import sys
from io import StringIO

# Store captured HTML files
captured_html_files = []

try:
    from meteaudata.displayable import DisplayableBase
    
    # Store the original display method
    original_display = DisplayableBase.display
    
    def display_capture_wrapper(self, format="html", depth=2, max_depth=4, width=1200, height=800):
        """Wrapper for display method that captures HTML content."""
        if format == "html":
            # Get the complete HTML content with CSS styles
            try:
                # Import the HTML_STYLE constant to get the CSS
                from meteaudata.displayable import HTML_STYLE
                
                # Get the HTML content structure
                html_content = self._build_html_content(depth=depth)
                
                if html_content and isinstance(html_content, str):
                    # Extract CSS content from HTML_STYLE constant (remove <style> tags)
                    css_content = HTML_STYLE.replace('<style>', '').replace('</style>', '').strip()
                    
                    # Create complete HTML document with styles
                    complete_html = "<html>\\n<head>\\n<style type=\\"text/css\\">\\n" + css_content + "\\n</style>\\n</head>\\n<body>\\n<div class=\\"meteaudata-display\\">\\n" + html_content + "\\n</div>\\n</body>\\n</html>"
                    
                    # Save the complete HTML content to a file
                    file_count = len(captured_html_files) + 1
                    filename = f"display_content_{code_hash[:8]}_" + str(file_count) + ".html"
                    html_filename = OUTPUT_DIR / filename
                    with open(html_filename, 'w', encoding='utf-8') as f:
                        f.write(complete_html)
                    captured_html_files.append(str(html_filename))
                    print(f"[GENERATED_FILE]" + str(html_filename))
                    print(f"Captured HTML display: " + str(html_filename))
            except Exception as e:
                print(f"HTML capture failed: " + str(e))
        
        # Call the original display method for normal behavior
        return original_display(self, format, depth, max_depth, width, height)
    
    # Replace the display method
    DisplayableBase.display = display_capture_wrapper
    
except ImportError:
    pass  # meteaudata not available
'''
            
            full_code = setup_code + '\n' + html_capture_code + '\n' + code
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
        return f'''
import sys
import os
import warnings
from pathlib import Path

# Configure output directory
OUTPUT_DIR = Path(r"{self.assets_dir.absolute()}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Try to set up matplotlib if available
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Override matplotlib show to save files instead
    original_plt_show = plt.show
    def plt_show_override(*args, **kwargs):
        filename = OUTPUT_DIR / f"matplotlib_plot_{code_hash[:8]}.png"
        plt.savefig(str(filename), dpi=150, bbox_inches='tight')
        print(f"[GENERATED_FILE]{{filename}}")
        print(f"Plot saved to {{filename}}")
        plt.close()

    plt.show = plt_show_override
except ImportError:
    pass  # matplotlib not available

# Try to set up plotly if available
try:
    import plotly.io as pio
    import plotly.graph_objects as go
    
    def _save_plotly_plot(fig, filename_prefix="plotly_plot"):
        # Try to save as PNG (requires kaleido)
        png_filename = OUTPUT_DIR / f"{{filename_prefix}}_{code_hash[:8]}.png"
        html_filename = OUTPUT_DIR / f"{{filename_prefix}}_{code_hash[:8]}.html"
        
        try:
            # Try PNG export first
            fig.write_image(str(png_filename), width=800, height=600, scale=2)
            print(f"[GENERATED_FILE]{{png_filename}}")
            print(f"Plot saved as PNG: {{png_filename}}")
            return png_filename
        except Exception as e:
            # Fall back to HTML
            fig.write_html(str(html_filename), include_plotlyjs='cdn')
            print(f"[GENERATED_FILE]{{html_filename}}")
            print(f"Plot saved as HTML: {{html_filename}} (PNG export failed: {{e}})")
            return html_filename

    # Monkey patch plotly show
    original_plotly_show = go.Figure.show
    def plotly_show_override(self, *args, **kwargs):
        return _save_plotly_plot(self, "plotly_plot")

    go.Figure.show = plotly_show_override
    
    # Also patch the plotly express show method
    try:
        import plotly.express as px
        # Note: px plots return go.Figure objects, so they'll use the same override
    except ImportError:
        pass
        
except ImportError:
    pass  # plotly not available

# HTML display capture is now handled in the dynamic execution code

# Try to set up meteaudata plot interception
try:
    from meteaudata.types import Signal, TimeSeries, Dataset
    
    # Store original methods
    original_signal_plot = Signal.plot
    original_signal_plot_dependency_graph = Signal.plot_dependency_graph
    original_timeseries_plot = TimeSeries.plot
    
    def meteaudata_plot_wrapper(original_method, obj_type="plot"):
        def wrapper(self, *args, **kwargs):
            # Call the original method
            fig = original_method(self, *args, **kwargs)
            if fig is not None:
                # Save the plot
                filename = _save_plotly_plot(fig, f"meteaudata_{{obj_type}}")
                print(f"meteaudata {{obj_type}} saved to {{filename}}")
                return fig
            return None
        return wrapper
    
    # Monkey patch meteaudata methods
    Signal.plot = meteaudata_plot_wrapper(original_signal_plot, "signal_plot")
    Signal.plot_dependency_graph = meteaudata_plot_wrapper(original_signal_plot_dependency_graph, "dependency_graph")
    TimeSeries.plot = meteaudata_plot_wrapper(original_timeseries_plot, "timeseries_plot")
    
    # Also handle Dataset.plot if it exists
    try:
        original_dataset_plot = Dataset.plot
        Dataset.plot = meteaudata_plot_wrapper(original_dataset_plot, "dataset_plot")
    except AttributeError:
        pass  # Dataset might not have plot method
        
except ImportError:
    pass  # meteaudata not available

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
'''
    
    def _find_generated_files(self, code_hash: str) -> List[str]:
        """Find files generated during code execution."""
        generated = []
        if self.assets_dir.exists():
            for file in self.assets_dir.glob(f"*{code_hash[:8]}*"):
                generated.append(str(file))
        return generated


def process_docs_with_exec():
    """Process all markdown files looking for executable code blocks."""
    docs_dir = Path('docs')
    executor = CodeExecutor()
    
    print(f"Looking for markdown files in: {docs_dir.absolute()}")
    
    # Find all markdown files that might have executable code
    md_files = list(docs_dir.rglob('*_template.md'))
    print(f"Found {len(md_files)} markdown files")
    
    for md_file in md_files:
        print(f"Checking file: {md_file}")
        
        # Skip certain directories
        if any(part.startswith('.') for part in md_file.parts):
            print(f"  -> Skipping (hidden directory): {md_file}")
            continue
        if 'site' in md_file.parts:
            print(f"  -> Skipping (site directory): {md_file}")
            continue
            
        # Read and check if file has exec blocks
        try:
            content = md_file.read_text(encoding='utf-8')
            if 'python exec' in content:
                print(f"  -> Found exec blocks, processing: {md_file}")
                # Process in place for now
                executor.process_markdown_file(md_file, md_file)
            else:
                print(f"  -> No exec blocks found in: {md_file}")
        except Exception as e:
            print(f"Error processing {md_file}: {e}")


if __name__ == '__main__':
    process_docs_with_exec()