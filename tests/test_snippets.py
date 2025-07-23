import sys
from pathlib import Path
import importlib.util

def import_snippet(snippet_name: str):
    """Dynamically import a snippet file."""
    # Look for snippets in the docs directory
    project_root = Path(__file__).parent.parent
    snippet_path = project_root / "docs" / "snippets" / f"{snippet_name}.py"
    
    if not snippet_path.exists():
        # Fallback to old location for backward compatibility
        snippet_path = project_root / "snippets" / f"{snippet_name}.py"
    
    if not snippet_path.exists():
        raise ImportError(f"Snippet {snippet_name} not found")
    
    spec = importlib.util.spec_from_file_location(snippet_name, snippet_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_create_signal():
    """Test that create_signal snippet runs without errors."""
    try:
        import_snippet("create_signal")
        print("create_signal snippet executed successfully")
        assert True
    except Exception as e:
        print(f"create_signal snippet failed: {e}")
        raise

def test_create_dataset():
    """Test that create_dataset snippet runs without errors."""
    try:
        import_snippet("create_dataset")
        print("create_dataset snippet executed successfully") 
        assert True
    except Exception as e:
        print(f"create_dataset snippet failed: {e}")
        raise

def test_all_together_dataset():
    """Test that all_together_dataset snippet runs without errors."""
    try:
        import_snippet("all_together_dataset")
        print("all_together_dataset snippet executed successfully")
        assert True
    except Exception as e:
        print(f"all_together_dataset snippet failed: {e}")
        raise