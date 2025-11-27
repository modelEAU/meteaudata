"""pytest configuration and fixtures."""
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def mock_webbrowser_open():
    """Automatically mock webbrowser.open for all tests to prevent browser pop-ups.

    This fixture prevents the show_graph_in_browser() method from actually opening
    a browser window during test execution. The browser opening functionality is
    still testable, but won't interrupt the test runner or CI/CD pipelines.

    The mock is automatically applied to all tests (autouse=True), so no explicit
    fixture parameter is needed in test functions.
    """
    with patch('webbrowser.open') as mock_open:
        yield mock_open
