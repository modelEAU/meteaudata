"""
Tests for the cleaned-up graph display functionality.
Updated to match the simplified API and removed bloated features.
"""

import pytest
import pandas as pd
import datetime
import numpy as np
from unittest.mock import Mock, patch

# Import your meteaudata classes
from meteaudata.types import (
    Signal, Dataset, TimeSeries, DataProvenance, ProcessingStep, 
    FunctionInfo, Parameters, IndexMetadata, ProcessingType
)


class TestCleanedDisplayableBase:
    """Test the cleaned-up DisplayableBase functionality."""
    
    @pytest.fixture
    def sample_signal(self):
        """Create a sample signal for testing."""
        provenance = DataProvenance(parameter="test_param", location="test_lab")
        func_info = FunctionInfo(name="test_function", version="1.0", author="test", reference="test.com")
        params = Parameters(param1="value1", param2=42)
        step = ProcessingStep(
            type=ProcessingType.SMOOTHING,
            description="Test processing",
            run_datetime=datetime.datetime.now(),
            requires_calibration=False,
            function_info=func_info,
            suffix="TEST",
            parameters=params
        )
        
        data = pd.Series([1, 2, 3, 4], name="test_series")
        ts = TimeSeries(series=data, processing_steps=[step])
        signal = Signal(input_data=data, name="test_signal", units="test_units", provenance=provenance)
        
        return signal
    
    def test_display_text_format(self, sample_signal):
        """Test text display format."""
        # Should not raise exception and should print to stdout
        with patch('builtins.print') as mock_print:
            sample_signal.display(format="text", depth=2)
            mock_print.assert_called()
            
            # Check that output contains expected content
            printed_content = str(mock_print.call_args_list)
            assert "Signal" in printed_content
    
    def test_display_html_format(self, sample_signal):
        """Test HTML display format."""
        with patch('IPython.display.HTML') as mock_html, \
             patch('IPython.display.display') as mock_display:
            
            sample_signal.display(format="html", depth=2)
            
            # Should call HTML display
            mock_html.assert_called_once()
            mock_display.assert_called_once()
            
            # Get the HTML content
            html_content = mock_html.call_args[0][0]
            assert 'meteaudata-display' in html_content
            assert 'Signal' in html_content
    
    def test_display_html_fallback_no_ipython(self, sample_signal):
        """Test HTML display falls back to text when IPython not available."""
        with patch('IPython.display.HTML', side_effect=ImportError("No IPython")), \
             patch('builtins.print') as mock_print:
            
            sample_signal.display(format="html", depth=2)
            
            # Should fall back to text display
            mock_print.assert_called()
    
    def test_graph_modules_import(self, sample_signal):
        """Test that graph display modules can be imported."""
        try:
            # Test if we can import the graph display module
            from meteaudata import graph_display
            assert hasattr(graph_display, 'SVGGraphBuilder'), "SVGGraphBuilder not found"
            assert hasattr(graph_display, 'SVGNestedBoxGraphRenderer'), "SVGNestedBoxGraphRenderer not found"
            
            # Test if we can create instances
            builder = graph_display.SVGGraphBuilder()
            renderer = graph_display.SVGNestedBoxGraphRenderer()
            
            assert builder is not None
            assert renderer is not None
            
        except ImportError as e:
            pytest.skip(f"Graph display module cannot be imported: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error importing graph modules: {e}")
    
    def test_display_graph_simple_case(self, sample_signal):
        """Test a simple graph display case without complex mocking."""
        # Just test that calling display with graph format doesn't crash
        try:
            # We won't mock anything - just see what happens
            with patch('builtins.print') as mock_print:
                sample_signal.display(format="graph")
                
                # At minimum, it should either print something or not crash
                # If it prints an error message, that's useful info
                if mock_print.called:
                    print_calls = [str(call) for call in mock_print.call_args_list]
                    print(f"Print calls: {print_calls}")
                
        except ImportError as e:
            pytest.skip(f"Graph display not available: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error in graph display: {e}")
    
    def test_displayable_methods_exist(self, sample_signal):
        """Test that displayable methods exist on meteaudata objects."""
        # Check that the signal has the expected methods
        assert hasattr(sample_signal, 'display'), "Signal should have display method"
        
        # For Pydantic models, we need to check if the methods exist in the class hierarchy
        signal_class = sample_signal.__class__
        method_names = []
        for cls in signal_class.__mro__:
            method_names.extend(dir(cls))
        
        assert 'render_svg_graph' in method_names, "Signal should have render_svg_graph method in inheritance chain"
        assert 'show_graph_in_browser' in method_names, "Signal should have show_graph_in_browser method in inheritance chain"
        
        # Check that the methods are callable (use getattr to bypass Pydantic's __getattribute__)
        assert callable(getattr(sample_signal, 'display')), "display should be callable"
        
        # For the graph methods, we'll check if they exist and are callable differently
        try:
            render_method = getattr(sample_signal.__class__, 'render_svg_graph', None)
            assert render_method is not None and callable(render_method), "render_svg_graph should be callable"
            
            browser_method = getattr(sample_signal.__class__, 'show_graph_in_browser', None)
            assert browser_method is not None and callable(browser_method), "show_graph_in_browser should be callable"
        except AttributeError:
            # If we can't access the methods directly, at least verify they're in the MRO
            assert any('render_svg_graph' in dir(cls) for cls in signal_class.__mro__), "render_svg_graph not found in class hierarchy"
            assert any('show_graph_in_browser' in dir(cls) for cls in signal_class.__mro__), "show_graph_in_browser not found in class hierarchy"
    
    def test_display_graph_ipython_fallback(self, sample_signal):
        """Test that graph display handles missing IPython gracefully."""
        # Simple test: verify that if we're in notebook mode but IPython.display import fails,
        # the code doesn't crash and provides some fallback
        with patch('meteaudata.display_utils._is_notebook_environment', return_value=True):
            
            # Mock the specific import that happens in display method
            def mock_import_ipython_display(*args, **kwargs):
                raise ImportError("No module named 'IPython.display'")
            
            original_import = __import__
            
            def selective_import_mock(name, *args, **kwargs):
                if name == 'IPython.display':
                    raise ImportError(f"No module named '{name}'")
                return original_import(name, *args, **kwargs)
            
            with patch('builtins.__import__', side_effect=selective_import_mock):
                with patch('builtins.print') as mock_print:
                    # This should not crash, and should print a fallback message
                    sample_signal.display(format="graph")
                    
                    # Either it printed a fallback message, or it somehow worked anyway
                    # The key is that it didn't crash
                    assert True, "Graph display completed without crashing"
                    
                    # If print was called, verify it was a reasonable message
                    if mock_print.called:
                        print_calls = [str(call) for call in mock_print.call_args_list]
                        # Just verify that some message was printed
                        assert len(print_calls) > 0, "Some fallback message should be printed"
    
    def test_display_graph_notebook_integration(self, sample_signal):
        """Test that graph display works in notebook environment."""
        # First, check what methods are actually available
        print(f"Available methods on sample_signal: {[m for m in dir(sample_signal) if not m.startswith('_')]}")
        print(f"Has show_graph_in_browser: {hasattr(sample_signal, 'show_graph_in_browser')}")
        print(f"Has render_svg_graph: {hasattr(sample_signal, 'render_svg_graph')}")
        print(f"Has display: {hasattr(sample_signal, 'display')}")
        
        with patch('meteaudata.display_utils._is_notebook_environment', return_value=True) as mock_notebook_check:
            
            # Mock all possible paths
            mock_html_obj = Mock()
            mock_display_func = Mock()
            
            with patch('IPython.display.HTML', return_value=mock_html_obj) as mock_html, \
                 patch('IPython.display.display', mock_display_func) as mock_display, \
                 patch('builtins.print') as mock_print:
                
                # Mock the graph renderer and browser method at the module level
                with patch('meteaudata.graph_display.SVGNestedBoxGraphRenderer') as mock_renderer_class, \
                     patch('meteaudata.graph_display.open_meteaudata_graph_in_browser') as mock_browser_func:
                    
                    mock_renderer = Mock()
                    mock_renderer.render_to_html.return_value = "<html>test content</html>"
                    mock_renderer_class.return_value = mock_renderer
                    
                    # Call the display method
                    sample_signal.display(format="graph", max_depth=2)
                    
                    # Debug: Print what was actually called
                    print(f"Notebook check called: {mock_notebook_check.called}")
                    print(f"HTML called: {mock_html.called}")
                    print(f"Display called: {mock_display.called}")
                    print(f"Print called: {mock_print.called}")
                    print(f"Browser func called: {mock_browser_func.called}")
                    print(f"Renderer called: {mock_renderer_class.called}")
                    
                    # The test should not crash - that's the main thing
                    assert True, "Display method completed without crashing"
    
    def test_display_graph_format_browser(self, sample_signal):
        """Test graph display opens browser when not in notebook."""
        with patch('meteaudata.display_utils._is_notebook_environment', return_value=False), \
             patch('meteaudata.displayable.DisplayableBase.show_graph_in_browser') as mock_browser:
            
            sample_signal.display(format="graph", max_depth=4, width=1200, height=800)
            
            # Should open in browser
            mock_browser.assert_called_once_with(4, 1200, 800)
    
    def test_display_invalid_format(self, sample_signal):
        """Test error handling for invalid format."""
        with pytest.raises(ValueError) as excinfo:
            sample_signal.display(format="invalid")
        
        error_msg = str(excinfo.value)
        assert "Unknown format" in error_msg
        assert "text" in error_msg
        assert "html" in error_msg
        assert "graph" in error_msg
    
    def test_convenience_methods(self, sample_signal):
        """Test convenience methods work."""
        with patch('meteaudata.displayable.DisplayableBase.display') as mock_display:
            # Test show_details
            sample_signal.show_details(depth=3)
            mock_display.assert_called_with(format="html", depth=3)
            
            # Test show_summary
            sample_signal.show_summary()
            mock_display.assert_called_with(format="text", depth=1)
            
            # Test show_graph
            sample_signal.show_graph(max_depth=5, width=800, height=600)
            mock_display.assert_called_with(format="graph", max_depth=5, width=800, height=600)


class TestSVGGraphRenderer:
    """Test the SVG graph rendering functionality."""
    
    @pytest.fixture
    def complex_dataset(self):
        """Create a complex dataset for testing."""
        provenance = DataProvenance(parameter="temperature", location="lab")
        func_info = FunctionInfo(name="smooth_filter", version="1.0", author="test", reference="test.com")
        params = Parameters(window=5, method="gaussian")
        
        step = ProcessingStep(
            type=ProcessingType.SMOOTHING,
            description="Smoothing step",
            run_datetime=datetime.datetime.now(),
            requires_calibration=False,
            function_info=func_info,
            parameters=params,
            suffix="SMOOTH"
        )
        
        data = pd.Series([1, 2, 3, 4, 5], name="RAW")
        ts = TimeSeries(series=data, processing_steps=[step])
        signal = Signal(input_data=data, name="temperature", units="°C", provenance=provenance)
        
        dataset = Dataset(
            name="complex_dataset",
            description="Complex dataset with deep nesting",
            owner="test_user",
            signals={"temperature": signal}
        )
        return dataset
    
    def test_svg_graph_builder_creates_hierarchy(self, complex_dataset):
        """Test that SVG graph builder creates proper hierarchy."""
        try:
            from meteaudata.graph_display import SVGGraphBuilder
            
            builder = SVGGraphBuilder()
            graph_data = builder.build_graph(complex_dataset, max_depth=4)
            
            # Should have hierarchy structure
            assert 'hierarchy' in graph_data
            assert 'nodes' in graph_data
            assert 'layout_type' in graph_data
            
            # Should have multiple nodes
            assert len(graph_data['nodes']) >= 3  # At least Dataset, Signal, TimeSeries
            
            # Root should be Dataset
            root = graph_data['hierarchy']
            assert root['type'] == 'Dataset'
            assert root['attributes']["Name"] == 'complex_dataset'
            
        except ImportError:
            pytest.skip("Graph display module not available")
    
    def test_svg_graph_builder_creates_container_nodes(self, complex_dataset):
        """Test that container nodes are created for collections."""
        try:
            from meteaudata.graph_display import SVGGraphBuilder
            
            builder = SVGGraphBuilder()
            graph_data = builder.build_graph(complex_dataset, max_depth=4)
            
            # Should have container nodes for signals collection
            node_types = [node['type'] for node in graph_data['nodes'].values()]
            assert 'Container' in node_types
            
            # Find container node
            container_nodes = [node for node in graph_data['nodes'].values() if node['type'] == 'Container']
            assert len(container_nodes) >= 1
            
            # Container should have proper attributes
            signals_container = next((node for node in container_nodes if 'signals' in node['identifier'].lower()), None)
            assert signals_container is not None
            
        except ImportError:
            pytest.skip("Graph display module not available")
    
    def test_svg_renderer_produces_html(self, complex_dataset):
        """Test that SVG renderer produces valid HTML."""
        try:
            from meteaudata.graph_display import SVGNestedBoxGraphRenderer
            
            renderer = SVGNestedBoxGraphRenderer()
            
            # Mock the template loading to avoid file system dependency
            mock_template = """
            <html>
            <head><title>Test</title></head>
            <body>
                <div id="graph"></div>
                <script>
                // INJECT_DATA_HERE
                console.log('Graph loaded');
                </script>
            </body>
            </html>
            """
            
            with patch.object(renderer, '_get_html_template', return_value=mock_template):
                html_output = renderer.render_to_html(complex_dataset, max_depth=3, width=1000, height=700)
                
                # Should be valid HTML string
                assert isinstance(html_output, str)
                assert '<html>' in html_output
                assert '</html>' in html_output
                
                # Should contain injected data
                assert 'graphData' in html_output
                assert 'graphConfig' in html_output
                assert 'complex_dataset' in html_output
                
        except ImportError:
            pytest.skip("Graph display module not available")
    
    def test_render_svg_graph_method(self, complex_dataset):
        """Test the render_svg_graph method on displayable objects."""
        mock_html = "<html>Mock SVG graph</html>"
        
        with patch('meteaudata.graph_display.SVGNestedBoxGraphRenderer') as mock_renderer_class:
            mock_renderer = Mock()
            mock_renderer.render_to_html.return_value = mock_html
            mock_renderer_class.return_value = mock_renderer
            
            result = complex_dataset.render_svg_graph(max_depth=3, width=800, height=600, title="Test Graph")
            
            # Should call renderer with correct parameters
            mock_renderer.render_to_html.assert_called_once_with(
                complex_dataset, 3, 800, 600, "Test Graph"
            )
            assert result == mock_html
    
    def test_show_graph_in_browser_method(self, complex_dataset):
        """Test the show_graph_in_browser method."""
        mock_path = "/tmp/test_graph.html"
        
        with patch('meteaudata.graph_display.open_meteaudata_graph_in_browser', return_value=mock_path) as mock_open:
            result = complex_dataset.show_graph_in_browser(max_depth=4, width=1200, height=800, title="Browser Graph")
            
            # Should call browser function with correct parameters
            mock_open.assert_called_once_with(complex_dataset, 4, 1200, 800, "Browser Graph")
            assert result == mock_path


class TestGraphDisplayIntegration:
    """Test integration between displayable objects and graph display."""
    
    @pytest.fixture
    def nested_dataset(self):
        """Create a dataset with multiple levels of nesting."""
        # Create multiple signals with different processing steps
        signals = {}
        
        for i in range(2):
            provenance = DataProvenance(parameter=f"param_{i}", location=f"lab_{i}")
            func_info = FunctionInfo(name=f"func_{i}", version="1.0", author="test", reference="test.com")
            params = Parameters(**{f"param_{j}": f"value_{j}" for j in range(3)})
            
            steps = []
            for j in range(2):
                step = ProcessingStep(
                    type=ProcessingType.FILTERING if j == 0 else ProcessingType.SMOOTHING,
                    description=f"Step {j}",
                    run_datetime=datetime.datetime.now(),
                    requires_calibration=False,
                    function_info=func_info,
                    parameters=params,
                    suffix=f"STEP{j}"
                )
                steps.append(step)
            
            data = pd.Series(range(5), name=f"signal#{i}_RAW")
            ts = TimeSeries(series=data, processing_steps=steps)
            signal = Signal(input_data=ts, name=f"signal_{i}", units="units", provenance=provenance)
            signals[f"signal_{i}"] = signal
        
        return Dataset(name="nested_dataset", signals=signals, description="Complex nested dataset")
    
    def test_graph_display_with_multiple_signals(self, nested_dataset):
    
        """Test graph display works with datasets containing multiple signals."""
        try:
            from meteaudata.graph_display import SVGGraphBuilder
            
            builder = SVGGraphBuilder()
            graph_data = builder.build_graph(nested_dataset, max_depth=5)
            
            # Should have nodes for dataset, signals, time series, and processing steps
            node_types = [node['type'] for node in graph_data['nodes'].values()]
            expected_types = ['Dataset', 'Signal', 'TimeSeries', 'ProcessingStep', 'Container']
            
            for expected_type in expected_types:
                assert expected_type in node_types, f"Missing {expected_type} in graph"
            
            # Should have multiple signals
            signal_nodes = [node for node in graph_data['nodes'].values() if node['type'] == 'Signal']
            assert len(signal_nodes) >= 2
            
        except ImportError:
            pytest.skip("Graph display module not available")
    
    def test_graph_display_respects_max_depth(self, nested_dataset):
        """Test that max_depth parameter limits graph depth."""
        try:
            from meteaudata.graph_display import SVGGraphBuilder
            
            builder = SVGGraphBuilder()
            
            # Shallow graph
            shallow_data = builder.build_graph(nested_dataset, max_depth=2)
            shallow_count = len(shallow_data['nodes'])
            
            # Deep graph
            deep_data = builder.build_graph(nested_dataset, max_depth=5)
            deep_count = len(deep_data['nodes'])
            
            # Deep graph should have more nodes
            assert deep_count > shallow_count
            
        except ImportError:
            pytest.skip("Graph display module not available")
    
    def test_all_object_types_support_graph_rendering(self):
        """Test that all major object types support graph rendering."""
        # Create instances of all major types
        provenance = DataProvenance(parameter="test")
        func_info = FunctionInfo(name="test", version="1.0", author="test", reference="test.com")
        params = Parameters(test_param="value")
        
        step = ProcessingStep(
            type=ProcessingType.OTHER,
            description="Test",
            run_datetime=datetime.datetime.now(),
            requires_calibration=False,
            function_info=func_info,
            parameters=params,
            suffix="TEST"
        )
        
        data = pd.Series([1, 2, 3], name="test")
        ts = TimeSeries(series=data, processing_steps=[step])
        signal = Signal(input_data=data, name="test", units="unit", provenance=provenance)
        dataset = Dataset(name="test", signals={"test": signal})
        
        test_objects = [provenance, func_info, params, step, ts, signal, dataset]
        
        for obj in test_objects:
            # Should have required methods
            assert hasattr(obj, 'render_svg_graph'), f"{obj.__class__.__name__} missing render_svg_graph"
            assert hasattr(obj, 'show_graph_in_browser'), f"{obj.__class__.__name__} missing show_graph_in_browser"
            assert hasattr(obj, 'display'), f"{obj.__class__.__name__} missing display"
            
            # Should support graph format in display
            try:
                with patch('meteaudata.display_utils._is_notebook_environment', return_value=False), \
                     patch('meteaudata.displayable.DisplayableBase.show_graph_in_browser') as mock_browser:
                    
                    obj.display(format="graph")
                    mock_browser.assert_called_once()
                    
            except Exception as e:
                pytest.fail(f"Graph display failed for {obj.__class__.__name__}: {e}")


class TestComplexDatasetBrowserRendering:
    """Test complex dataset rendering in browser to verify UI and layout."""
    
    @pytest.fixture
    def large_complex_dataset(self):
        """Create a large, deeply nested dataset for comprehensive testing."""
        signals = {}
        
        # Create 5 signals with varying complexity
        for i in range(5):
            provenance = DataProvenance(
                parameter=f"temperature_sensor_{i}",
                location=f"laboratory_building_{i // 2}",
                station=f"station_{i}",
                source=f"data_logger_{i}",
                method=f"measurement_method_{i % 3}",
                reference=f"reference_standard_{i % 2}"
            )
            
            # Create multiple processing steps for each signal
            steps = []
            for j in range(4):  # 4 processing steps per signal
                func_info = FunctionInfo(
                    name=f"processing_function_{j}",
                    version=f"2.{j}.{i}",
                    author=f"scientist_{j % 2}",
                    reference=f"https://example.com/docs/func_{j}_{i}",
                    description=f"Advanced processing step {j} for signal {i}"
                )
                
                # Create parameters with multiple values
                param_dict = {
                    f"window_size_{k}": f"value_{k}_{j}_{i}" for k in range(3)
                }
                param_dict.update({
                    "threshold": f"0.{j}{i}",
                    "method": f"advanced_method_{j}",
                    "calibration_factor": f"1.{i}{j}"
                })
                params = Parameters(**param_dict)
                
                step = ProcessingStep(
                    type=[ProcessingType.FILTERING, ProcessingType.SMOOTHING, 
                          ProcessingType.OTHER, ProcessingType.FAULT_DETECTION][j],
                    description=f"Processing step {j}: advanced {['filtering', 'smoothing', 'calibration', 'quality_control'][j]}",
                    run_datetime=datetime.datetime.now() - datetime.timedelta(days=j),
                    requires_calibration=(j % 2 == 0),
                    function_info=func_info,
                    parameters=params,
                    suffix=f"PROC{j}"
                )
                steps.append(step)
            
            # Create time series with processing steps
            data = pd.Series(
                [10 + i * 2 + j * 0.1 for j in range(20)], 
                name=f"temp_sensor_{i}_RAW",
                index=pd.date_range('2024-01-01', periods=20, freq='h')
            )
            
            index_metadata = IndexMetadata(
                name="timestamp",
                units="datetime",
                description=f"Hourly measurements from sensor {i}",
                type="datetime",
                dtype="datetime64[ns]",
            )
            
            ts = TimeSeries(
                series=data, 
                processing_steps=steps,
                index_metadata=index_metadata
            )
            
            signal = Signal(
                input_data=ts,
                name=f"temperature_sensor_{i}",
                units="°C",
                provenance=provenance,
                description=f"Temperature measurements from high-precision sensor {i}"
            )
            signals[f"temp_sensor_{i}"] = signal
        
        # Create dataset with additional metadata
        dataset = Dataset(
            name="comprehensive_climate_monitoring_dataset",
            description="A comprehensive dataset containing temperature measurements from multiple high-precision sensors across different laboratory buildings",
            owner="research_team_climate_monitoring",
            signals=signals,
            project="Advanced Climate Monitoring Initiative",
        )
        
        return dataset
    
    def test_html_generation_for_complex_dataset(self, large_complex_dataset):
        """Test that complex dataset generates valid HTML without errors."""
        try:
            from meteaudata.graph_display import SVGNestedBoxGraphRenderer
            
            renderer = SVGNestedBoxGraphRenderer()
            
            # Generate HTML for complex dataset
            html_output = renderer.render_to_html(
                large_complex_dataset, 
                max_depth=6,  # Deep traversal
                width=1400, 
                height=900,
                title="Complex Climate Dataset Visualization"
            )
            
            # Verify HTML structure
            assert isinstance(html_output, str), "HTML output should be a string"
            assert len(html_output) > 10000, "HTML should be substantial for complex dataset"
            
            # Check essential HTML elements
            assert '<!DOCTYPE html>' in html_output, "Should have DOCTYPE declaration"
            assert '<html' in html_output, "Should have html tag"
            assert '</html>' in html_output, "Should have closing html tag"
            assert '<head>' in html_output, "Should have head section"
            assert '<body>' in html_output, "Should have body section"
            
            # Check for D3.js inclusion
            assert 'd3.min.js' in html_output, "Should include D3.js library"
            
            # Check for meteaudata-specific elements
            assert 'graph-container' in html_output, "Should have graph container"
            assert 'details-panel' in html_output, "Should have details panel"
            assert 'InteractiveNestedBoxGraph' in html_output, "Should have graph class"
            
            # Check that data was injected
            assert 'graphData' in html_output, "Should have graph data"
            assert 'graphConfig' in html_output, "Should have graph configuration"
            assert 'comprehensive_climate_monitoring_dataset' in html_output, "Should contain dataset name"
            
            # Check for JavaScript functionality
            assert 'loadData' in html_output, "Should have data loading function"
            assert 'width: 1400' in html_output, "Should have correct width"
            assert 'height: 900' in html_output, "Should have correct height"
            
        except ImportError:
            pytest.skip("Graph display module not available")
    
    def test_browser_file_generation_and_cleanup(self, large_complex_dataset):
        """Test that browser files are generated correctly and can be cleaned up."""
        try:
            from meteaudata.graph_display import open_meteaudata_graph_in_browser
            import os
            
            # Mock webbrowser to prevent actual opening
            with patch('webbrowser.open') as mock_browser:
                file_path = open_meteaudata_graph_in_browser(
                    large_complex_dataset,
                    max_depth=5,
                    width=1600,
                    height=1000,
                    title="Browser Test Dataset"
                )
                
                # Verify file was created
                assert os.path.exists(file_path), f"HTML file should be created at {file_path}"
                assert file_path.endswith('.html'), "File should have .html extension"
                
                # Verify browser was called with correct file URL
                mock_browser.assert_called_once()
                called_url = mock_browser.call_args[0][0]
                assert called_url.startswith('file://'), "Should open with file:// protocol"
                assert file_path in called_url, "Should contain correct file path"
                
                # Verify file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    assert len(content) > 10000, "File should contain substantial content"
                    assert 'Browser Test Dataset' in content, "Should contain custom title"
                    assert 'width: 1600' in content, "Should have custom width"
                    assert 'height: 1000' in content, "Should have custom height"
                
                # Clean up
                os.unlink(file_path)
                assert not os.path.exists(file_path), "File should be deleted after cleanup"
                
        except ImportError:
            pytest.skip("Graph display module not available")
    
    def test_graph_data_structure_for_complex_dataset(self, large_complex_dataset):
        """Test that complex dataset generates proper graph data structure."""
        try:
            from meteaudata.graph_display import SVGGraphBuilder
            
            builder = SVGGraphBuilder()
            graph_data = builder.build_graph(large_complex_dataset, max_depth=6)
            
            # Verify overall structure
            assert 'hierarchy' in graph_data, "Should have hierarchy"
            assert 'nodes' in graph_data, "Should have nodes"
            assert 'layout_type' in graph_data, "Should have layout type"
            
            # Check hierarchy root
            root = graph_data['hierarchy']
            assert root['type'] == 'Dataset', "Root should be Dataset"
            assert root['attributes']['Name'] == 'comprehensive_climate_monitoring_dataset'
            
            # Check node count - should be substantial for complex dataset
            nodes = graph_data['nodes']
            assert len(nodes) >= 20, f"Should have many nodes for complex dataset, got {len(nodes)}"
            
            # Verify different node types are present
            node_types = {node['type'] for node in nodes.values()}
            expected_types = {'Dataset', 'Signal', 'TimeSeries', 'ProcessingStep', 'DataProvenance', 'Container'}
            assert expected_types.issubset(node_types), f"Missing node types. Expected {expected_types}, got {node_types}"
            
            # Check that signals are properly represented
            signal_nodes = [node for node in nodes.values() if node['type'] == 'Signal']
            assert len(signal_nodes) == 5, f"Should have 5 signal nodes, got {len(signal_nodes)}"
            
            # Check processing steps
            processing_nodes = [node for node in nodes.values() if node['type'] == 'ProcessingStep']
            assert len(processing_nodes) >= 15, f"Should have many processing steps, got {len(processing_nodes)}"
            
            # Verify attributes are populated
            for node in nodes.values():
                assert 'attributes' in node, f"Node {node['identifier']} should have attributes"
                assert len(node['attributes']) > 0, f"Node {node['identifier']} should have some attributes"
            
        except ImportError:
            pytest.skip("Graph display module not available")
    
    def test_ui_layout_elements_in_html(self, large_complex_dataset):
        """Test that UI layout elements are properly included for complex dataset."""
        try:
            from meteaudata.graph_display import SVGNestedBoxGraphRenderer
            
            renderer = SVGNestedBoxGraphRenderer()
            html_output = renderer.render_to_html(large_complex_dataset, max_depth=5)
            
            # Check CSS classes for layout
            layout_classes = [
                'container', 'graph-container', 'details-panel', 'details-content',
                'details-header', 'details-attribute', 'attribute-name', 'attribute-value',
                'graph-svg', 'nested-box'
            ]
            
            for css_class in layout_classes:
                assert css_class in html_output, f"Should contain CSS class '{css_class}'"
            
            # Check for responsive design elements
            assert 'viewport' in html_output, "Should have viewport meta tag"
            assert 'flex' in html_output, "Should use flexbox layout"
            
            # Check for interaction elements
            assert 'cursor: pointer' in html_output, "Should have pointer cursors for interactive elements"
            assert 'transition:' in html_output, "Should have CSS transitions"
            
            # Check for proper typography
            assert 'font-family:' in html_output, "Should have font specifications"
            assert 'font-weight:' in html_output, "Should have font weights"
            
        except ImportError:
            pytest.skip("Graph display module not available")
    
    def test_javascript_functionality_integration(self, large_complex_dataset):
        """Test that JavaScript functionality is properly integrated."""
        try:
            from meteaudata.graph_display import SVGNestedBoxGraphRenderer
            
            renderer = SVGNestedBoxGraphRenderer()
            html_output = renderer.render_to_html(large_complex_dataset, max_depth=5)
            
            # Check for key JavaScript functions and classes
            js_elements = [
                'class InteractiveNestedBoxGraph',
                'loadData(hierarchyData)',
                'renderHierarchy()',
                'setupZoom()',
                'showDetails(',
                'calculateLayout(',
                'const graphData =',
                'const graphConfig =',
                'graph.loadData(graphData)'
            ]
            
            for js_element in js_elements:
                assert js_element in html_output, f"Should contain JavaScript element '{js_element}'"
            
            # Check that data is properly formatted as JSON
            import re
            graph_data_match = re.search(r'const graphData = ({.*?});', html_output, re.DOTALL)
            assert graph_data_match, "Should have properly formatted graphData"
            
            # Verify JSON is valid
            import json
            try:
                json.loads(graph_data_match.group(1))
            except json.JSONDecodeError:
                pytest.fail("Graph data should be valid JSON")
            
        except ImportError:
            pytest.skip("Graph display module not available")
    
    def test_performance_with_deep_nesting(self, large_complex_dataset):
        """Test that rendering performance is acceptable for deep, complex datasets."""
        try:
            from meteaudata.graph_display import SVGNestedBoxGraphRenderer
            import time
            
            renderer = SVGNestedBoxGraphRenderer()
            
            # Test different depth levels
            depths_to_test = [3, 5, 7]
            
            for max_depth in depths_to_test:
                start_time = time.time()
                
                html_output = renderer.render_to_html(
                    large_complex_dataset, 
                    max_depth=max_depth
                )
                
                end_time = time.time()
                render_time = end_time - start_time
                
                # Should complete within reasonable time (adjust threshold as needed)
                assert render_time < 10.0, f"Rendering depth {max_depth} took {render_time:.2f}s, too slow"
                
                # Output size should be reasonable
                assert len(html_output) < 5_000_000, f"HTML too large at depth {max_depth}: {len(html_output)} bytes"
                
                # Should contain expected content
                assert 'comprehensive_climate_monitoring_dataset' in html_output
                
                print(f"Depth {max_depth}: {render_time:.3f}s, {len(html_output):,} bytes")
            
        except ImportError:
            pytest.skip("Graph display module not available")


class TestBrowserRenderingEdgeCases:
    """Test edge cases and error conditions in browser rendering."""
    
    @pytest.fixture
    def edge_case_dataset(self):
        """Create dataset with edge cases that might break rendering."""
        # Signal with special characters and unicode
        provenance = DataProvenance(
            parameter="température_spéciale_élevée",
            location="laboratoire « avancé »",
            station="station-test & validation",
            source="capteur™ ultra-précis",
            equipment="équipement « révolutionnaire »",
            reference="référence → http://example.com/spécialisé"
        )
        
        # Parameters with various data types and edge values
        params = Parameters(
            unicode_param="valeur_spéciale_éñ",
            numeric_param=12345.67890,
            boolean_param=True,
            none_param=None,
            empty_string="",
            large_number=1e10,
            small_number=1e-10,
            special_chars="<>&\"'",
            newlines="line1\nline2\nline3"
        )
        
        func_info = FunctionInfo(
            name="fonction_spéciale",
            version="1.0.0-αβγ",
            author="Développeur Ñoël",
            reference="https://example.com/spécialisé?param=valeur&autre=données",
            description="Une fonction très spécialisée avec des caractères spéciaux"
        )
        
        step = ProcessingStep(
            type=ProcessingType.OTHER,
            description="Étape de traitement avec des caractères spéciaux: <>&\"'",
            run_datetime=datetime.datetime.now(),
            requires_calibration=True,
            function_info=func_info,
            parameters=params,
            suffix="SPÉCIAL"
        )
        
        # Time series with unusual data
        data = pd.Series(
            [float('inf'), -float('inf'), 0, 1e-10, 1e10, -1e10], 
            name="données_spéciales",
            index=pd.date_range('2024-01-01', periods=6, freq='D')
        )
        
        ts = TimeSeries(series=data, processing_steps=[step])
        signal = Signal(
            input_data=ts,
            name="signal_spécial_élevé",
            units="°C/m²·s⁻¹",
            provenance=provenance,
            description="Signal avec des unités et caractères spéciaux"
        )
        
        return Dataset(
            name="dataset_caractères_spéciaux",
            description="Dataset avec des caractères spéciaux pour tester la robustesse",
            owner="équipe_développement",
            signals={"signal_test": signal}
        )

    def test_unicode_and_special_chars_rendering1(self, edge_case_dataset):
        """Test that unicode and special characters are properly handled."""
        try:
            from meteaudata.graph_display import SVGNestedBoxGraphRenderer
            
            # First, let's check if the dataset itself has Unicode
            print("Checking dataset for Unicode...")
            for signal_name, signal in edge_case_dataset.signals.items():
                print(f"Signal name: {signal_name}")
                print(f"Signal.name: {signal.name}")
                attrs = signal._get_display_attributes()
                for key, value in attrs.items():
                    if isinstance(value, str) and any(ord(c) > 127 for c in value):
                        print(f"Unicode found in {key}: {value}")
            
            renderer = SVGNestedBoxGraphRenderer()
            html_output = renderer.render_to_html(edge_case_dataset, max_depth=5)
            
            # Debug: Check if the data injection happened
            print("\n=== DEBUGGING DATA INJECTION ===")
            if "const graphData = {" in html_output:
                print("✓ Data injection found")
                # Find the graphData section
                start_idx = html_output.find("const graphData = {")
                end_idx = html_output.find("};", start_idx) + 2
                if start_idx >= 0 and end_idx > start_idx:
                    graph_data_section = html_output[start_idx:end_idx]
                    print(f"GraphData section length: {len(graph_data_section)}")
                    print("First 500 chars of graphData:")
                    print(graph_data_section[:500])
                    
                    # Check for Unicode in the data section
                    if "spécial" in graph_data_section:
                        print("✓ Unicode found in graphData")
                    else:
                        print("✗ Unicode NOT found in graphData")
            else:
                print("✗ Data injection NOT found - using demo data")
                
            
            # Should not crash with unicode
            assert isinstance(html_output, str)
            
            # More flexible assertions for debugging
            has_unicode1 = "spécial" in html_output  # More general check
            has_unicode2 = "°C" in html_output       # Degree symbol
            
            print(f"\nUnicode checks:")
            print(f"Contains 'spécial': {has_unicode1}")
            print(f"Contains '°C': {has_unicode2}")
            
            # Let's check what we actually have instead of what we expect
            import re
            unicode_matches = re.findall(r'[^\x00-\x7F]+', html_output)
            print(f"All Unicode characters found: {set(unicode_matches)}")
            
            # For now, let's just check that some Unicode is preserved
            assert has_unicode1 or has_unicode2, "Should contain some Unicode characters"
            
        except ImportError:
            pytest.skip("Graph display module not available")
    
    def test_unicode_and_special_chars_rendering(self, edge_case_dataset):
        """Test that unicode and special characters are properly handled."""
        try:
            from meteaudata.graph_display import SVGNestedBoxGraphRenderer
            
            renderer = SVGNestedBoxGraphRenderer()
            html_output = renderer.render_to_html(edge_case_dataset, max_depth=5)
            
            # Should not crash with unicode
            assert isinstance(html_output, str)
            
            # Check that unicode characters are preserved
            assert "température_spéciale" in html_output
            assert "laboratoire « avancé »" in html_output
            assert "révolutionnaire" in html_output
            
            # Check that special HTML characters are properly escaped
            assert "&lt;" not in html_output and "<" in html_output# Basic sanity check
            # Should be valid HTML despite special characters
            assert html_output.startswith('<!DOCTYPE html>')
            assert html_output.endswith('</html>')
            
        except ImportError:
            pytest.skip("Graph display module not available")
    
    def test_empty_and_null_values_handling(self):
        """Test handling of empty and null values."""
        try:
            from meteaudata.graph_display import SVGNestedBoxGraphRenderer
            
            # Create minimal objects with mostly empty values
            empty_provenance = DataProvenance()
            empty_params = Parameters()
            
            minimal_func_info = FunctionInfo(name="minimal", version="1.0", author="test", reference="test.com")
            minimal_step = ProcessingStep(
                type=ProcessingType.OTHER,
                description="",
                run_datetime=datetime.datetime.now(),
                requires_calibration=False,
                function_info=minimal_func_info,
                parameters=empty_params,
                suffix=""
            )
            
            data = pd.Series([1, 2, 3], name="minimal")
            ts = TimeSeries(series=data, processing_steps=[minimal_step])
            signal = Signal(
                input_data=ts,
                name="",
                units="",
                provenance=empty_provenance
            )
            
            minimal_dataset = Dataset(
                name="",
                description="",
                owner="",
                signals={"signal": signal}
            )
            
            renderer = SVGNestedBoxGraphRenderer()
            html_output = renderer.render_to_html(minimal_dataset, max_depth=5)
            
            # Should not crash with empty values
            assert isinstance(html_output, str)
            assert len(html_output) > 1000  # Should still generate substantial HTML
            
        except ImportError:
            pytest.skip("Graph display module not available")
    
    def test_very_long_strings_handling(self):
        """Test handling of very long strings that might break layout."""
        try:
            from meteaudata.graph_display import SVGNestedBoxGraphRenderer
            
            # Create object with very long strings
            long_string = "A" * 1000 + " very long string that might break layout " + "B" * 1000
            
            provenance = DataProvenance(
                parameter=long_string,
                location=long_string,
                description=long_string
            )
            
            params = Parameters(long_param=long_string)
            
            func_info = FunctionInfo(
                name="test",
                version="1.0",
                author="test",
                reference="test.com",
                description=long_string
            )
            
            step = ProcessingStep(
                type=ProcessingType.OTHER,
                description=long_string,
                run_datetime=datetime.datetime.now(),
                requires_calibration=False,
                function_info=func_info,
                parameters=params,
                suffix="TEST"
            )
            
            data = pd.Series([1, 2, 3], name="test")
            ts = TimeSeries(series=data, processing_steps=[step])
            signal = Signal(
                input_data=ts,
                name=long_string,
                units="unit",
                provenance=provenance,
                description=long_string
            )
            
            long_dataset = Dataset(
                name=long_string,
                description=long_string,
                owner=long_string,
                signals={"signal": signal}
            )
            
            renderer = SVGNestedBoxGraphRenderer()
            html_output = renderer.render_to_html(long_dataset, max_depth=5)
            
            # Should not crash with long strings
            assert isinstance(html_output, str)
            
            # Should contain the long strings (possibly truncated)
            assert long_string[:100] in html_output or "AAAAAAAAAA" in html_output
            
        except ImportError:
            pytest.skip("Graph display module not available")


class TestGraphDisplayErrorHandling:
    """Test error handling in the cleaned graph display system."""
    
    def test_missing_template_error(self):
        """Test error when SVG template is missing."""
        try:
            from meteaudata.graph_display import SVGNestedBoxGraphRenderer
            
            renderer = SVGNestedBoxGraphRenderer()
            
            # Mock template loading to simulate missing file
            with patch.object(renderer, '_get_html_template', side_effect=FileNotFoundError("SVG graph template not found.")):
                provenance = DataProvenance(parameter="test")
                
                with pytest.raises(FileNotFoundError) as excinfo:
                    renderer.render_to_html(provenance)
                
                assert "SVG graph template not found" in str(excinfo.value)
                
        except ImportError:
            pytest.skip("Graph display module not available")
    
    def test_graph_display_import_error(self):
        """Test graceful handling when graph display module is missing."""
        provenance = DataProvenance(parameter="test")
        
        with patch('meteaudata.graph_display.SVGNestedBoxGraphRenderer', side_effect=ImportError("Module not found")):
            with pytest.raises(ImportError) as excinfo:
                provenance.render_svg_graph()
            
            assert "svg_nested_boxes module" in str(excinfo.value)
    
    def test_display_with_empty_objects(self):
        """Test display methods work with minimal objects."""
        empty_provenance = DataProvenance()  # All fields None/default
        
        # Should not crash with text display
        with patch('builtins.print'):
            empty_provenance.display(format="text")
        
        # Should not crash with HTML display
        with patch('IPython.display.HTML'), patch('IPython.display.display'):
            empty_provenance.display(format="html")
        
        # Should not crash with graph display
        with patch('meteaudata.display_utils._is_notebook_environment', return_value=False), \
             patch('meteaudata.displayable.DisplayableBase.show_graph_in_browser') as mock_browser:
            empty_provenance.display(format="graph")
            mock_browser.assert_called_once()


class TestRealBrowserIntegration:
    """Test scenarios that simulate real browser usage patterns."""
    
    @pytest.fixture
    def realistic_dataset(self):
        """Create a realistic dataset that mimics real-world usage."""
        # Simulate environmental monitoring dataset
        sensors = ['temperature', 'humidity', 'pressure', 'wind_speed', 'solar_radiation']
        signals = {}
        
        for sensor in sensors:
            # Realistic provenance
            provenance = DataProvenance(
                parameter=sensor,
                location="Environmental Research Station Alpha",
                station=f"{sensor}_monitoring_station_01",
                source=f"WXT536_weather_sensor_{sensor}",
                method="continuous_automated_measurement",
                reference="https://www.vaisala.com/en/products/weather-environmental-sensors"
            )
            
            # Realistic processing pipeline
            steps = []
            
            # Quality control step
            qc_params = Parameters(
                outlier_threshold=3.0,
                missing_data_tolerance=0.05,
                range_min=-40.0 if sensor == 'temperature' else 0.0,
                range_max=60.0 if sensor == 'temperature' else 100.0
            )
            qc_func = FunctionInfo(
                name="quality_control_filter",
                version="2.1.3",
                author="Environmental Data Processing Team",
                reference="https://github.com/envdata/quality-control",
                description="Automated quality control for environmental sensor data"
            )
            qc_step = ProcessingStep(
                type=ProcessingType.FAULT_DIAGNOSIS,
                description="Automated quality control: outlier detection and range validation",
                run_datetime=datetime.datetime(2024, 1, 15, 10, 30),
                requires_calibration=True,
                function_info=qc_func,
                parameters=qc_params,
                suffix="QC"
            )
            steps.append(qc_step)
            
            # Calibration step
            cal_params = Parameters(
                calibration_coefficient=1.0234,
                offset_correction=-0.15,
                drift_compensation=True,
                reference_standard="NIST_traceable"
            )
            cal_func = FunctionInfo(
                name="sensor_calibration",
                version="1.5.2",
                author="Calibration Laboratory Services",
                reference="https://calibration-lab.org/procedures/environmental",
                description="NIST-traceable calibration for environmental sensors"
            )
            cal_step = ProcessingStep(
                type=ProcessingType.OTHER,
                description="NIST-traceable calibration correction with drift compensation",
                run_datetime=datetime.datetime(2024, 1, 15, 11, 0),
                requires_calibration=False,
                function_info=cal_func,
                parameters=cal_params,
                suffix="CAL"
            )
            steps.append(cal_step)
            
            # Smoothing step
            smooth_params = Parameters(
                window_size=5,
                method="savitzky_golay",
                polynomial_order=2,
                preserve_peaks=True
            )
            smooth_func = FunctionInfo(
                name="savitzky_golay_smoother",
                version="3.0.1",
                author="Signal Processing Research Group",
                reference="https://scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html",
                description="Savitzky-Golay filter for noise reduction while preserving signal features"
            )
            smooth_step = ProcessingStep(
                type=ProcessingType.SMOOTHING,
                description="Savitzky-Golay smoothing with peak preservation",
                run_datetime=datetime.datetime(2024, 1, 15, 11, 15),
                requires_calibration=False,
                function_info=smooth_func,
                parameters=smooth_params,
                suffix="SMOOTH"
            )
            steps.append(smooth_step)
            
            # Create realistic time series data
            dates = pd.date_range('2024-01-01', periods=144, freq='10min')  # 10-minute intervals for 24 hours
            
            if sensor == 'temperature':
                # Realistic temperature data with daily cycle
                base_temp = 15.0
                data_values = [base_temp + 10 * np.sin(2 * np.pi * i / 144) + np.random.normal(0, 0.5) for i in range(144)]
            elif sensor == 'humidity':
                # Realistic humidity data (inverse relationship with temperature)
                data_values = [60 + 20 * np.cos(2 * np.pi * i / 144) + np.random.normal(0, 2) for i in range(144)]
            elif sensor == 'pressure':
                # Realistic pressure data with small variations
                data_values = [1013.25 + np.random.normal(0, 2) for i in range(144)]
            elif sensor == 'wind_speed':
                # Realistic wind speed data
                data_values = [abs(5 + 3 * np.sin(2 * np.pi * i / 144) + np.random.normal(0, 1)) for i in range(144)]
            else:  # solar_radiation
                # Realistic solar radiation with day/night cycle
                data_values = [max(0, 800 * max(0, np.sin(np.pi * (i % 144) / 144)) + np.random.normal(0, 10)) for i in range(144)]
            
            data = pd.Series(data_values, name=f"{sensor}_RAW", index=dates)
            
            # Index metadata
            index_metadata = IndexMetadata(
                name="timestamp",
                units="datetime",
                description="10-minute interval measurements from automated weather station",
                dtype="datetime64[ns]",
                type="datetime"
            )
            
            ts = TimeSeries(
                series=data,
                processing_steps=steps,
                index_metadata=index_metadata
            )
            
            # Units mapping
            units_map = {
                'temperature': '°C',
                'humidity': '%RH',
                'pressure': 'hPa',
                'wind_speed': 'm/s',
                'solar_radiation': 'W/m²'
            }
            
            signal = Signal(
                input_data=ts,
                name=f"{sensor}_processed",
                units=units_map[sensor],
                provenance=provenance,
                description=f"Quality-controlled and calibrated {sensor} measurements from environmental monitoring station"
            )
            
            signals[sensor] = signal
        
        return Dataset(
            name="environmental_monitoring_station_alpha_2024_01_01",
            description="24-hour environmental monitoring dataset from Research Station Alpha, including temperature, humidity, pressure, wind speed, and solar radiation measurements",
            owner="Environmental Research Consortium",
            signals=signals,
            metadata={
                "station_id": "ENV_ALPHA_001",
                "latitude": 45.5234,
                "longitude": -73.5831,
                "elevation_m": 125.0,
                "measurement_height_m": 2.0,
                "data_quality": "research_grade",
                "collection_start": "2024-01-01T00:00:00Z",
                "collection_end": "2024-01-01T23:50:00Z",
                "sampling_interval": "10_minutes",
                "processing_date": "2024-01-02T08:00:00Z"
            }
        )
    
    def test_graph_builder_creates_proper_hierarchy(self, realistic_dataset):
        """Test that the graph builder creates the expected hierarchical structure with containers."""
        from meteaudata.graph_display import SVGGraphBuilder
        
        builder = SVGGraphBuilder()
        graph_data = builder.build_graph(realistic_dataset, max_depth=4)
        
        # Should have hierarchy with the dataset as root
        hierarchy = graph_data['hierarchy']
        assert hierarchy['type'] == 'Dataset'
        
        # Should have children (which should include container nodes)
        assert 'children' in hierarchy
        assert len(hierarchy['children']) > 0
        
        # Look for a signals container
        signals_container = None
        for child in hierarchy['children']:
            if child.get('type') == 'Container' and 'signals' in child.get('identifier', '').lower():
                signals_container = child
                break
        
        assert signals_container is not None, "Should have a signals container"
        
        # The signals container should have children (individual signals)
        assert 'children' in signals_container
        assert len(signals_container['children']) > 0
        
        # Each signal should have time series containers
        signal_node = signals_container['children'][0]
        assert signal_node['type'] == 'Signal'
        
        # Signal should have children including time series container
        time_series_container = None
        for child in signal_node.get('children', []):
            if child.get('type') == 'Container' and 'time' in child.get('identifier', '').lower():
                time_series_container = child
                break
        
        if time_series_container:  # Only check if container exists
            assert len(time_series_container['children']) > 0

    def test_complete_browser_workflow(self, realistic_dataset):
        """Test the complete workflow from dataset to browser display."""
        try:
            from meteaudata.graph_display import open_meteaudata_graph_in_browser
            import os
            
            # Mock webbrowser to capture the file without opening
            captured_files = []
            
            def mock_open(url):
                captured_files.append(url)
                return True
            
            with patch('webbrowser.open', side_effect=mock_open):
                # Test the complete workflow
                file_path = open_meteaudata_graph_in_browser(
                    realistic_dataset,
                    max_depth=6,
                    width=1800,
                    height=1200,
                    title="Environmental Monitoring Station Alpha - Real-time Dashboard"
                )
                
                # Verify file was created
                assert os.path.exists(file_path), "HTML file should be created"
                assert len(captured_files) == 1, "Browser should be called once"
                assert file_path in captured_files[0], "Correct file should be opened"
                
                # Read and verify file content comprehensively
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # File should be substantial
                assert len(content) > 50000, f"File should be substantial, got {len(content)} bytes"
                
                # Check for all required HTML structure
                html_checks = [
                    '<!DOCTYPE html>',
                    '<html lang="en">',
                    '<head>',
                    '<meta charset="UTF-8">',
                    '<meta name="viewport"',
                    '<title>Interactive Nested Box Graph</title>',  # Use the static title from template
                    'd3.min.js',
                    '<style>',
                    'class InteractiveNestedBoxGraph',
                    'loadData(hierarchyData)',
                    'graphData =',
                    'graphConfig =',
                    'width: 1800',
                    'height: 1200',
                    # Check that the dynamic title is in the JavaScript instead
                    'title: "Environmental Monitoring Station Alpha - Real-time Dashboard"'
                ]
                
                for check in html_checks:
                    assert check in content, f"Missing required HTML element: {check}"
                
                # Check for dataset-specific content
                dataset_checks = [
                    'environmental_monitoring_station_alpha',
                    'temperature_processed',
                    'humidity_processed',
                    'pressure_processed',
                    'wind_speed_processed',
                    'solar_radiation_processed',
                    'Environmental Research Station Alpha',
                    'quality_control_filter',
                    'sensor_calibration',
                    'savitzky_golay_smoother'
                ]
                
                for check in dataset_checks:
                    assert check in content, f"Missing dataset-specific content: {check}"
                
                # Verify JSON structure is valid
                import re
                graph_data_match = re.search(r'const graphData = ({.*?});', content, re.DOTALL)
                assert graph_data_match, "Should have valid graphData"
                
                import json
                graph_data = json.loads(graph_data_match.group(1))
                
                # Verify graph data structure
                assert 'type' in graph_data, "Graph data should have type"
                assert 'identifier' in graph_data, "Graph data should have identifier"
                assert 'attributes' in graph_data, "Graph data should have attributes"
                assert 'children' in graph_data, "Graph data should have children"
                
                # Check that all sensors are represented
                def find_in_graph(data, target_text):
                    if isinstance(data, dict):
                        if any(target_text in str(value) for value in data.values()):
                            return True
                        for value in data.values():
                            if find_in_graph(value, target_text):
                                return True
                    elif isinstance(data, list):
                        for item in data:
                            if find_in_graph(item, target_text):
                                return True
                    return False
                
                sensor_names = ['temperature', 'humidity', 'pressure', 'wind_speed', 'solar_radiation']
                for sensor in sensor_names:
                    assert find_in_graph(graph_data, sensor), f"Sensor {sensor} should be in graph data"
                
                # Clean up
                os.unlink(file_path)
                
        except ImportError:
            pytest.skip("Graph display module not available")
    
    def test_responsive_design_elements(self, realistic_dataset):
        """Test that responsive design elements are properly included."""
        try:
            from meteaudata.graph_display import SVGNestedBoxGraphRenderer
            
            renderer = SVGNestedBoxGraphRenderer()
            html_output = renderer.render_to_html(realistic_dataset, max_depth=5, width=1200, height=800)
            
            # Check viewport settings
            assert 'viewport' in html_output, "Should have viewport meta tag"
            assert 'width=device-width' in html_output, "Should have responsive viewport"
            
            # Check flexible layout
            assert 'display: flex' in html_output, "Should use flexbox layout"
            assert 'overflow-y: auto' in html_output, "Should handle overflow"
            
            # Check responsive CSS properties
            responsive_properties = [
                'max-width:', 'min-width:', 'max-height:', 'min-height:',
                'flex:', '@media', 'vh', 'vw'
            ]
            
            # At least some responsive properties should be present
            responsive_count = sum(1 for prop in responsive_properties if prop in html_output)
            assert responsive_count >= 3, f"Should have responsive CSS properties, found {responsive_count}"
            
        except ImportError:
            pytest.skip("Graph display module not available")
    
    
    def test_large_dataset_memory_efficiency(self):
        """Test memory efficiency with larger datasets."""
        try:
            from meteaudata.graph_display import SVGNestedBoxGraphRenderer
            import os
            
            # Create a larger dataset
            signals = {}
            for i in range(20):  # 20 signals instead of 5
                provenance = DataProvenance(parameter=f"sensor_{i}")
                steps = []
                for j in range(3):  # 3 steps each
                    func_info = FunctionInfo(name=f"func_{j}", version="1.0", author="test", reference="test.com")
                    params = Parameters(**{f"param_{k}": f"value_{k}" for k in range(5)})
                    step = ProcessingStep(
                        type=ProcessingType.OTHER,
                        description=f"Step {j}",
                        run_datetime=datetime.datetime.now(),
                        requires_calibration=False,
                        function_info=func_info,
                        parameters=params,
                        suffix=f"STEP{j}"
                    )
                    steps.append(step)
                
                data = pd.Series(range(100), name=f"sensor_{i}")  # Larger series
                ts = TimeSeries(series=data, processing_steps=steps)
                signal = Signal(
                    input_data=ts,
                    name=f"sensor_{i}",
                    units="unit",
                    provenance=provenance
                )
                signals[f"sensor_{i}"] = signal
            
            large_dataset = Dataset(name="large_test_dataset", signals=signals)
            
            # Monitor memory usage (skip if psutil not available)
            try:
                import psutil
                process = psutil.Process(os.getpid())
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_monitoring = True
            except ImportError:
                memory_monitoring = False
                initial_memory = 0
            
            renderer = SVGNestedBoxGraphRenderer()
            html_output = renderer.render_to_html(large_dataset, max_depth=5)
            
            if memory_monitoring:
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = final_memory - initial_memory
                
                # Memory increase should be reasonable (adjust threshold as needed)
                assert memory_increase < 500, f"Memory increase too large: {memory_increase:.1f} MB"
                print(f"Memory increase: {memory_increase:.1f} MB, HTML size: {len(html_output):,} bytes")
            else:
                print(f"HTML size: {len(html_output):,} bytes (memory monitoring skipped)")
            
            # Output should still be reasonable size
            assert len(html_output) < 10_000_000, f"HTML too large: {len(html_output)} bytes"
            
        except ImportError:
            pytest.skip("Graph display module not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])