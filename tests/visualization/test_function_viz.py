import pytest
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from algogym.visualization import FunctionVisualizer
from algogym.functions import BaseFunction
from algogym.algorithms import BaseAlgorithm


# Simple test function and algorithm implementations
class TestFunction(BaseFunction):
    """Simple test function that returns x^2."""
    
    def __init__(self, input_dim=1, output_dim=1):
        domain = (-5.0, 5.0) if input_dim == 1 else ([-5.0] * input_dim, [5.0] * input_dim)
        super().__init__(input_dim=input_dim, output_dim=output_dim, domain=domain)
    
    def __call__(self, x):
        x_val = self._validate_input(x)
        if self.input_dim == 1:
            return x_val ** 2
        else:
            # For multi-dimensional input, return sum of squares
            return np.sum(x_val ** 2, axis=1, keepdims=True)


class TestAlgorithm(BaseAlgorithm):
    """Simple test algorithm that approximates a function with a polynomial."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.input_dim = 1  # Only supports 1D for testing
        self._approximated_function = None
    
    def _validate_config(self, config):
        pass
    
    def train(self, target_function=None, X_data=None, y_data=None):
        # Just store the function for testing purposes
        self._target_function = target_function
    
    def predict(self, x):
        # Return an approximation (x^2 + 0.5 for testing)
        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                return x ** 2 + 0.5
            else:
                return x[:, 0] ** 2 + 0.5
        else:
            return x ** 2 + 0.5


# Fixture for a function visualizer
@pytest.fixture
def function_visualizer():
    return FunctionVisualizer()


# Fixture for a simple test function
@pytest.fixture
def test_function():
    return TestFunction()


# Fixture for a simple test algorithm
@pytest.fixture
def test_algorithm():
    return TestAlgorithm()


# Test function visualization creation
def test_visualize_function_1d(function_visualizer, test_function, test_algorithm):
    """Test visualization of a 1D function."""
    # Turn off interactive mode to avoid showing plots during tests
    plt.ioff()
    
    # Train the algorithm
    test_algorithm.train(target_function=test_function)
    
    # Visualize without showing (show=False)
    fig = function_visualizer.visualize_function(
        function=test_function,
        approximation=test_algorithm,
        title="Test Function",
        show=False
    )
    
    # Check that a figure was returned
    assert isinstance(fig, plt.Figure)
    
    # Close the figure to clean up
    plt.close(fig)


# Test progress visualization
def test_visualize_progress(function_visualizer):
    """Test visualization of algorithm progress."""
    # Turn off interactive mode to avoid showing plots during tests
    plt.ioff()
    
    # Create sample metrics history
    metrics_history = {
        'MSE': [0.5, 0.4, 0.3, 0.2, 0.1],
        'MAE': [0.4, 0.3, 0.2, 0.1, 0.05]
    }
    
    # Visualize without showing (show=False)
    fig = function_visualizer.visualize_progress(
        metrics_history=metrics_history,
        title="Test Progress",
        show=False
    )
    
    # Check that a figure was returned
    assert isinstance(fig, plt.Figure)
    
    # Close the figure to clean up
    plt.close(fig)


# Test comparison visualization
def test_visualize_comparison(function_visualizer, test_function):
    """Test visualization of algorithm comparison."""
    # Turn off interactive mode to avoid showing plots during tests
    plt.ioff()
    
    # Create multiple algorithms with different behaviors
    algo1 = TestAlgorithm()
    algo2 = TestAlgorithm()
    
    # Train the algorithms
    algo1.train(target_function=test_function)
    algo2.train(target_function=test_function)
    
    # Visualize without showing (show=False)
    fig = function_visualizer.visualize_comparison(
        functions=[test_function, test_function],
        approximations=[algo1, algo2],
        labels=["Algorithm 1", "Algorithm 2"],
        title="Test Comparison",
        show=False
    )
    
    # Check that a figure was returned
    assert isinstance(fig, plt.Figure)
    
    # Close the figure to clean up
    plt.close(fig)


# Test error handling for unsupported dimensions
def test_visualize_function_unsupported_dims(function_visualizer):
    """Test that trying to visualize a high-dimensional function raises an error."""
    # Create a 3D function
    high_dim_func = TestFunction(input_dim=3, output_dim=1)
    
    # Expect a ValueError
    with pytest.raises(ValueError):
        function_visualizer.visualize_function(
            function=high_dim_func,
            show=False
        )


# Test configuring the visualizer
def test_visualizer_config():
    """Test that configuration options are applied correctly."""
    custom_config = {
        'figsize': (12, 8),
        'dpi': 150,
        'cmap': 'plasma',
        'n_samples': 500
    }
    
    # Create a visualizer with custom config
    visualizer = FunctionVisualizer(config=custom_config)
    
    # Check that config was applied
    assert visualizer.config['figsize'] == (12, 8)
    assert visualizer.config['dpi'] == 150
    assert visualizer.config['cmap'] == 'plasma'
    assert visualizer.config['n_samples'] == 500 