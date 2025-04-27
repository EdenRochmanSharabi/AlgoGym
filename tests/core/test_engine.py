import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call

import sys
import os
# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from algogym.core.engine import ExperimentEngine
from algogym.functions.base import BaseFunction
from algogym.algorithms.base import BaseAlgorithm
from algogym.data.base import BaseDataLoader
from algogym.evaluation.metrics import mean_squared_error

# --- Mocks and Fixtures ---

@pytest.fixture
def mock_function():
    func = MagicMock(spec=BaseFunction)
    func.input_dim = 1
    func.output_dim = 1
    func.domain = (np.array([-1.0]), np.array([1.0]))
    # Make sample_domain return predictable shapes
    def sample_domain_mock(n_samples):
        return np.linspace(-1, 1, n_samples).reshape(n_samples, 1)
    func.sample_domain.side_effect = sample_domain_mock
    # Make call return predictable shape
    def call_mock(x):
         # Ensure input is 2D for processing
        if x.ndim == 1:
             x_proc = x.reshape(-1, 1)
        else:
             x_proc = x
        # Return shape (N, 1)
        y = x_proc * 2 
        # Match expected return shape based on original input shape
        if x.ndim <= 1:
             return y.flatten() # Return (N,) or scalar
        else:
             return y # Return (N, D)
    func.side_effect = call_mock # Use side_effect for __call__
    func.__call__ = call_mock # Also mock __call__ directly if needed
    func.__repr__ = MagicMock(return_value="MockFunction()")
    return func

@pytest.fixture
def mock_data_loader():
    loader = MagicMock(spec=BaseDataLoader)
    X_train = np.array([[0.1], [0.2], [0.3]])
    y_train = np.array([[0.2], [0.4], [0.6]])
    loader.load_data.return_value = (X_train, y_train)
    loader.__repr__ = MagicMock(return_value="MockDataLoader()")
    return loader

@pytest.fixture
def mock_algorithm():
    algo = MagicMock(spec=BaseAlgorithm)
    algo.__repr__ = MagicMock(return_value="MockAlgorithm()")
    # Make predict return predictable shape
    def predict_mock(x):
         # Assume algo output_dim is 1 for mock
         if x.ndim == 1:
             return np.zeros(1) # Single prediction
         else:
              return np.zeros((x.shape[0], 1))
    algo.predict.side_effect = predict_mock
    return algo

@pytest.fixture
def mock_metrics():
    return {"mse": MagicMock(spec=mean_squared_error, return_value=0.5)}

# --- Test Initialization ---

def test_engine_init_success(mock_function, mock_data_loader, mock_algorithm, mock_metrics):
    # Test with function
    engine_f = ExperimentEngine(target_function=mock_function, algorithm=mock_algorithm)
    assert engine_f.target_function == mock_function
    assert engine_f.algorithm == mock_algorithm
    assert engine_f.data_loader is None
    assert "mse" in engine_f.metrics # Default metric
    
    # Test with data loader
    engine_d = ExperimentEngine(data_loader=mock_data_loader, algorithm=mock_algorithm)
    assert engine_d.target_function is None
    assert engine_d.data_loader == mock_data_loader
    assert engine_d.algorithm == mock_algorithm
    
    # Test with both
    engine_b = ExperimentEngine(target_function=mock_function, data_loader=mock_data_loader, algorithm=mock_algorithm)
    assert engine_b.target_function == mock_function
    assert engine_b.data_loader == mock_data_loader
    
    # Test with custom metrics
    engine_m = ExperimentEngine(target_function=mock_function, algorithm=mock_algorithm, metrics=mock_metrics)
    assert engine_m.metrics == mock_metrics

def test_engine_init_failure(mock_function, mock_data_loader, mock_algorithm):
    with pytest.raises(ValueError, match="Either target_function or data_loader must be provided"):
        ExperimentEngine(algorithm=mock_algorithm)
    with pytest.raises(ValueError, match="An algorithm instance must be provided"):
        ExperimentEngine(target_function=mock_function)

# --- Test _get_data --- 
# (Tested implicitly via run, but can add specific tests if needed)

# --- Test run --- 

def test_engine_run_with_function(mock_function, mock_algorithm, mock_metrics):
    engine = ExperimentEngine(target_function=mock_function, algorithm=mock_algorithm, metrics=mock_metrics, train_samples=10, test_samples=5)
    results = engine.run()
    
    # Check algorithm calls
    mock_algorithm.train.assert_called_once()
    mock_algorithm.predict.assert_called_once()
    
    # Check train data passed to algorithm.train
    train_call_args = mock_algorithm.train.call_args
    assert train_call_args[1]['target_function'] == mock_function
    np.testing.assert_array_equal(train_call_args[1]['X_data'], np.linspace(-1, 1, 10).reshape(10, 1))
    # y_data should be approximately X_data * 2 based on mock function
    np.testing.assert_allclose(train_call_args[1]['y_data'], (np.linspace(-1, 1, 10)*2).reshape(10, 1))

    # Check test data passed to algorithm.predict
    predict_call_args = mock_algorithm.predict.call_args
    np.testing.assert_array_equal(predict_call_args[0][0], np.linspace(-1, 1, 5).reshape(5, 1))
    
    # Check metric calls
    mock_metrics["mse"].assert_called_once()
    metric_call_args = mock_metrics["mse"].call_args
    np.testing.assert_allclose(metric_call_args[0][0], (np.linspace(-1, 1, 5)*2).reshape(5, 1)) # y_test
    np.testing.assert_allclose(metric_call_args[0][1], np.zeros((5, 1))) # Mocked prediction
    
    # Check results structure
    assert "start_time" in results
    assert "end_time" in results
    assert "total_time" in results
    assert "data_load_time" in results
    assert "train_time" in results
    assert "evaluation_time" in results
    assert results["train_data_shape"] == (10, 1)
    assert results["test_data_shape"] == (5, 1)
    assert results["evaluation_scores"] == {"mse": 0.5}
    assert "error" not in results

def test_engine_run_with_loader_no_function(mock_data_loader, mock_algorithm, mock_metrics):
    engine = ExperimentEngine(data_loader=mock_data_loader, algorithm=mock_algorithm, metrics=mock_metrics)
    results = engine.run()
    
    # Check algorithm calls
    mock_algorithm.train.assert_called_once()
    mock_algorithm.predict.assert_called_once()
    
    # Check train data passed to algorithm.train (from loader)
    X_train_expected = np.array([[0.1], [0.2], [0.3]])
    y_train_expected = np.array([[0.2], [0.4], [0.6]])
    train_call_args = mock_algorithm.train.call_args
    assert train_call_args[1]['target_function'] is None
    np.testing.assert_array_equal(train_call_args[1]['X_data'], X_train_expected)
    np.testing.assert_array_equal(train_call_args[1]['y_data'], y_train_expected)
    
    # Check test data passed to algorithm.predict (should be same as train data)
    predict_call_args = mock_algorithm.predict.call_args
    np.testing.assert_array_equal(predict_call_args[0][0], X_train_expected)
    
    # Check metric calls (tested on training data)
    mock_metrics["mse"].assert_called_once()
    metric_call_args = mock_metrics["mse"].call_args
    np.testing.assert_array_equal(metric_call_args[0][0], y_train_expected) # y_test = y_train
    np.testing.assert_allclose(metric_call_args[0][1], np.zeros((3, 1))) # Mocked prediction

    # Check results
    assert results["train_data_shape"] == (3, 1)
    assert results["test_data_shape"] == (3, 1) # Same as train
    assert results["evaluation_scores"] == {"mse": 0.5}
    assert "error" not in results

def test_engine_run_with_loader_and_function(mock_function, mock_data_loader, mock_algorithm, mock_metrics):
    # Function should take priority for sampling train/test data
    engine = ExperimentEngine(target_function=mock_function, data_loader=mock_data_loader, 
                              algorithm=mock_algorithm, metrics=mock_metrics,
                              train_samples=8, test_samples=4)
    results = engine.run()
    
    # Check basics of what we care about
    mock_data_loader.load_data.assert_not_called() # Loader should not be used
    mock_function.sample_domain.assert_called() # Function sampler used
    assert mock_function.sample_domain.call_count >= 2  # Called for both train & test

    # Check that algorithm was called with expected data
    mock_algorithm.train.assert_called_once()
    mock_algorithm.predict.assert_called_once()
    
    # Check results structure
    assert results["train_data_shape"] == (8, 1)
    assert results["test_data_shape"] == (4, 1)
    assert "error" not in results

def test_engine_run_handles_train_error(mock_function, mock_algorithm):
    mock_algorithm.train.side_effect = ValueError("Training failed!")
    engine = ExperimentEngine(target_function=mock_function, algorithm=mock_algorithm)
    results = engine.run()
    
    mock_algorithm.train.assert_called_once()
    mock_algorithm.predict.assert_not_called() # Should not predict if train fails
    assert "error" in results
    assert results["error"] == "Training failed!"
    assert "evaluation_scores" not in results # Should not be added if error occurs before eval

def test_engine_run_handles_predict_error(mock_function, mock_algorithm, mock_metrics):
    mock_algorithm.predict.side_effect = ValueError("Predict failed!")
    engine = ExperimentEngine(target_function=mock_function, algorithm=mock_algorithm, metrics=mock_metrics)
    results = engine.run()
    
    mock_algorithm.train.assert_called_once()
    mock_algorithm.predict.assert_called_once()
    mock_metrics["mse"].assert_not_called() # Should not call metrics if predict fails
    assert "error" in results
    assert results["error"] == "Predict failed!"
    assert results["evaluation_scores"] is None # Eval scores should be None

def test_engine_run_handles_metric_error(mock_function, mock_algorithm, mock_metrics):
    mock_metrics["mse"].side_effect = ValueError("Metric failed!")
    engine = ExperimentEngine(target_function=mock_function, algorithm=mock_algorithm, metrics=mock_metrics)
    results = engine.run()
    
    mock_algorithm.train.assert_called_once()
    mock_algorithm.predict.assert_called_once()
    mock_metrics["mse"].assert_called_once() # Metric is called
    assert "error" not in results # Experiment itself doesn't fail
    assert results["evaluation_scores"] == {"mse": None} # Score for the failed metric is None

def test_engine_repr(mock_function, mock_algorithm):
    engine = ExperimentEngine(target_function=mock_function, algorithm=mock_algorithm)
    repr_str = repr(engine)
    assert "ExperimentEngine" in repr_str
    assert "target=MockFunction()" in repr_str
    assert "algorithm=MockAlgorithm()" in repr_str 