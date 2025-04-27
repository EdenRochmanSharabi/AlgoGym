import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
import os
# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from algogym.functions.base import BaseFunction
from algogym.functions.examples import PolynomialFunction
from algogym.data.loaders import FunctionSampler, CsvLoader
from algogym.data.base import BaseDataLoader # For type checking

# --- Test FunctionSampler ---

def test_function_sampler_init():
    func = PolynomialFunction()
    sampler = FunctionSampler(function=func, n_samples=100)
    assert isinstance(sampler, BaseDataLoader)
    assert sampler.function == func
    assert sampler.n_samples == 100

def test_function_sampler_init_errors():
    func = PolynomialFunction()
    with pytest.raises(TypeError, match="must be an instance of BaseFunction"):
        FunctionSampler(function="not a function", n_samples=10)
    with pytest.raises(ValueError, match="must be a positive integer"):
        FunctionSampler(function=func, n_samples=0)
    with pytest.raises(ValueError, match="must be a positive integer"):
        FunctionSampler(function=func, n_samples=-5)

@pytest.mark.parametrize("input_dim, output_dim, n_samples", [
    (1, 1, 50),
    (2, 1, 20),
    (1, 3, 30),
    (3, 2, 10),
])
def test_function_sampler_load_data(input_dim, output_dim, n_samples):
    # Use a simple function for testing shapes
    class TestFunc(BaseFunction):
        def __init__(self, input_dim, output_dim):
            domain = (np.zeros(input_dim), np.ones(input_dim)) if input_dim > 1 else (0, 1)
            super().__init__(input_dim=input_dim, output_dim=output_dim, domain=domain)
        def __call__(self, x):
             x_val = self._validate_input(x) # Shape (N, D)
             y = np.ones((x_val.shape[0], self.output_dim))
             # Handle return shape - ALWAYS return batch shape (N, output_dim) for sampler tests
             # The sampler calls the function with a batch, so we expect a batch back.
             return y
             
    func = TestFunc(input_dim=input_dim, output_dim=output_dim)
    sampler = FunctionSampler(function=func, n_samples=n_samples)
    X, y = sampler.load_data()

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    
    expected_X_shape = (n_samples, input_dim)
    expected_y_shape = (n_samples, output_dim)
    
    assert X.shape == expected_X_shape, f"Expected X shape {expected_X_shape}, got {X.shape}"
    assert y.shape == expected_y_shape, f"Expected y shape {expected_y_shape}, got {y.shape}"
    
    # Check if X values are within the domain [0, 1]
    if input_dim == 1:
         assert np.all((X >= 0) & (X <= 1))
    else:
         assert np.all((X >= 0) & (X <= 1))


def test_function_sampler_repr():
    func = PolynomialFunction(a=2, b=1, c=0)
    sampler = FunctionSampler(function=func, n_samples=55)
    repr_str = repr(sampler)
    assert "FunctionSampler" in repr_str
    assert "PolynomialFunction" in repr_str # Check function repr is included
    assert "n_samples=55" in repr_str
    assert "a=2" in repr_str # Check function params are in its repr

# --- Test CsvLoader ---

@pytest.fixture
def create_dummy_csv(tmp_path):
    """Creates a dummy CSV file in a temporary directory."""
    csv_path = tmp_path / "test_data.csv"
    data = {
        'input1': [1, 2, 3, 4, 5],
        'input2': [10, 20, 30, 40, 50],
        'target1': [11, 22, 33, 44, 55],
        'target2': [-1, -2, -3, -4, -5],
        'extra': ['a', 'b', 'c', 'd', 'e']
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path

def test_csv_loader_init():
    loader = CsvLoader(filepath="dummy.csv", input_cols=['in1', 'in2'], target_cols=['out'])
    assert isinstance(loader, BaseDataLoader)
    assert loader.filepath == "dummy.csv"
    assert loader.input_cols == ['in1', 'in2']
    assert loader.target_cols == ['out']

    loader_no_target = CsvLoader(filepath="dummy.csv", input_cols=['in1'])
    assert loader_no_target.target_cols is None

def test_csv_loader_load_data_success(create_dummy_csv):
    csv_path = create_dummy_csv
    
    # Test with targets
    loader_targets = CsvLoader(filepath=str(csv_path), input_cols=['input1', 'input2'], target_cols=['target1', 'target2'])
    X, y = loader_targets.load_data()
    
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (5, 2)
    assert y.shape == (5, 2)
    np.testing.assert_array_equal(X[:, 0], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(X[:, 1], [10, 20, 30, 40, 50])
    np.testing.assert_array_equal(y[:, 0], [11, 22, 33, 44, 55])
    np.testing.assert_array_equal(y[:, 1], [-1, -2, -3, -4, -5])

    # Test without targets
    loader_no_targets = CsvLoader(filepath=str(csv_path), input_cols=['input1'])
    X_no_tgt, y_no_tgt = loader_no_targets.load_data()
    assert isinstance(X_no_tgt, np.ndarray)
    assert y_no_tgt is None
    assert X_no_tgt.shape == (5, 1)
    np.testing.assert_array_equal(X_no_tgt[:, 0], [1, 2, 3, 4, 5])

def test_csv_loader_load_data_errors(create_dummy_csv):
    csv_path = create_dummy_csv
    
    # File not found
    loader_bad_path = CsvLoader(filepath="nonexistent.csv", input_cols=['input1'])
    with pytest.raises(FileNotFoundError, match="CSV file not found"):
        loader_bad_path.load_data()
        
    # Missing input column
    loader_missing_in = CsvLoader(filepath=str(csv_path), input_cols=['input1', 'missing_in'], target_cols=['target1'])
    with pytest.raises(ValueError, match="Input columns not found in CSV: \['missing_in'\]"):
        loader_missing_in.load_data()

    # Missing target column
    loader_missing_tgt = CsvLoader(filepath=str(csv_path), input_cols=['input1'], target_cols=['target1', 'missing_tgt'])
    with pytest.raises(ValueError, match="Target columns not found in CSV: \['missing_tgt'\]"):
        loader_missing_tgt.load_data()

def test_csv_loader_repr():
    loader = CsvLoader(filepath="path/to/data.csv", input_cols=['a'], target_cols=['b'])
    repr_str = repr(loader)
    assert "CsvLoader" in repr_str
    assert "filepath='path/to/data.csv'" in repr_str
    assert "input_cols=['a']" in repr_str
    assert "target_cols=['b']" in repr_str

    loader_no_tgt = CsvLoader(filepath="data.csv", input_cols=['x'])
    repr_str_no_tgt = repr(loader_no_tgt)
    assert "target_cols=None" in repr_str_no_tgt 