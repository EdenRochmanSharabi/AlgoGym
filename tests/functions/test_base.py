import pytest
import numpy as np
from typing import Union

import sys
import os
# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from algogym.functions.base import BaseFunction

# Concrete implementation for testing BaseFunction
class ConcreteFunction(BaseFunction):
    def __init__(self, input_dim=1, output_dim=1, domain=(-1.0, 1.0)):
        super().__init__(input_dim=input_dim, output_dim=output_dim, domain=domain)

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x_validated = self._validate_input(x) # Shape (N, D)
        # Simple identity function for testing purposes
        # x_validated is always (N, D), even if N=1 or D=1.
        
        # Calculate the result based on input_dim and output_dim
        # Use first column regardless of input_dim for this simple function
        if x_validated.ndim == 1:
            # Handle case for 1D array
            base_result = x_validated
        else:
            base_result = x_validated[:, 0] # Shape (N,)

        # Replicate/reshape for output_dim
        if self.output_dim == 1:
            if x_validated.ndim == 1:
                result_batch = base_result.reshape(-1, 1)
            else:
                result_batch = base_result[:, np.newaxis] # Ensure (N, 1) 
        else:
            # Tile the base result to match output dimension
            if x_validated.ndim == 1:
                result_batch = np.tile(base_result.reshape(-1, 1), (1, self.output_dim))
            else:
                result_batch = np.tile(base_result[:, np.newaxis], (1, self.output_dim)) # Shape (N, output_dim)

        # Handle return shape based on original input type
        if isinstance(x, (int, float)) or (isinstance(x, np.ndarray) and x.ndim < self.input_dim):
            # Input was scalar or single point (e.g., float for 1D, (D,) array for >1D)
            # Note: Check ndim < input_dim to catch single D-dim points (ndim=1 when D>1)
            if self.output_dim == 1:
                 # If input was single point and output is 1D, return scalar
                 # Check result_batch shape - if N=1, return scalar
                 return result_batch[0, 0] if result_batch.shape[0] == 1 else result_batch.flatten() 
            else:
                # If input was single point and output is multi-D, return (output_dim,) array
                return result_batch[0] if result_batch.shape[0] == 1 else result_batch
        else: # Input was 2D batch (N, D) or 1D batch (N,) for 1D input
            if self.output_dim == 1:
                # Return (N, 1) or (N,) depending on test expectations? 
                # Let's stick to (N, 1) as BaseFunction docstring implies batch out if batch in.
                return result_batch # Return (N, 1)
            else:
                 return result_batch # Return (N, output_dim)

# --- Test Initialization and Properties ---

def test_basefunction_init_success():
    func_1d = ConcreteFunction(input_dim=1, output_dim=1, domain=(-2.0, 2.0))
    assert func_1d.input_dim == 1
    assert func_1d.output_dim == 1
    np.testing.assert_array_equal(func_1d.domain[0], np.array([-2.0]))
    np.testing.assert_array_equal(func_1d.domain[1], np.array([2.0]))

    func_2d = ConcreteFunction(input_dim=2, output_dim=3, domain=([-1.0, -2.0], [1.0, 2.0]))
    assert func_2d.input_dim == 2
    assert func_2d.output_dim == 3
    np.testing.assert_array_equal(func_2d.domain[0], np.array([-1.0, -2.0]))
    np.testing.assert_array_equal(func_2d.domain[1], np.array([1.0, 2.0]))

@pytest.mark.parametrize("input_dim, output_dim, domain, error_msg", [
    (0, 1, (-1, 1), "Input dimension must be a positive integer"),
    (1, 0, (-1, 1), "Output dimension must be a positive integer"),
    (1, 1, (1, -1), "Lower bound must be less than upper bound"),
    (1, 1, (1, 1), "Lower bound must be less than upper bound"),
    (2, 1, ([-1, -1], [0]), "Domain bounds must have shape \(2,\)"),
    (2, 1, ([-1], [0, 0]), "Domain bounds must have shape \(2,\)"),
    (2, 1, ([0, 0], [0, 0]), "All lower bounds must be strictly less"),
    (2, 1, ([1, -1], [0, 0]), "All lower bounds must be strictly less"),
    (1, 1, [-1, 1], "Domain must be a tuple"),
    (1, 1, (-1,), "Domain must be a tuple of length 2"),
])
def test_basefunction_init_failure(input_dim, output_dim, domain, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        ConcreteFunction(input_dim=input_dim, output_dim=output_dim, domain=domain)

# --- Test _validate_input --- 
# Note: _validate_input is implicitly tested via __call__ in the concrete class

@pytest.mark.parametrize("func, x_in, expected_out", [
    (ConcreteFunction(input_dim=1, output_dim=1, domain=(-1, 1)), 0.5, 0.5), # Scalar in, scalar out
    (ConcreteFunction(input_dim=1, output_dim=1, domain=(-1, 1)), np.array([0.1, -0.2]), np.array([[0.1], [-0.2]])), # Array in, array out (N, 1)
    (ConcreteFunction(input_dim=2, output_dim=1, domain=([-1,-1],[1,1])), np.array([0.1, 0.2]), 0.1), # 2D Point in, scalar out 
    (ConcreteFunction(input_dim=2, output_dim=1, domain=([-1,-1],[1,1])), np.array([[0.1, 0.2], [-0.3, 0.4]]), np.array([[0.1], [-0.3]])), # 2D Batch in, 1D array out (N, 1)
    (ConcreteFunction(input_dim=1, output_dim=2, domain=(-1, 1)), 0.5, np.array([0.5, 0.5])), # 1D Scalar in, 2D Point out
    (ConcreteFunction(input_dim=1, output_dim=2, domain=(-1, 1)), np.array([0.1, -0.2]), np.array([[0.1, 0.1], [-0.2, -0.2]])), # 1D Array in, 2D Batch out
    (ConcreteFunction(input_dim=2, output_dim=3, domain=([-1,-1],[1,1])), np.array([0.1, 0.2]), np.array([0.1, 0.1, 0.1])), # 2D Point in, 3D Point out
    (ConcreteFunction(input_dim=2, output_dim=3, domain=([-1,-1],[1,1])), np.array([[0.1, 0.2], [-0.3, 0.4]]), np.array([[0.1, 0.1, 0.1], [-0.3, -0.3, -0.3]])), # 2D Batch in, 3D Batch out
])
def test_basefunction_call_success(func, x_in, expected_out):
    y_out = func(x_in)
    np.testing.assert_array_almost_equal(y_out, expected_out)

@pytest.mark.parametrize("func, x_in, error_type, error_msg_match", [
    # TypeErrors
    (ConcreteFunction(input_dim=1, domain=(-1, 1)), "a string", TypeError, "must be float or 1D NumPy array"),
    (ConcreteFunction(input_dim=2, domain=([-1,-1],[1,1])), 0.5, TypeError, "must be a NumPy array"),
    (ConcreteFunction(input_dim=2, domain=([-1,-1],[1,1])), [0.1, 0.2], TypeError, "must be a NumPy array"),
    # ValueErrors (Dimension)
    (ConcreteFunction(input_dim=1, domain=(-1, 1)), np.array([[0.1],[0.2]]), ValueError, r"must be 1D \(N,\), got 2D"),
    (ConcreteFunction(input_dim=2, domain=([-1,-1],[1,1])), np.array([0.1]), ValueError, "wrong dimension. Expected 2, got 1"),
    (ConcreteFunction(input_dim=2, domain=([-1,-1],[1,1])), np.array([[0.1],[0.2]]), ValueError, "wrong dimension. Expected 2, got 1"),
    (ConcreteFunction(input_dim=2, domain=([-1,-1],[1,1])), np.zeros((2,3,1)), ValueError, r"must be 1D \(D,\) or 2D \(N, D\), got 3D"),
    # ValueErrors (Domain) - Using raw strings and updated patterns
    (ConcreteFunction(input_dim=1, domain=(-1, 1)), 1.1, ValueError, r"Input point 1\.1 at index 0 is outside the domain \[\s*\[-1\.?\],\s*\[1\.?]\s*\]\. Violation at dimension 0: value 1\.1 is outside \[-1(?:\.0)?, 1(?:\.0)?\]"),
    (ConcreteFunction(input_dim=1, domain=(-1, 1)), -1.1, ValueError, r"Input point -1\.1 at index 0 is outside the domain \[\s*\[-1\.?\],\s*\[1\.?]\s*\]\. Violation at dimension 0: value -1\.1 is outside \[-1(?:\.0)?, 1(?:\.0)?\]"),
    (ConcreteFunction(input_dim=1, domain=(-1, 1)), np.array([0.5, 1.1]), ValueError, r"Input point 1\.1 at index 1 is outside the domain \[\s*\[-1\.?\],\s*\[1\.?]\s*\]\. Violation at dimension 0: value 1\.1 is outside \[-1(?:\.0)?, 1(?:\.0)?\]"),
    (ConcreteFunction(input_dim=2, domain=([-1,-1],[1,1])), np.array([0.5, 1.1]), ValueError, r"Input point \[\s*0\.5\s+1\.1\s*\] at index 0 is outside the domain \[\s*\[-1\.\s+-1\.\]\,\s*\[\s*1\.\s+1\.\]\s*\]\. Violation at dimension 1: value 1\.1 is outside \[-1\.0, 1\.0\]"),
    (ConcreteFunction(input_dim=2, domain=([-1,-1],[1,1])), np.array([1.1, 0.5]), ValueError, r"Input point \[\s*1\.1\s+0\.5\s*\] at index 0 is outside the domain \[\s*\[-1\.\s+-1\.\]\,\s*\[\s*1\.\s+1\.\]\s*\]\. Violation at dimension 0: value 1\.1 is outside \[-1\.0, 1\.0\]"),
    (ConcreteFunction(input_dim=2, domain=([-1,-1],[1,1])), np.array([[-0.5,-0.5], [0.5, 1.1]]), ValueError, r"Input point \[\s*0\.5\s+1\.1\s*\] at index 1 is outside the domain \[\s*\[-1\.\s+-1\.\]\,\s*\[\s*1\.\s+1\.\]\s*\]\. Violation at dimension 1: value 1\.1 is outside \[-1\.0, 1\.0\]"),
])
def test_basefunction_call_failure(func, x_in, error_type, error_msg_match):
    with pytest.raises(error_type, match=error_msg_match):
        func(x_in)

# --- Test sample_domain --- 

@pytest.mark.parametrize("func, n_samples", [
    (ConcreteFunction(input_dim=1, domain=(-5, 5)), 10),
    (ConcreteFunction(input_dim=3, domain=([-1,-2,-3],[1,2,3])), 50),
])
def test_basefunction_sample_domain_success(func, n_samples):
    samples = func.sample_domain(n_samples)
    
    expected_shape = (n_samples,) if func.input_dim == 1 else (n_samples, func.input_dim)
    assert samples.shape == expected_shape, f"Expected shape {expected_shape}, got {samples.shape}"
    
    lower, upper = func.domain
    if func.input_dim == 1:
        assert np.all(samples >= lower[0])
        assert np.all(samples <= upper[0])
    else:
        assert np.all(samples >= lower), f"Some samples below lower bound {lower}"
        assert np.all(samples <= upper), f"Some samples above upper bound {upper}"

@pytest.mark.parametrize("func, n_samples, error_type, error_msg", [
    (ConcreteFunction(input_dim=1, domain=(-1, 1)), 0, ValueError, "Number of samples must be a positive integer"),
    (ConcreteFunction(input_dim=1, domain=(-1, 1)), -1, ValueError, "Number of samples must be a positive integer"),
    (ConcreteFunction(input_dim=1, domain=(-1, 1)), 10.5, ValueError, "Number of samples must be a positive integer"),
])
def test_basefunction_sample_domain_failure(func, n_samples, error_type, error_msg):
     with pytest.raises(error_type, match=error_msg):
        func.sample_domain(n_samples)

# --- Test __repr__ ---
def test_basefunction_repr():
    func = ConcreteFunction(input_dim=2, output_dim=1, domain=([-1, 0], [1, 5]))
    repr_str = repr(func)
    assert "ConcreteFunction" in repr_str
    assert "input_dim=2" in repr_str
    assert "output_dim=1" in repr_str
    assert "domain=(array([-1.,  0.]), array([1., 5.]))" in repr_str 