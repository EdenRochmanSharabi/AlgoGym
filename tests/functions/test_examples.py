import pytest
import numpy as np
import re

import sys
import os
# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from algogym.functions.examples import SineFunction, PolynomialFunction, RosenbrockFunction

# --- Test SineFunction ---

def test_sine_function_call():
    func = SineFunction()
    # Test scalar input
    assert func(0.0) == pytest.approx(0.0)
    assert func(np.pi / 2) == pytest.approx(1.0)
    assert func(-np.pi / 2) == pytest.approx(-1.0)
    assert func(np.pi) == pytest.approx(0.0)
    # Test array input
    x = np.array([0, np.pi / 2, np.pi])
    y_expected = np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(func(x), y_expected, atol=1e-7)
    # Test input outside default domain raises error
    with pytest.raises(ValueError, match="is outside the domain"):
        func(3 * np.pi)

def test_sine_function_repr():
    func = SineFunction(domain=(-np.pi, np.pi))
    repr_str = repr(func)
    assert "SineFunction" in repr_str
    # Use regex to allow for slight floating point variations
    assert re.search(r"domain=\(array\(\[-?3.141592\d*\]\), array\(\[3.141592\d*\]\)\)", repr_str)

# --- Test PolynomialFunction ---

def test_polynomial_function_call():
    # Default: f(x) = x^2 - 2x + 1 = (x-1)^2
    func_default = PolynomialFunction()
    assert func_default(1.0) == pytest.approx(0.0)
    assert func_default(0.0) == pytest.approx(1.0)
    assert func_default(2.0) == pytest.approx(1.0)
    assert func_default(3.0) == pytest.approx(4.0)
    
    # Custom: f(x) = 2x^2 + 3x - 5
    func_custom = PolynomialFunction(a=2, b=3, c=-5, domain=(-10, 10))
    assert func_custom(0.0) == pytest.approx(-5.0)
    assert func_custom(1.0) == pytest.approx(2 + 3 - 5)
    assert func_custom(-1.0) == pytest.approx(2 - 3 - 5)
    x = np.array([0, 1, -1])
    y_expected = np.array([-5.0, 0.0, -6.0])
    np.testing.assert_allclose(func_custom(x), y_expected)
     # Test input outside custom domain raises error
    with pytest.raises(ValueError, match="is outside the domain"):
        func_custom(11.0)

def test_polynomial_function_repr():
    func = PolynomialFunction(a=1.5, b=0, c=-2.5, domain=(0, 1))
    repr_str = repr(func)
    assert "PolynomialFunction" in repr_str
    assert "a=1.5" in repr_str
    assert "b=0" in repr_str
    assert "c=-2.5" in repr_str
    # Use regex for float comparison
    assert re.search(r"domain=\(array\(\[0\.?0*\]\), array\(\[1\.?0*\]\)\)", repr_str)
    
# --- Test RosenbrockFunction ---

def test_rosenbrock_function_call():
    # 2D (default)
    func_2d = RosenbrockFunction()
    # Minimum at (1, 1)
    assert func_2d(np.array([1.0, 1.0])) == pytest.approx(0.0)
    assert func_2d(np.array([0.0, 0.0])) == pytest.approx(1.0) # (1-0)^2 + 100*(0-0^2)^2 = 1
    # Adjusted input to be within default domain [-2, 2]
    assert func_2d(np.array([2.0, 2.0])) == pytest.approx(401.0) # (1-2)^2 + 100*(2-2^2)^2 = 1 + 100*(-2)^2 = 1 + 400 = 401
    
    # Test batch input (2D)
    # Adjusted input to be within default domain [-2, 2]
    x_batch_2d = np.array([[1.0, 1.0], [0.0, 0.0], [2.0, 2.0]])
    y_expected_2d = np.array([0.0, 1.0, 401.0])
    np.testing.assert_allclose(func_2d(x_batch_2d), y_expected_2d)

    # 3D
    func_3d = RosenbrockFunction(input_dim=3)
    # Minimum at (1, 1, 1)
    assert func_3d(np.array([1.0, 1.0, 1.0])) == pytest.approx(0.0)
    # f(0,0,0) = [(1-0)^2 + 100(0-0^2)^2] + [(1-0)^2 + 100(0-0^2)^2] = 1 + 1 = 2
    assert func_3d(np.array([0.0, 0.0, 0.0])) == pytest.approx(2.0)
    
    # Test input outside default domain raises error (default scale is 2)
    # Adjusted regex to match actual error message format and handle whitespace
    with pytest.raises(ValueError, match=r"Input point \[\s*2\.1\s+1\.\s*\] at index 0 is outside the domain \[\s*\[-2\.\s+-2\.\]\,\s*\[\s*2\.\s+2\.\]\s*\]\. Violation at dimension 0: value 2\.1 is outside \[-2\.0, 2\.0\]"):
        func_2d(np.array([2.1, 1.0]))

    # Test call with invalid dimension
    with pytest.raises(ValueError, match="requires input_dim >= 2"):
        RosenbrockFunction(input_dim=1)

def test_rosenbrock_function_repr():
    func = RosenbrockFunction(input_dim=4, domain_scale=5.0)
    repr_str = repr(func)
    assert "RosenbrockFunction" in repr_str
    assert "input_dim=4" in repr_str
    assert "domain=(array([-5., -5., -5., -5.]), array([5., 5., 5., 5.]))" in repr_str 