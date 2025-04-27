import pytest
import numpy as np

import sys
import os
# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from algogym.evaluation.metrics import mean_squared_error, mean_absolute_error

# --- Test mean_squared_error ---

@pytest.mark.parametrize("y_true, y_pred, expected_mse", [
    (np.array([1, 2, 3]), np.array([1, 2, 3]), 0.0), # Perfect match
    (np.array([1, 2, 3]), np.array([2, 3, 4]), 1.0), # ((1^2 + 1^2 + 1^2) / 3) = 1
    (np.array([1, 2, 3]), np.array([0, 1, 2]), 1.0), # ((-1)^2 + (-1)^2 + (-1)^2) / 3 = 1
    (np.array([1, 2, 3]), np.array([1, 1, 4]), (0**2 + (-1)**2 + 1**2)/3 ), # (0 + 1 + 1) / 3 = 2/3
    # 2D Output
    (np.array([[1, 10], [2, 20]]), np.array([[1, 10], [2, 20]]), 0.0),
    (np.array([[1, 10], [2, 20]]), np.array([[0, 11], [3, 21]]), (( (-1)**2 + 1**2 ) + ( 1**2 + 1**2 )) / 4 ), # (1+1+1+1)/4 = 1
])
def test_mean_squared_error_calculation(y_true, y_pred, expected_mse):
    mse = mean_squared_error(y_true, y_pred)
    assert mse == pytest.approx(expected_mse)
    assert isinstance(mse, float)

def test_mean_squared_error_shape_mismatch():
    y_true = np.array([1, 2, 3])
    y_pred_wrong_shape = np.array([1, 2])
    y_pred_wrong_dim = np.array([[1], [2], [3]])
    
    with pytest.raises(ValueError, match="Shape mismatch"):
        mean_squared_error(y_true, y_pred_wrong_shape)
    with pytest.raises(ValueError, match="Shape mismatch"):
        mean_squared_error(y_true, y_pred_wrong_dim)
        
# --- Test mean_absolute_error ---

@pytest.mark.parametrize("y_true, y_pred, expected_mae", [
    (np.array([1, 2, 3]), np.array([1, 2, 3]), 0.0), # Perfect match
    (np.array([1, 2, 3]), np.array([2, 3, 4]), 1.0), # (|1| + |1| + |1|) / 3 = 1
    (np.array([1, 2, 3]), np.array([0, 1, 2]), 1.0), # (|-1| + |-1| + |-1|) / 3 = 1
    (np.array([1, 2, 3]), np.array([1, 1, 4]), (np.abs(0) + np.abs(-1) + np.abs(1))/3 ), # Fixed calculation (0 + 1 + 1) / 3 = 2/3
    # 2D Output
    (np.array([[1, 10], [2, 20]]), np.array([[1, 10], [2, 20]]), 0.0),
    (np.array([[1, 10], [2, 20]]), np.array([[0, 11], [3, 21]]), ( (np.abs(-1) + np.abs(1)) + (np.abs(1) + np.abs(1)) ) / 4 ), # Fixed calculation (1+1+1+1)/4 = 1
])
def test_mean_absolute_error_calculation(y_true, y_pred, expected_mae):
    mae = mean_absolute_error(y_true, y_pred)
    assert mae == pytest.approx(expected_mae)
    assert isinstance(mae, float)

def test_mean_absolute_error_shape_mismatch():
    y_true = np.array([1, 2, 3])
    y_pred_wrong_shape = np.array([1, 2])
    y_pred_wrong_dim = np.array([[1], [2], [3]])
    
    with pytest.raises(ValueError, match="Shape mismatch"):
        mean_absolute_error(y_true, y_pred_wrong_shape)
    with pytest.raises(ValueError, match="Shape mismatch"):
        mean_absolute_error(y_true, y_pred_wrong_dim) 