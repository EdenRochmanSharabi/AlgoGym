import pytest
import numpy as np

import sys
import os
# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from algogym.algorithms.neighbors import KNearestNeighbors
from algogym.algorithms.base import BaseAlgorithm

@pytest.fixture
def knn_data():
    # Simple 1D data: y = 2x
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([[2.0], [4.0], [6.0], [8.0], [10.0]])
    return X, y

@pytest.fixture
def knn_data_2d():
     # Simple 2D data: y = x1 + x2
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [0, 0]])
    y = np.array([[2], [3], [3], [4], [0]])
    return X, y


def test_knn_init():
    knn = KNearestNeighbors(config={"k": 5, "metric": "cosine"})
    assert isinstance(knn, BaseAlgorithm)
    assert knn.config["k"] == 5
    assert knn.config["metric"] == "cosine"
    assert knn._X_train is None
    assert knn._y_train is None

def test_knn_init_defaults():
    knn = KNearestNeighbors()
    assert knn.config["k"] == 3
    assert knn.config["metric"] == "euclidean"

def test_knn_init_validation_errors():
    with pytest.raises(ValueError, match="'k' must be a positive integer"):
        KNearestNeighbors(config={"k": 0})
    with pytest.raises(ValueError, match="'k' must be a positive integer"):
        KNearestNeighbors(config={"k": -1})
    with pytest.raises(ValueError, match="'k' must be a positive integer"):
        KNearestNeighbors(config={"k": 3.5})
    # Metric validation is deferred to scipy

def test_knn_train(knn_data):
    X, y = knn_data
    knn = KNearestNeighbors(config={"k": 2})
    knn.train(X_data=X, y_data=y)
    
    np.testing.assert_array_equal(knn._X_train, X)
    np.testing.assert_array_equal(knn._y_train, y)
    assert knn._input_dim == 1
    assert knn._output_dim == 1
    assert knn._approximated_function == (knn._X_train, knn._y_train)

def test_knn_train_warns_k_too_large(knn_data, capsys):
    X, y = knn_data
    k_large = 10
    knn = KNearestNeighbors(config={"k": k_large})
    knn.train(X_data=X, y_data=y)
    
    captured = capsys.readouterr()
    assert f"Warning: k ({k_large}) is greater than the number of training samples ({X.shape[0]})" in captured.out
    assert knn.config["k"] == X.shape[0]

def test_knn_train_errors(knn_data):
    X, y = knn_data
    knn = KNearestNeighbors()
    with pytest.raises(ValueError, match="requires explicit X_data and y_data"):
        knn.train() # No data
    with pytest.raises(ValueError, match="requires explicit X_data and y_data"):
        knn.train(X_data=X) # Missing y
    with pytest.raises(ValueError, match="requires explicit X_data and y_data"):
        knn.train(y_data=y) # Missing X
        
    X_wrong_shape = X[:-1] # Make X shorter than y
    with pytest.raises(ValueError, match="Number of samples mismatch"):
         knn.train(X_data=X_wrong_shape, y_data=y)

def test_knn_predict_before_train():
    knn = KNearestNeighbors()
    with pytest.raises(RuntimeError, match="algorithm has not been trained"):
        knn.predict(np.array([1.0]))
        
@pytest.mark.parametrize("k, x_pred, expected_y", [
    (1, np.array([[2.1]]), 4.0), # Closest to 2.0 -> y=4.0
    (1, np.array([[2.9]]), 6.0), # Closest to 3.0 -> y=6.0
    (2, np.array([[1.1]]), 3.0), # Closest to 1, 2 -> avg(2, 4) = 3
    (2, np.array([[4.9]]), 9.0), # Closest to 5, 4 -> avg(10, 8) = 9
    (3, np.array([[2.5]]), 6.0), # Closest to 2, 3, 1 or 4? (1, 2, 3 or 2, 3, 4) -> avg(4,6,2)=4 or avg(4,6,8)=6. Let's check: dists are 1.5, 0.5, 0.5, 1.5, 2.5. Closest are 2,3,4. avg(4,6,8)=6
])        
def test_knn_predict_1d(knn_data, k, x_pred, expected_y):
    X, y = knn_data
    knn = KNearestNeighbors(config={"k": k})
    knn.train(X_data=X, y_data=y)
    y_pred = knn.predict(x_pred)
    assert y_pred.shape == (1, 1) # Check shape before comparing value
    assert y_pred[0,0] == pytest.approx(expected_y)

@pytest.mark.parametrize("k, x_pred_batch, expected_y_batch", [
    (1, np.array([[2.1], [2.9]]), np.array([[4.0], [6.0]])),
    (2, np.array([[1.1], [4.9]]), np.array([[3.0], [9.0]])),
])
def test_knn_predict_1d_batch(knn_data, k, x_pred_batch, expected_y_batch):
    X, y = knn_data
    knn = KNearestNeighbors(config={"k": k})
    knn.train(X_data=X, y_data=y)
    y_pred = knn.predict(x_pred_batch)
    np.testing.assert_allclose(y_pred, expected_y_batch)
    
@pytest.mark.parametrize("k, x_pred, expected_y", [
    (1, np.array([[1.1, 1.1]]), 2.0), # Closest to [1,1] -> y=2
    (1, np.array([[1.9, 2.1]]), 4.0), # Closest to [2,2] -> y=4
    (2, np.array([[0.1, 0.1]]), 1.0), # Closest to [0,0] and [1,1]? dists: 0.14, 1.4, 1.4, 2.8, 0 -> [0,0],[1,1]. avg(0,2)=1
    (3, np.array([[1.5, 1.5]]), 2.6666666666666665), # Closest to [1,1],[1,2],[2,1]? dists: 0.7, 0.7, 0.7, 1, 2.1 -> [1,1],[1,2],[2,1]. avg(2,3,3)=8/3=2.66...
]) 
def test_knn_predict_2d(knn_data_2d, k, x_pred, expected_y):
    X, y = knn_data_2d
    knn = KNearestNeighbors(config={"k": k})
    knn.train(X_data=X, y_data=y)
    y_pred = knn.predict(x_pred)
    assert y_pred.shape == (1, 1) 
    assert y_pred[0,0] == pytest.approx(expected_y)

def test_knn_predict_shape_handling(knn_data):
    X, y = knn_data # Input dim 1, Output dim 1
    knn = KNearestNeighbors(config={"k": 1})
    knn.train(X_data=X, y_data=y)
    
    # Single scalar-like input (still treated as 1 sample)
    # Pass as 1D array
    y_pred_single = knn.predict(np.array([2.1])) 
    assert y_pred_single.shape == (1, 1), f"Expected shape (1, 1) for single 1D input, got {y_pred_single.shape}"
    assert y_pred_single[0, 0] == pytest.approx(4.0) # Closest to 2.0 -> y=4.0

    # Batch input (N, 1)
    y_pred_batch = knn.predict(np.array([[2.1], [2.9]])) 
    assert y_pred_batch.shape == (2, 1)
    np.testing.assert_allclose(y_pred_batch, np.array([[4.0], [6.0]]))
    
    # Predict with wrong input dim
    x_wrong_dim = np.array([[0.1, 0.2]])
    with pytest.raises(ValueError, match="Input batch dimension mismatch"):
        knn.predict(x_wrong_dim)
        
def test_knn_repr():
    knn = KNearestNeighbors(config={"k": 7, "metric": "cityblock"})
    repr_str = repr(knn)
    assert "KNearestNeighbors" in repr_str
    assert "k=7" in repr_str
    assert "metric='cityblock'" in repr_str 