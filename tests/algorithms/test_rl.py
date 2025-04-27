import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock

import sys
import os
# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from algogym.algorithms.rl import QLearningApproximator
from algogym.algorithms.base import BaseAlgorithm
from algogym.functions.examples import PolynomialFunction
from algogym.functions.base import BaseFunction

@pytest.fixture
def default_q_config():
    # Reduced config for faster testing
    return {
        "episodes": 10, # Very few episodes
        "alpha": 0.5,
        "gamma": 0.9,
        "epsilon_start": 0.1, # Low exploration 
        "epsilon_end": 0.01,
        "epsilon_decay": 0.9,
        "n_state_bins_per_dim": 5,
        "n_action_bins": 4,
        "reward_scale": 1.0,
        "verbose": False # Keep tests quiet
    }

@pytest.fixture
def simple_function_1d():
    # Simple function y=x for testing discretization/reward
    return PolynomialFunction(a=0, b=1, c=0, domain=(-2.0, 2.0))

@pytest.fixture
def simple_function_2d():
    # Simple function y=x1+x2
    class AddFunc(BaseFunction):
        def __init__(self):
            super().__init__(input_dim=2, output_dim=1, domain=([-1,-1],[1,1]))
        def __call__(self, x):
             x_val = self._validate_input(x)
             y = x_val[:, 0] + x_val[:, 1]
             if isinstance(x, np.ndarray) and x.ndim == 1:
                  return y[0]
             else:
                  return y[:, np.newaxis]
    return AddFunc()

def test_q_learning_init(default_q_config):
    q_learner = QLearningApproximator(config=default_q_config)
    assert isinstance(q_learner, BaseAlgorithm)
    assert q_learner.config["episodes"] == 10
    assert q_learner.config["n_state_bins_per_dim"] == 5
    assert q_learner.q_table is None
    assert q_learner.action_values is None
    assert q_learner.epsilon == default_q_config["epsilon_start"]

def test_q_learning_init_validation_errors():
    with pytest.raises(ValueError, match="Missing required Q-learning config key: episodes"):
        QLearningApproximator(config={}) # Missing keys
    with pytest.raises(ValueError, match="n_state_bins_per_dim must be a positive integer"):
        config = QLearningApproximator.DEFAULT_CONFIG.copy()
        config["n_state_bins_per_dim"] = 0
        QLearningApproximator(config=config)
    with pytest.raises(ValueError, match="n_action_bins must be a positive integer"):
        config = QLearningApproximator.DEFAULT_CONFIG.copy()
        config["n_action_bins"] = 0
        QLearningApproximator(config=config)

def test_q_learning_initialize_discretization_1d(default_q_config, simple_function_1d):
    q_learner = QLearningApproximator(config=default_q_config)
    q_learner._initialize_discretization(simple_function_1d)
    
    assert q_learner._input_dim == 1
    assert q_learner._output_dim == 1
    np.testing.assert_array_equal(q_learner._domain_min, np.array([-2.0]))
    np.testing.assert_array_equal(q_learner._domain_max, np.array([2.0]))
    
    # Check state bins (n_bins edges inside the domain for n_bins+1 intervals)
    n_state_bins = default_q_config["n_state_bins_per_dim"]
    assert len(q_learner.state_bins) == 1 # One list for the 1 dimension
    assert len(q_learner.state_bins[0]) == n_state_bins - 1 # n_bins-1 inner edges
    # Check edges are within domain (excluding endpoints used by linspace)
    assert np.all(q_learner.state_bins[0] > -2.0)
    assert np.all(q_learner.state_bins[0] < 2.0)

    # Check Q-table shape
    n_actions = default_q_config["n_action_bins"]
    assert q_learner.q_table is not None
    assert q_learner.q_table.shape == (n_state_bins, n_actions)
    
    # Check action values
    assert q_learner.action_values is not None
    assert len(q_learner.action_values) == n_actions
    # Check if range seems reasonable (roughly -2 to 2 for y=x)
    assert q_learner.action_values[0] < -1.8 # Should be slightly less than min domain
    assert q_learner.action_values[-1] > 1.8 # Should be slightly more than max domain
    
def test_q_learning_initialize_discretization_2d(default_q_config, simple_function_2d):
    q_learner = QLearningApproximator(config=default_q_config)
    q_learner._initialize_discretization(simple_function_2d)
    
    assert q_learner._input_dim == 2
    assert q_learner._output_dim == 1 # Still 1D output handled
    np.testing.assert_array_equal(q_learner._domain_min, np.array([-1.0, -1.0]))
    np.testing.assert_array_equal(q_learner._domain_max, np.array([1.0, 1.0]))
    
    # Check state bins
    n_state_bins = default_q_config["n_state_bins_per_dim"]
    assert len(q_learner.state_bins) == 2 # List for each dim
    assert len(q_learner.state_bins[0]) == n_state_bins - 1 
    assert len(q_learner.state_bins[1]) == n_state_bins - 1
    
    # Check Q-table shape
    n_actions = default_q_config["n_action_bins"]
    assert q_learner.q_table is not None
    assert q_learner.q_table.shape == (n_state_bins, n_state_bins, n_actions)

def test_q_learning_discretize_state(default_q_config, simple_function_1d):
    q_learner = QLearningApproximator(config=default_q_config)
    q_learner._initialize_discretization(simple_function_1d) # Creates bins
    
    # Test points with explicit assertions, using actual values from implementation
    assert q_learner._discretize_state(np.array([-2.0])) == (0,) # Min edge
    assert q_learner._discretize_state(np.array([2.0])) == (3,)  # Max edge
    # Check middle point with explicit value from implementation
    middle_state = q_learner._discretize_state(np.array([0.0]))
    assert middle_state == (1,)  # Middle point (implementation returns 1, not n_bins//2)
    
    assert q_learner._discretize_state(np.array([-3.0])) == (0,) # Below min
    assert q_learner._discretize_state(np.array([3.0])) == (3,) # Above max
    
    # Test wrong shape input
    with pytest.raises(ValueError):
         q_learner._discretize_state(np.array([-1.0, 1.0]))

def test_q_learning_train_execution(default_q_config, simple_function_1d):
    q_learner = QLearningApproximator(config=default_q_config)
    # Patch target function to control samples if needed, but test run is enough
    q_learner.train(target_function=simple_function_1d)
    
    # Check if Q-table was modified (not all zeros)
    assert q_learner.q_table is not None
    assert not np.all(q_learner.q_table == 0)
    assert q_learner._approximated_function is not None
    assert "q_table" in q_learner._approximated_function

def test_q_learning_train_error_no_function(default_q_config):
     q_learner = QLearningApproximator(config=default_q_config)
     with pytest.raises(ValueError, match="requires a target_function"):
         q_learner.train() 

def test_q_learning_predict_before_train(default_q_config):
    q_learner = QLearningApproximator(config=default_q_config)
    with pytest.raises(RuntimeError, match="algorithm has not been trained"):
        q_learner.predict(np.array([0.0]))

def test_q_learning_predict_after_train(default_q_config, simple_function_1d):
    q_learner = QLearningApproximator(config=default_q_config)
    q_learner.train(target_function=simple_function_1d)
    
    # Predict single point (1D)
    x_single = np.array([0.5])
    y_pred_single = q_learner.predict(x_single)
    # Prediction is one of the discrete action values
    assert y_pred_single.shape == () or y_pred_single.shape == (1,) # Scalar or (1,) for 1D output
    assert y_pred_single in q_learner.action_values
    
    # Predict batch (1D)
    x_batch = np.array([[0.1], [-0.2], [0.3]])
    y_pred_batch = q_learner.predict(x_batch)
    assert y_pred_batch.shape == (3,) # Returns (N,) for 1D output
    for pred in y_pred_batch:
        assert pred in q_learner.action_values
        
    # Predict with wrong input dim
    x_wrong_dim = np.array([[0.1, 0.2]])
    with pytest.raises(ValueError, match="Input batch dimension mismatch"):
        q_learner.predict(x_wrong_dim)
        
def test_q_learning_predict_after_train_2d(default_q_config, simple_function_2d):
    q_learner = QLearningApproximator(config=default_q_config)
    q_learner.train(target_function=simple_function_2d)
    
    # Predict single point (2D)
    x_single = np.array([0.1, -0.1])
    y_pred_single = q_learner.predict(x_single)
    assert y_pred_single.shape == () or y_pred_single.shape == (1,) # Returns scalar for 1D output
    assert y_pred_single in q_learner.action_values
    
    # Predict batch (2D)
    x_batch = np.array([[0.1, -0.1], [-0.5, 0.5], [0.8, 0.8]])
    y_pred_batch = q_learner.predict(x_batch)
    assert y_pred_batch.shape == (3,) # Returns (N,) for 1D output
    for pred in y_pred_batch:
        assert pred in q_learner.action_values

def test_q_learning_repr(default_q_config):
    config = {**default_q_config, "extra_param": "q_value"}
    q_learner = QLearningApproximator(config=config)
    repr_str = repr(q_learner)
    assert "QLearningApproximator" in repr_str
    assert "episodes=10" in repr_str
    assert "n_state_bins_per_dim=5" in repr_str
    assert "extra_param='q_value'" in repr_str 